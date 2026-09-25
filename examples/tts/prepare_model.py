"""Download Qwen3-TTS 1.7B VoiceDesign and convert it into ncnn models.

    uv run prepare_model.py [DATA_DIR]

DATA_DIR is `data/` beside this file by default. The checkpoint goes into `DATA_DIR/checkpoint`,
from `Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign` on the Hugging Face Hub at a pinned revision, and
this writes into `DATA_DIR/ncnn`

* `tts_PIECE.ncnn.param`, `tts_PIECE.ncnn.bin` -- each piece below;
* `tts.json` -- every piece's ncnn input and output blobs, and the token ids, sizes and
  sampling settings the steps around the pieces need;
* `tts.npz` -- the codec embeddings those steps look codes up in;
* `text.npy` -- the text embedding table, already through the talker's text projection;
* `tokenizer.json` -- the checkpoint's Qwen2 tokenizer, for `tokenizers` alone;
* `tts.reference.json`, `reference.wav` -- what the PyTorch model made of the example in
  `context_example/`, to check the pieces against.

Qwen3-TTS speaks in frames of 16 codes, 12.5 frames a second. The talker, a 28-layer Qwen3,
reads the instruction and the text and makes each frame's first code; the code predictor, a
5-layer one, makes the other 15 from the talker's hidden state, one after another; and the
codec's decoder turns the frames into a 24 kHz waveform. VoiceDesign is the checkpoint whose
voice is described by the instruction rather than picked from a list. The pieces:

* `talker` -- embeddings to the talker's last hidden states and its logits for the first code;
* `predictor` -- the talker's hidden state and the codes so far, embedded, to the logits of
  every one of the predictor's 15 heads;
* `codec` -- a chunk of frames, their codes looked up and summed, to its waveform.

The talker is autoregressive, so it keeps a KV cache: each attention layer takes its keys and
values so far as two more inputs, and gives back the new positions' own, which the caller
writes into the cache after them. The cache is the caller's, in blocks of positions the mask
hides until they are written: its shape changes once a block rather than once a step, which
the guest's device needs -- a buffer of a new size a step, it goes astray after a hundred or
so. It is plain blobs rather than ncnn's SDPA cache (`7=1`), whose layout is the backend's own
and which, fed back through Python's `Mat`, overruns the host buffer it is read into. The
predictor runs a frame's 16 positions at most, and runs them all again for each code, without
a cache. What goes around the pieces -- the prompt's layout,
sampling, looking codes up, chunking the codec -- is `run_tts.py`'s.

**What ncnn needs changed.** Each piece is rewritten from the qwen-tts modules, with their
weights, so that the graph pnnx traces is one ncnn can run:

* RoPE spelled as pnnx fuses it into `RotaryEmbed`, with the angles an input: the caller gives
  them for the positions it runs, half the head's width, as `RotaryEmbed` reads them;
* the text-only mRoPE of the talker is plain RoPE: its three position ids are the same;
* the attention mask an input, a [queries, keys] matrix of 0 and MASKED, the keys the cache's
  and the new ones;
* the residual stream scaled down for its RMSNorms, as `RESIDUAL_SCALE` says;
* what depends on no input -- the text projection of every token, the codec's codebooks
  through their output projections, SnakeBeta's exponentials -- computed here once.

The lengths vary -- tokens, cache, frames -- so each piece is traced at two, and ncnn takes any.
"""

import json
import math
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from huggingface_hub import snapshot_download
from torch import nn

REPO = "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
REVISION = "5ecdb67327fd37bb2e042aab12ff7391903235d3"

# What a masked attention score becomes: past anything a softmax sees here, so the weight it
# gives is 0 all the same, and inside fp16's range.
MASKED = -1e4

# The codec decodes this many frames at a time, with this many before them as context, as
# qwen-tts' `chunked_decode` does; but 60 frames where it takes 300, whose last blocks' buffers,
# 1920 samples a frame by 96 channels in fp32, are past what the guest's device allocates.
CHUNK, CONTEXT = 60, 25

HERE = Path(__file__).parent


def rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """`rotate_half` RoPE, as pnnx fuses it into `RotaryEmbed`."""
    half = x.shape[-1] // 2
    return x * cos + torch.cat([-x[..., half:], x[..., :half]], dim=-1) * sin


# What the talker's residual stream is scaled by for its RMSNorms. One of its 2048 features runs
# to 5500, whose square ncnn's RMSNorm stores in fp16 in fp16 mode, past its 65504: scaled, and
# the epsilon with it, the norm is the same, and the square fits.
RESIDUAL_SCALE = 1 / 32


def rms(norm: nn.Module, x: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    return F.rms_norm(x * scale, (norm.weight.numel(),), norm.weight, norm.variance_epsilon * scale * scale)


def attention(att, x, cos, sin, mask, heads: int, kv_heads: int, head_dim: int, qk_norm: bool, past=None):
    """Self-attention; with `past`, the cached keys and values go before the new ones, and the
    new ones come back as well."""
    # The length as -1, not read off a shape: a traced shape is a constant.
    q = att.q_proj(x).reshape(1, -1, heads, head_dim)
    k = att.k_proj(x).reshape(1, -1, kv_heads, head_dim)
    if qk_norm:
        q, k = rms(att.q_norm, q), rms(att.k_norm, k)
    q, k = rope(q.transpose(1, 2), cos, sin), rope(k.transpose(1, 2), cos, sin)
    v = att.v_proj(x).reshape(1, -1, kv_heads, head_dim).transpose(1, 2)
    new = k, v
    if past is not None:
        k, v = torch.cat([past[0], k], dim=2), torch.cat([past[1], v], dim=2)
    h = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, enable_gqa=kv_heads != heads)
    out = att.o_proj(h.transpose(1, 2).reshape(1, -1, heads * head_dim))
    return out if past is None else (out, *new)


def qwen3_layer(layer, x, cos, sin, mask, config, past=None):
    heads, kv_heads = config.num_attention_heads, config.num_key_value_heads
    h = rms(layer.input_layernorm, x, RESIDUAL_SCALE)
    a = attention(layer.self_attn, h, cos, sin, mask, heads, kv_heads, config.head_dim, True, past)
    a, cache = (a, ()) if past is None else (a[0], a[1:])
    x = x + a
    return x + layer.mlp(rms(layer.post_attention_layernorm, x, RESIDUAL_SCALE)), cache


# --- The talker and the code predictor ------------------------------------------------------


class Talker(nn.Module):
    """Embeddings, each a text token's and a code's summed, and each layer's cached keys and
    values, to the last hidden states, the logits for the first code of the next frame, and each
    layer's keys and values of the new positions."""

    def __init__(self, talker):
        super().__init__()
        self.config = talker.config
        self.layers = talker.model.layers
        self.norm = talker.model.norm
        self.head = talker.codec_head

    def forward(self, embeds, cos, sin, mask, *past):
        x, caches = embeds, []
        for i, layer in enumerate(self.layers):
            x, cache = qwen3_layer(layer, x, cos, sin, mask, self.config, past[2 * i : 2 * i + 2])
            caches += cache
        x = rms(self.norm, x, RESIDUAL_SCALE)
        return (x, self.head(x), *caches)


class Predictor(nn.Module):
    """The talker's hidden state then the frame's codes so far, embedded at the talker's
    width, to the logits of all 15 heads at each position, one head after another: the caller
    takes the head of the code it is at."""

    def __init__(self, talker):
        super().__init__()
        cp = talker.code_predictor
        self.config = cp.config
        self.proj = cp.small_to_mtp_projection
        self.layers = cp.model.layers
        self.norm = cp.model.norm
        w = torch.cat([h.weight for h in cp.lm_head], dim=0)
        self.heads = nn.Linear(w.shape[1], w.shape[0], bias=False)
        with torch.no_grad():
            self.heads.weight.copy_(w)

    def forward(self, embeds, cos, sin, mask):
        x = self.proj(embeds)
        for layer in self.layers:
            x, _ = qwen3_layer(layer, x, cos, sin, mask, self.config)
        return self.heads(rms(self.norm, x, RESIDUAL_SCALE))


# --- The codec's decoder --------------------------------------------------------------------


class Snake(nn.Module):
    """SnakeBeta, x + sin(x * e^alpha)^2 / e^beta, with its exponentials taken here."""

    def __init__(self, snake):
        super().__init__()
        self.register_buffer("alpha", snake.alpha.detach().exp().reshape(1, -1, 1))
        self.register_buffer("inv_beta", (1.0 / (snake.beta.detach().exp() + snake.no_div_by_zero)).reshape(1, -1, 1))

    def forward(self, x):
        s = torch.sin(x * self.alpha)
        return x + self.inv_beta * (s * s)


def conv(c, x):
    """`CausalConvNet`, stride 1 as all of them here are: padded on the left alone. Its own
    forward pads the right by a length it reads off the shape, which a trace makes a constant."""
    assert c.stride == 1
    return c.conv(F.pad(x, (c.padding, 0)))


def convnext(b, x):
    h = conv(b.dwconv, x).transpose(1, 2)
    h = b.pwconv2(b.act(b.pwconv1(b.norm(h))))
    return x + (b.gamma * h).transpose(1, 2)


def trans_conv(t, x):
    """`CausalTransConvNet`: its right padding cut, at an end reckoned from the length, which
    pnnx keeps; it drops a slice to a constant negative end."""
    x = t.conv(x)
    return x[..., : x.shape[-1] - t.right_pad] if t.right_pad else x


class Codec(nn.Module):
    """A chunk of frames, as the sum of their 16 codes' embeddings through the quantizers'
    output projections, to its waveform: 1920 samples a frame at 24 kHz, in -1..1.

    Each module of the decoder it runs is held once: pnnx takes a module reached by two paths
    for weights passed in, and loses the ops."""

    def __init__(self, decoder):
        super().__init__()
        self.config = decoder.config
        tr = decoder.pre_transformer
        self.pre_conv = decoder.pre_conv
        self.input_proj, self.output_proj = tr.input_proj, tr.output_proj
        self.layers, self.norm = tr.layers, tr.norm
        self.upsample = decoder.upsample
        # the decoder: a conv, blocks of snake, transposed conv and three residual units, snake, conv
        self.first = decoder.decoder[0]
        self.blocks = nn.ModuleList([nn.ModuleList(b.block[1:]) for b in decoder.decoder[1:-2]])
        self.snakes = nn.ModuleList([Snake(b.block[0]) for b in decoder.decoder[1:-2]])
        self.unit_snakes = nn.ModuleList(
            [nn.ModuleList([nn.ModuleList([Snake(u.act1), Snake(u.act2)]) for u in b.block[2:]]) for b in decoder.decoder[1:-2]]
        )
        self.last_snake = Snake(decoder.decoder[-2])
        self.last = decoder.decoder[-1]

    def forward(self, x, cos, sin, mask):
        c = self.config
        h = conv(self.pre_conv, x.transpose(1, 2)).transpose(1, 2)
        h = self.input_proj(h)
        for layer in self.layers:
            a = attention(layer.self_attn, rms(layer.input_layernorm, h), cos, sin, mask, c.num_attention_heads, c.num_key_value_heads, c.head_dim, False)
            h = h + layer.self_attn_layer_scale.scale * a
            h = h + layer.mlp_layer_scale.scale * layer.mlp(rms(layer.post_attention_layernorm, h))
        h = self.output_proj(rms(self.norm, h)).transpose(1, 2)
        for t, block in self.upsample:
            h = convnext(block, trans_conv(t, h))
        h = conv(self.first, h)
        for block, snake, units in zip(self.blocks, self.snakes, self.unit_snakes):
            h = trans_conv(block[0], snake(h))
            for unit, (s1, s2) in zip(block[1:], units):
                h = h + conv(unit.conv2, s2(conv(unit.conv1, s1(h))))
        h = conv(self.last, self.last_snake(h))
        return h.clamp(min=-1, max=1)


# --- The pieces, as they are traced ---------------------------------------------------------


def angles(n: int, head_dim: int, theta: float, start: int = 0):
    """RoPE's cos and sin for positions start..start+n, at the head's full width as the traced
    graph takes them; `RotaryEmbed` reads the first half of each row."""
    inv = 1.0 / theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
    f = torch.outer(torch.arange(start, start + n, dtype=torch.float32), inv)
    e = torch.cat([f, f], dim=-1)
    return e.cos()[None, None], e.sin()[None, None]


def causal(n: int, window: int | None = None) -> torch.Tensor:
    i = torch.arange(n)
    far = (i[None, :] > i[:, None]) | ((i[:, None] - i[None, :] >= window) if window else False)
    return torch.zeros(n, n).masked_fill(far, MASKED)[None, None]


def pieces(model) -> dict:
    """Each piece: its module, what its inputs and outputs are, and example inputs at two
    lengths. Shapes are with the batch of one ncnn drops."""
    g = torch.Generator().manual_seed(0)
    r = lambda *s: torch.randn(*s, generator=g)
    talker = model.talker
    tc, pc = talker.config, talker.code_predictor.config
    dc = model.speech_tokenizer.model.decoder.config

    def predictor_ex(n):
        return (r(1, n, tc.hidden_size) * 0.05, *angles(n, pc.head_dim, pc.rope_theta), causal(n))

    def talker_ex(n, cached):
        # the cache's keys first, of which the last is one not yet written
        mask = torch.cat([torch.zeros(1, 1, n, cached), causal(n)], dim=-1)
        mask[..., cached - 1] = MASKED
        past = [r(1, tc.num_key_value_heads, cached, tc.head_dim) for _ in range(2 * tc.num_hidden_layers)]
        return (r(1, n, tc.hidden_size) * 0.05, *angles(n, tc.head_dim, tc.rope_theta), mask, *past)

    def frames(n):
        return (r(1, n, dc.codebook_dim), *angles(n, dc.head_dim, dc.rope_theta), causal(n, dc.sliding_window))

    rope_about = "RoPE's cos, or sin, of each position, [T, head_dim / 2]"
    mask_about = f"0 where a query sees a key and {MASKED} where not, [T, T]"
    cache_about = f"the cache's first, those not yet written masked, [T, cache + T]"
    layers = tc.num_hidden_layers
    return {
        "codec": dict(
            module=lambda: Codec(model.speech_tokenizer.model.decoder),
            inputs=[
                ("frames", "each frame's codes looked up in `rvq` and summed, [T, 512]"),
                ("cos", rope_about),
                ("sin", rope_about),
                ("mask", f"causal within the sliding window, [T, T]"),
            ],
            outputs=["waveform"],
            example=frames(10),
            example2=frames(13),
            # SnakeBeta scales by up to 360 in the last blocks, which makes fp16 weights'
            # rounding audible: the codec's stay fp32, and run so.
            fp16=False,
        ),
        "predictor": dict(
            module=lambda: Predictor(talker),
            inputs=[
                ("embeds", "the talker's hidden state, then each code so far embedded, [T, 2048]"),
                ("cos", rope_about),
                ("sin", rope_about),
                ("mask", mask_about),
            ],
            outputs=["logits"],
            example=predictor_ex(2),
            example2=predictor_ex(3),
        ),
        "talker": dict(
            module=lambda: Talker(talker),
            inputs=[
                ("embeds", "each position's text embedding plus its code's, [T, 2048]"),
                ("cos", rope_about),
                ("sin", rope_about),
                ("mask", f"0 where a query sees a key and {MASKED} where not, {cache_about}"),
                *[(f"past_{kv}{i}", f"layer {i}'s cached {'keys' if kv == 'k' else 'values'}, [8, cache, 128]") for i in range(layers) for kv in "kv"],
            ],
            outputs=["hidden", "logits", *[f"new_{kv}{i}" for i in range(layers) for kv in "kv"]],
            example=talker_ex(3, 1),
            example2=talker_ex(5, 2),
        ),
    }


def export(name: str, spec: dict, ncnn_dir: Path, work: Path) -> dict:
    import pnnx

    stem = f"tts_{name}"
    module = spec["module"]().eval()
    with torch.no_grad():
        pnnx.export(
            module,
            str(work / f"{stem}.pt"),
            spec["example"],
            spec["example2"],
            ncnnparam=str(work / f"{stem}.ncnn.param"),
            ncnnbin=str(work / f"{stem}.ncnn.bin"),
            ncnnpy=str(work / f"{stem}_ncnn.py"),
            pnnxparam=str(work / f"{stem}.pnnx.param"),
            pnnxbin=str(work / f"{stem}.pnnx.bin"),
            pnnxpy=str(work / f"{stem}_pnnx.py"),
            pnnxonnx=str(work / f"{stem}.pnnx.onnx"),
            fp16=spec.get("fp16", True),
        )
    io = {
        "inputs": [{"name": f"in{i}", "role": role, "about": about} for i, (role, about) in enumerate(spec["inputs"])],
        "outputs": [{"name": f"out{i}", "role": role} for i, role in enumerate(spec["outputs"])],
        "fp16": spec.get("fp16", True),
    }
    shutil.move(work / f"{stem}.ncnn.param", ncnn_dir / f"{stem}.ncnn.param")
    # The `.bin` is what says a piece is done, so it appears once its param is in place.
    shutil.move(work / f"{stem}.ncnn.bin", ncnn_dir / f"{stem}.ncnn.bin")
    for f in work.glob(f"{stem}*"):
        f.unlink()
    return io


# --- What the steps around the pieces need --------------------------------------------------


def text_table(model, path: Path):
    """Every text token's embedding through the talker's text projection, in half precision,
    as the pieces' weights are: [151936, 2048]."""
    talker = model.talker
    emb = talker.model.text_embedding.weight
    table = np.lib.format.open_memmap(path.with_suffix(".tmp.npy"), mode="w+", dtype=np.float16, shape=tuple(emb.shape[:1]) + (talker.config.hidden_size,))
    with torch.no_grad():
        for i in range(0, emb.shape[0], 8192):
            table[i : i + 8192] = talker.text_projection(emb[i : i + 8192]).numpy().astype(np.float16)
    table.flush()
    del table
    path.with_suffix(".tmp.npy").rename(path)


def constants(model) -> dict:
    talker = model.talker
    q = model.speech_tokenizer.model.decoder.quantizer

    def codebooks(rvq):
        w = rvq.output_proj.weight.detach()[:, :, 0]  # a 1x1 conv: [512, 256]
        return [(layer._codebook.embedding_sum / layer._codebook.cluster_usage.clamp(min=layer._codebook.epsilon)[:, None]).detach() @ w.T for layer in rvq.vq.layers]

    a = lambda x: x.detach().float().numpy()
    return {
        "codec_embed": a(talker.model.codec_embedding.weight),
        "predictor_embed": np.stack([a(e.weight) for e in talker.code_predictor.model.codec_embedding]).astype(np.float16),
        "rvq": np.stack([a(t) for t in codebooks(q.rvq_first) + codebooks(q.rvq_rest)]),
    }


def settings(model, generate_config: dict) -> dict:
    c = model.config
    tc = c.talker_config
    pc = tc.code_predictor_config
    dc = model.speech_tokenizer.model.decoder.config
    return {
        "masked": MASKED,
        "tokens": {
            "tts_bos": c.tts_bos_token_id,
            "tts_eos": c.tts_eos_token_id,
            "tts_pad": c.tts_pad_token_id,
        },
        "codec": {
            "bos": tc.codec_bos_id,
            "eos": tc.codec_eos_token_id,
            "pad": tc.codec_pad_id,
            "think": tc.codec_think_id,
            "nothink": tc.codec_nothink_id,
            "think_bos": tc.codec_think_bos_id,
            "think_eos": tc.codec_think_eos_id,
            "languages": dict(tc.codec_language_id),
        },
        "talker": {"vocab": tc.vocab_size, "head_dim": tc.head_dim, "kv_heads": tc.num_key_value_heads, "layers": tc.num_hidden_layers, "rope_theta": tc.rope_theta},
        "predictor": {"vocab": pc.vocab_size, "head_dim": pc.head_dim, "rope_theta": pc.rope_theta, "groups": tc.num_code_groups},
        "decoder": {
            "head_dim": dc.head_dim,
            "rope_theta": dc.rope_theta,
            "sliding_window": dc.sliding_window,
            "upsample": int(model.speech_tokenizer.model.decoder.total_upsample),
            "chunk": CHUNK,
            "context": CONTEXT,
            "sample_rate": model.speech_tokenizer.get_output_sample_rate(),
        },
        # The talker never makes the codes above the codebook but its end: they are the prompt's.
        "suppress_from": tc.vocab_size - 1024,
        "min_new_tokens": 2,
        "sampling": generate_config,
    }


# --- Checking ncnn on the host against PyTorch ----------------------------------------------


def cos_sim(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.ravel().astype(np.float64), b.ravel().astype(np.float64)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))


def check(ncnn_dir: Path, specs: dict, ios: dict):
    """Each piece, on its example inputs, against the PyTorch module it was traced from."""
    import ncnn

    for name, spec in specs.items():
        module = spec["module"]().eval()
        net = ncnn.Net()
        net.opt.use_vulkan_compute = ncnn.get_gpu_count() > 0
        net.opt.use_fp16_storage = net.opt.use_fp16_packed = net.opt.use_fp16_arithmetic = False
        net.load_param(str(ncnn_dir / f"tts_{name}.ncnn.param"))
        net.load_model(str(ncnn_dir / f"tts_{name}.ncnn.bin"))
        for example in [spec["example"], spec["example2"]]:
            with torch.no_grad():
                want = module(*example)
                want = want if isinstance(want, tuple) else (want,)
            ex = net.create_extractor()
            x, cos, sin, mask, *past = example
            half = cos.shape[-1] // 2
            for inp, t in zip(ios[name]["inputs"], [x[0], cos[0, 0, :, :half], sin[0, 0, :, :half], mask[0, 0], *(p[0] for p in past)]):
                # Held while it is cloned: `ncnn.Mat` wraps the array, and a temporary is gone by then.
                a = np.ascontiguousarray(t.numpy(), np.float32)
                ex.input(inp["name"], ncnn.Mat(a).clone())
            sims = {}
            for out, w in zip(ios[name]["outputs"], want):
                ret, m = ex.extract(out["name"])
                assert ret == 0, f"{name}: extract {out['name']} failed: {ret}"
                sims[out["role"]] = cos_sim(np.array(m), w.numpy())
            line = [f"{role} {v:.5f}" for role, v in list(sims.items())[:2]]
            if len(sims) > 2:
                line.append(f"cache {min(list(sims.values())[2:]):.5f} at worst")
            print(f"  {name}: " + ", ".join(line), flush=True)
        del module, net


def reference(wrapper, ncnn_dir: Path):
    """What the PyTorch model makes of the example: its codes, sampled with a fixed seed, and
    the log-probability it gave each, so that the pieces can be run on the same codes and
    compared; and the waveform the codec makes of them."""
    import soundfile as sf

    model = wrapper.model
    instruct = (HERE / "context_example" / "instruct.txt").read_text().strip()
    text = (HERE / "context_example" / "text.txt").read_text().strip()
    # The logits the talker and the predictor give, before sampling's processors: what the
    # pieces give. A frame is one talker call, then the predictor's 15 for the frame before it.
    calls = []
    hooks = [
        m.register_forward_hook(lambda _m, _a, out, kind=kind: calls.append((kind, out.logits[0, -1].float().log_softmax(-1))))
        for kind, m in [("talker", model.talker), ("predictor", model.talker.code_predictor)]
    ]
    try:
        torch.manual_seed(0)
        with torch.no_grad():
            codes, _ = model.generate(
                input_ids=wrapper._tokenize_texts([wrapper._build_assistant_text(text)]),
                instruct_ids=wrapper._tokenize_texts([wrapper._build_instruct_text(instruct)]),
                languages=["Auto"],
                non_streaming_mode=True,
                **wrapper._merge_generate_kwargs(),
            )
    finally:
        for h in hooks:
            h.remove()
    codes = codes[0]
    talker = [lp for kind, lp in calls if kind == "talker"]
    predictor = [lp for kind, lp in calls if kind == "predictor"]
    logp = [
        [round(float(talker[f][codes[f, 0]]), 4)] + [round(float(predictor[f * 15 + g][codes[f, g + 1]]), 4) for g in range(15)]
        for f in range(codes.shape[0])
    ]
    with torch.no_grad():
        wavs, sr = model.speech_tokenizer.decode([{"audio_codes": codes}])
    sf.write(ncnn_dir / "reference.wav", wavs[0], sr)
    ref = {"revision": REVISION, "instruct": instruct, "text": text, "codes": codes.tolist(), "logp": logp, "sample_rate": sr}
    (ncnn_dir / "tts.reference.json").write_text(json.dumps(ref) + "\n")
    print(f"  reference: {codes.shape[0]} frames, {len(wavs[0]) / sr:.1f}s", flush=True)


def download(checkpoint: Path):
    """The checkpoint at the repo root; the Hub client skips what is already there."""
    snapshot_download(REPO, revision=REVISION, local_dir=checkpoint)


def main():
    data = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "data"
    checkpoint, ncnn_dir = data / "checkpoint", data / "ncnn"
    work = ncnn_dir / "work"
    ncnn_dir.mkdir(parents=True, exist_ok=True)
    names = ["codec", "predictor", "talker"]
    done = lambda: [n for n in names if (ncnn_dir / f"tts_{n}.ncnn.bin").exists()]
    extras = ["tts.npz", "text.npy", "tokenizer.json", "tts.reference.json", "tts.json"]
    if len(done()) == len(names) and all((ncnn_dir / f).exists() for f in extras):
        print("tts: already converted")
        return
    # Here rather than at the top: it warns of what it cannot find as it is imported.
    from qwen_tts import Qwen3TTSModel

    download(checkpoint)
    wrapper = Qwen3TTSModel.from_pretrained(str(checkpoint), dtype=torch.float32, attn_implementation="sdpa")
    model = wrapper.model.eval()

    if not (ncnn_dir / "tts.reference.json").exists():
        print("tts: taking the reference with PyTorch", flush=True)
        reference(wrapper, ncnn_dir)
    if not (ncnn_dir / "text.npy").exists():
        print("tts: projecting the text embeddings", flush=True)
        text_table(model, ncnn_dir / "text.npy")
    # Not needed past here, and the talker's trace wants the memory.
    model.talker.model.text_embedding = None
    np.savez(ncnn_dir / "tts.npz", **constants(model))
    wrapper.processor.tokenizer.save_pretrained(work / "tokenizer")
    shutil.copy(work / "tokenizer" / "tokenizer.json", ncnn_dir / "tokenizer.json")
    shutil.rmtree(work / "tokenizer")

    specs = pieces(model)
    assert list(specs) == names
    work.mkdir(parents=True, exist_ok=True)
    config = json.loads((ncnn_dir / "tts.json").read_text()) if (ncnn_dir / "tts.json").exists() else {}
    ios = config.get("pieces", {})
    for name in names:
        if name not in done() or name not in ios:
            print(f"tts_{name}: converting", flush=True)
            ios[name] = export(name, specs[name], ncnn_dir.resolve(), work.resolve())
            (ncnn_dir / "tts.json").write_text(json.dumps({"pieces": ios}, indent=2) + "\n")
    shutil.rmtree(work, ignore_errors=True)

    (ncnn_dir / "tts.json").write_text(json.dumps({"pieces": ios, **settings(model, model.generate_config)}, indent=2) + "\n")
    print("checking ncnn on the host against PyTorch", flush=True)
    check(ncnn_dir, specs, ios)


if __name__ == "__main__":
    main()
