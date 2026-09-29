"""Download the Laya checkpoint and convert it into an ncnn model.

    uv run prepare_model.py [DATA_DIR]

DATA_DIR is `data/` beside this file by default. The checkpoint goes into `DATA_DIR/checkpoint`,
from `convaiinnovations/laya` on the Hugging Face Hub at a pinned revision (the English one at
the repo root: ModernBERT-large and Laya's decision head, ~840 MB in fp16), and this writes into
`DATA_DIR/ncnn`

* `laya.ncnn.param`, `laya.ncnn.bin` -- the model, weights in fp16;
* `laya.json` -- its blob names and shapes, and the constants the caller needs (below);
* `laya.embeddings.npy`, `laya.type_embed.npy` -- the lookup tables the caller runs first;
* `laya.act_head.npz` -- the act/escalate head the caller runs last;
* `tokenizer.json` -- the checkpoint's tokenizer, for `tokenizers` alone;
* `laya.reference.json` -- a request, and what laya's own PyTorch runtime answers to it.

**What the model is.** One question at a time, padded to `max_len` (512) tokens: the encoder,
the type embedding of the question, the two head layers, and the option scorer at every
position. It hands back one score per position and the `[CLS]` state. What is left is the
caller's, in numpy:

* the token embedding lookup and the type embedding lookup -- ncnn has no integer tensors;
* the attention masks, as additive `[L, L]` biases: a global one that hides the padding, and
  the sliding one the local layers use, which hides everything more than 64 tokens away too.
  `-1e4` rather than `-inf` or the float minimum, which the GPU kernels make NaN of, and inside
  fp16's range;
* picking the scores at the option markers, the temperature, and the act head, which reads
  the probabilities they make.

**The Hub model, one question at a time.** `Exported` below calls Laya's own modules on the
checkpoint's weights; what it leaves out is what pnnx cannot follow in them, and is the
caller's instead: the lookups, the masks, which `transformers` would build from the input,
and the gather at the markers and what follows it. The check after the conversion is against
laya's own runtime.

What it took to get it through pnnx and onto the GPU is in the comments below it: the norms
without a bias, the RoPE pnnx fuses and the masks ncnn's Vulkan SDPA does not broadcast.
"""

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from huggingface_hub import snapshot_download

REPO = "convaiinnovations/laya"
REVISION = "55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851"

# What the caller writes into the masks where a token is not to be attended.
MASKED = -1e4

# The request `laya.reference.json` records: the README's own example.
STATE = {
    "from": "user@acme.com",
    "subject": "Duplicate charge on invoice #4411",
    "body": "Hi, we were billed twice for March. Please refund the duplicate today or we will cancel our plan.",
}
QUESTIONS = {
    "department": {
        "type": "choice",
        "instructions": "Which department should handle this request?",
        "criteria": {
            "billing": "invoices, payments, refunds",
            "technical": "bugs, outages, system errors",
            "sales": "pricing, new contracts",
            "other": "everything else",
        },
    },
    "urgency": {
        "type": "score",
        "instructions": "How urgent is this request?",
        "criteria": ["not urgent", "soon", "critical deadline or blocking issue"],
    },
    "churn_risk": {"type": "noul", "instructions": "Does the user threaten to cancel or leave?"},
    "refund_requested": {"type": "noul", "instructions": "Does the user explicitly request a refund?"},
}

# Below this between ncnn on the host CPU and PyTorch, the conversion counts as wrong.
MIN_COS = 0.999


class Exported(nn.Module):
    """Laya's own modules for one question at a fixed length, from the embeddings to the scores.

    The encoder is `transformers`' ModernBertModel, called as it is: it takes the masks as a
    mapping from layer type to mask, which is what keeps its masking utilities out of the graph,
    and works out RoPE from positions that are `arange(length)` every time, which the trace
    makes constants of. The head layers are Laya's `nn.TransformerEncoderLayer`s, and the
    scorer Laya's, each called as it is.
    """

    def __init__(self, model, length: int):
        super().__init__()
        self.encoder, self.head, self.scorer = model.encoder, model.head.layers, model.scorer
        self.length, self.heads = length, model.encoder.config.num_attention_heads

    def forward(self, embeds, global_mask, local_mask, type_embed):
        # One mask a head, spelled out: ncnn's Vulkan SDPA reads a mask per head and does not
        # broadcast one across them as its CPU SDPA does.
        full = global_mask[:, None].expand(1, self.heads, self.length, self.length)
        sliding = local_mask[:, None].expand(1, self.heads, self.length, self.length)
        h = self.encoder(inputs_embeds=embeds, attention_mask={"full_attention": full, "sliding_attention": sliding}).last_hidden_state
        h = h + type_embed
        for layer in self.head:
            # A float `src_mask` of one mask a head, `[heads, L, L]`, is added to the scores just as
            # `src_key_padding_mask` would hide the padding.
            h = layer(h, src_mask=full.reshape(self.heads, self.length, self.length))
        return self.scorer(h).reshape(1, self.length), h[:, 0]


def masks(n: int, length: int, window: int) -> tuple[np.ndarray, np.ndarray]:
    """The global and the sliding attention biases for `n` real tokens in `length`."""
    pos = np.arange(length)
    visible = np.broadcast_to(pos[None] < n, (length, length))
    near = np.abs(pos[:, None] - pos[None]) <= window
    return (np.where(visible, 0, MASKED).astype(np.float32), np.where(visible & near, 0, MASKED).astype(np.float32))


def feeds(model, tok, ids: list[int], qtype: int, length: int, window: int) -> list[torch.Tensor]:
    table = model.encoder.embeddings.tok_embeddings.weight
    padded = torch.tensor(ids + [tok.pad_token_id] * (length - len(ids)))
    g, l = masks(len(ids), length, window)
    return [table[padded][None], torch.from_numpy(g)[None], torch.from_numpy(l)[None], model.type_emb.weight[qtype][None, None]]


def encode(agent, questions: dict) -> list[dict]:
    """Each question's token ids and option markers, as laya itself builds them."""
    from laya.common import QTYPES, build_sequence

    cfg = agent.cfg
    out = []
    for qid, q in questions.items():
        q = agent._to_internal(q)
        ids, markers = build_sequence(agent.tok, STATE, q, cfg["max_len"], cfg["head_max_len"])
        out.append({"id": qid, "ids": ids, "markers": markers, "qtype": QTYPES[q["t"]]})
    return out


def cos(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.ravel().astype(np.float64), b.ravel().astype(np.float64)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))


def check(ncnn_dir: Path, io: dict, model, tok, items: list[dict], window: int):
    """Run the ncnn model on the host CPU in fp32 against the Hub model in PyTorch."""
    import ncnn

    net = ncnn.Net()
    net.opt.use_vulkan_compute = False
    net.opt.use_fp16_storage = net.opt.use_fp16_packed = net.opt.use_fp16_arithmetic = False
    assert net.load_param(str(ncnn_dir / "laya.ncnn.param")) == 0
    assert net.load_model(str(ncnn_dir / "laya.ncnn.bin")) == 0
    length = io["max_len"]
    for item in items:
        k = len(item["markers"])
        with torch.no_grad():
            ids = torch.tensor([item["ids"]])
            logits, _ = model(ids, torch.ones_like(ids), torch.tensor([item["markers"]]),
                              torch.ones(1, k, dtype=torch.bool), torch.tensor([item["qtype"]]))
        ex = net.create_extractor()
        for spec, x in zip(io["inputs"], feeds(model, tok, item["ids"], item["qtype"], length, window)):
            x = np.ascontiguousarray(x.detach().numpy(), np.float32).reshape(spec["ncnn_shape"])
            ex.input(spec["name"], ncnn.Mat(x).clone())
        ret, scores = ex.extract(io["outputs"][0]["name"])
        assert ret == 0, f"extract failed: {ret}"
        got, want = np.array(scores).ravel()[item["markers"]], logits[0].numpy()
        c = cos(got, want)
        print(f"  {item['id']}: ncnn {np.round(got, 3)} torch {np.round(want, 3)} cos={c:.5f}", flush=True)
        if c < MIN_COS:
            raise SystemExit(f"{item['id']}: ncnn is too far from PyTorch")


def _patch_pnnx_on_windows() -> None:
    r"""Stop pnnx from importing the `*_pnnx.py` it generates.

    `pnnx.export` writes what this script is after -- the ncnn param and bin -- and then, as
    the last thing `convert()` does, imports the Python transcript of the graph it wrote
    beside them, so as to hand the caller back a torch module. Nothing here wants that
    module: the call below drops the return value and takes the files off disk.

    That import is where a conversion stops on Windows. pnnx writes the paths it was handed
    into the transcript as ordinary string literals, so `zipfile.ZipFile('C:\Users\...')`
    reaches the parser as escapes and `\U` in `C:\Users` is a SyntaxError before a line of
    the module runs. The paths that do parse are the worse half, since `\a`, `\b`, `\n` and
    `\t` become control characters and the module then opens a file nobody wrote.

    Repairing the literals only moves the wall: the transcript is not always Python at all,
    because an op pnnx has no Python spelling for is written out verbatim, and a line like
    `v_1103 = aten::to(v_1100, v_1102, v_1101, v_1101)` parses nowhere. Neither is worth
    fixing for a module that is thrown away, so the import is made to produce nothing
    instead. `convert()` reaches it through `importlib.util.spec_from_file_location`, which
    is wrapped here to hand a `_pnnx.py` back with a loader that never reads the source and
    defines the one name `convert()` goes on to touch. Every other module keeps the loader it
    would have had.

    Windows only, and gated rather than unconditional because a host where the import works
    gets a real torch module out of it -- worth keeping for anything that comes to want one.

    An op ncnn cannot run is a separate matter and is not hidden by this: it stays in the
    `.param` as an unregistered layer, and the check against PyTorch at the end is what
    catches it.
    """
    if sys.platform != "win32":
        return

    import importlib.abc
    import importlib.util

    # The wrapper goes on once, however often this is called: wrapping a wrapper would work,
    # and would also leave a chain as long as the model has pieces.
    if getattr(importlib.util.spec_from_file_location, "_pnnx_skips_transcript", False):
        return

    class _TranscriptLoader(importlib.abc.Loader):
        """Loader that gives the module a `Model` and never looks at the file."""

        def create_module(self, spec):
            return None  # the default module object is enough

        def exec_module(self, module):
            # `convert()` ends in `return foo.Model()`, which is the whole of what it asks
            # the transcript for, and the caller here discards it.
            module.Model = lambda *args, **kwargs: None

    spec_from_file_location = importlib.util.spec_from_file_location

    def patched(name, location=None, *args, **kwargs):
        spec = spec_from_file_location(name, location, *args, **kwargs)
        if spec is not None and str(location).endswith("_pnnx.py"):
            spec.loader = _TranscriptLoader()
        return spec

    patched._pnnx_skips_transcript = True
    importlib.util.spec_from_file_location = patched


def convert(checkpoint: Path, ncnn_dir: Path, work: Path):
    import laya
    import pnnx

    _patch_pnnx_on_windows()

    agent = laya.load(str(checkpoint), device="cpu")
    model = agent.model.float().eval()
    cfg, enc_cfg = agent.cfg, model.encoder.config
    length, window = cfg["max_len"], enc_cfg.local_attention // 2
    items = encode(agent, QUESTIONS)

    # ModernBERT's norms have a weight and no bias, which pnnx writes as an ncnn LayerNorm
    # with `affine=0` -- and ncnn then drops the weight as well. A zero bias keeps both.
    for norm in model.modules():
        if isinstance(norm, nn.LayerNorm) and norm.bias is None:
            norm.bias = nn.Parameter(torch.zeros_like(norm.weight))
    # `nn.TransformerEncoderLayer`'s fast path would trace as one op pnnx does not have.
    torch.backends.mha.set_fastpath_enabled(False)
    # ModernBERT's `rotate_half` -- (x1, x2) to (-x2, x1) -- as a product with a constant
    # matrix, where it slices and concatenates. Spelled its way, pnnx fuses RoPE into one ncnn
    # RotaryEmbed, whose Vulkan kernel gives other numbers than its CPU one.
    import transformers.models.modernbert.modeling_modernbert as modernbert

    dim = enc_cfg.hidden_size // enc_cfg.num_attention_heads
    turn = torch.zeros(dim, dim)
    turn[torch.arange(dim // 2) + dim // 2, torch.arange(dim // 2)] = -1
    turn[torch.arange(dim // 2), torch.arange(dim // 2) + dim // 2] = 1
    modernbert.rotate_half = lambda x: torch.matmul(x, turn.to(x.dtype))
    exported = Exported(model, length).eval()
    example = feeds(model, agent.tok, items[0]["ids"], items[0]["qtype"], length, window)
    with torch.no_grad():
        pnnx.export(
            exported,
            str(work / "laya.pt"),
            tuple(example),
            ncnnparam=str(ncnn_dir / "laya.ncnn.param"),
            ncnnbin=str(work / "laya.ncnn.bin"),
            ncnnpy=str(work / "laya_ncnn.py"),
            pnnxparam=str(work / "laya.pnnx.param"),
            pnnxbin=str(work / "laya.pnnx.bin"),
            pnnxpy=str(work / "laya_pnnx.py"),
            pnnxonnx=str(work / "laya.pnnx.onnx"),
            fp16=True,
        )

    table = model.encoder.embeddings.tok_embeddings.weight.detach().numpy()
    np.save(ncnn_dir / "laya.embeddings.npy", table.astype(np.float16))
    np.save(ncnn_dir / "laya.type_embed.npy", model.type_emb.weight.detach().numpy().astype(np.float32))
    act = model.act_head
    np.savez(
        ncnn_dir / "laya.act_head.npz",
        w0=act[0].weight.detach().numpy(), b0=act[0].bias.detach().numpy(),
        w1=act[2].weight.detach().numpy(), b1=act[2].bias.detach().numpy(),
    )
    shutil.copy(checkpoint / "tokenizer" / "tokenizer.json", ncnn_dir / "tokenizer.json")

    tok = agent.tok
    # pnnx drops the batch dim of every blob, and names them `in0..` and `out0..`.
    shapes_in = [list(x.shape) for x in example]
    io = {
        "inputs": [
            {"name": f"in{i}", "role": role, "shape": s, "ncnn_shape": s[1:]}
            for i, (role, s) in enumerate(zip(["embeds", "global_mask", "local_mask", "type_embed"], shapes_in))
        ],
        "outputs": [
            {"name": "out0", "role": "scores", "shape": [1, length], "ncnn_shape": [length]},
            {"name": "out1", "role": "pooled", "shape": [1, enc_cfg.hidden_size], "ncnn_shape": [enc_cfg.hidden_size]},
        ],
        "max_len": length,
        "head_max_len": cfg["head_max_len"],
        "window": window,
        "masked": MASKED,
        "tokens": {name: getattr(tok, f"{name}_token_id") for name in ("cls", "sep", "mask", "pad")},
        "mask_token": tok.mask_token,
        # Clamped as laya clamps them; see `laya.common.clamp_temperature`.
        "temperature": agent.temperature,
        "temperature_by_options": agent.temperature_by_options,
    }
    (ncnn_dir / "laya.json").write_text(json.dumps(io, indent=2) + "\n")

    # The reference: laya's own runtime on the request, and the ids it built for each question.
    answer = agent.predict(STATE, QUESTIONS)
    reference = {"state": STATE, "questions": QUESTIONS, "encoded": items, "answers": answer["answers"]}
    (ncnn_dir / "laya.reference.json").write_text(json.dumps(reference, indent=2, ensure_ascii=False) + "\n")

    # The `.bin` is what says the model is done, so it appears once the rest is in place.
    (work / "laya.ncnn.bin").rename(ncnn_dir / "laya.ncnn.bin")
    print("checking ncnn on the host CPU against PyTorch", flush=True)
    check(ncnn_dir, io, model, tok, items, window)


def download(checkpoint: Path):
    """The English checkpoint at the repo root; the Hub client skips what is already there."""
    snapshot_download(
        REPO,
        revision=REVISION,
        allow_patterns=["model.safetensors", "rl_agent_config.json", "encoder/*", "tokenizer/*"],
        local_dir=checkpoint,
    )


def main():
    data = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent / "data"
    checkpoint, ncnn_dir = data / "checkpoint", data / "ncnn"
    work = ncnn_dir / "work"
    if (ncnn_dir / "laya.ncnn.bin").exists():
        print("laya: already converted")
        return
    download(checkpoint)
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    print("laya: converting", flush=True)
    convert(checkpoint.resolve(), ncnn_dir.resolve(), work.resolve())
    shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()
