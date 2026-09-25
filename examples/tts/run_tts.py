"""Speak a text in a voice an instruction describes, with the ncnn Qwen3-TTS pieces under
/models on the Vulkan device.

Run inside the guest as `python3 run_tts.py MODE REQUEST`, from the tts skill mounted at
/skills/tts; MODE is one of MODES below and REQUEST a JSON `{"text": ..., "instruct": ...,
"language": ..., "out": path, "seed": n}`, of which `text` and `out` are required. `text_file`
and `instruct_file` name files to read them from instead. `TTS_MODELS` names another
directory than /models, to run it on the host.

What each piece takes and gives back, the token ids and the sampling settings are in
`tts.json`, which `prepare_model.py` wrote beside them, with the codec embeddings in `tts.npz`
and the text embeddings, already projected, in `text.npy`. The steps are qwen-tts'
`generate_voice_design`, redone in numpy around the pieces: the prompt laid out as
`Qwen3TTSForConditionalGeneration.generate` lays it out, with the whole text in it, the
talker's and the predictor's sampling as transformers' `generate` does it, and the codec's
decoding chunked as `chunked_decode` does.

stdout is one JSON object saying what was written, and all else goes to stderr. The waveform
is written to `out` as a 16-bit mono WAV at 24 kHz.

Run as `python3 run_tts.py check MODE`, it checks the pieces instead: it runs the talker and
the predictor on the codes the PyTorch model sampled for the example, in
`tts.reference.json`, and compares the log-probability each gave every code; and it decodes
them with the codec, against the PyTorch model's waveform in `reference.wav`.

Exits 2 if the wheel has no Vulkan, 3 if it finds no device, 4 on a request it cannot take, 5 on
a non-finite output or, when checking, on the pieces straying too far from the reference.
"""

import json
import os
import sys
import time
import wave

import ncnn
import numpy as np
from tokenizers import Tokenizer

MODELS = os.environ.get("TTS_MODELS", "/models")

# Whether to store and compute in fp16. ncnn quietly drops it if the device has no such type,
# which is fp32 again, and the `fp16-p/s/u/a` line on stderr says so.
MODES = {"fp32": False, "fp16": True}

# When checking: the most a code's log-probability may stray from the reference's, on
# average over the codes, by mode; and the least the waveform's correlation with it may be.
MAX_LOGP_DIFF = {"fp32": 0.05, "fp16": 0.15}
MIN_WAVE_COS = 0.95

# A frame is 80 ms, so this is about 10 minutes: a run that has not ended by then is not going
# to.
MAX_FRAMES = 8192


def log(*args):
    print(*args, file=sys.stderr, flush=True)


class BadRequest(Exception):
    pass


def device():
    """The Vulkan device ncnn will run on, or the reason there is none."""
    if not hasattr(ncnn, "get_gpu_count"):
        log("vulkan: NOT BUILT IN (no ncnn.get_gpu_count)")
        sys.exit(2)
    if ncnn.get_gpu_count() == 0:
        log("vulkan: no device")
        sys.exit(3)
    log(f"ncnn {getattr(ncnn, '__version__', '?')} on {ncnn.get_gpu_info(0).device_name()}")


def mat(a: np.ndarray) -> ncnn.Mat:
    # Held while it is cloned: `ncnn.Mat` wraps the array, and a temporary is gone by then.
    a = np.ascontiguousarray(a, np.float32)
    return ncnn.Mat(a).clone()


class Model:
    """The pieces, the tables the steps around them look things up in, and the tokenizer."""

    def __init__(self, mode: str):
        with open(f"{MODELS}/tts.json") as f:
            self.config = json.load(f)
        self.io = self.config["pieces"]
        consts = np.load(f"{MODELS}/tts.npz")
        self.codec_embed = consts["codec_embed"]
        self.predictor_embed = consts["predictor_embed"]
        self.rvq = consts["rvq"]
        # 600 MB, of which a prompt reads a few rows.
        self.text = np.load(f"{MODELS}/text.npy", mmap_mode="r")
        self.tokenizer = Tokenizer.from_file(f"{MODELS}/tokenizer.json")
        fp16 = MODES[mode]
        self.nets = {}
        for name in self.io:
            net = ncnn.Net()
            net.opt.use_vulkan_compute = True
            # A piece whose weights are fp32 runs so whatever the mode: see `tts.json`.
            half = fp16 and self.io[name]["fp16"]
            net.opt.use_fp16_storage = net.opt.use_fp16_packed = net.opt.use_fp16_arithmetic = half
            net.set_vulkan_device(0)
            t = time.time()
            assert net.load_param(f"{MODELS}/tts_{name}.ncnn.param") == 0, f"{name}: load_param failed"
            assert net.load_model(f"{MODELS}/tts_{name}.ncnn.bin") == 0, f"{name}: load_model failed"
            log(f"tts_{name} [{mode}]: loaded in {time.time() - t:.1f}s")
            self.nets[name] = net

    def release(self, *names: str):
        """Give the pieces' device memory back. The codec's buffers are large, and past what
        the guest's device allocates with the talker's weights and blobs still held."""
        for name in names:
            self.nets.pop(name).clear()

    def tokens(self, text: str) -> list[int]:
        return self.tokenizer.encode(text, add_special_tokens=False).ids

    def text_embeds(self, ids) -> np.ndarray:
        return self.text[np.asarray(ids)].astype(np.float32)


def angles(start: int, n: int, head_dim: int, theta: float) -> tuple[np.ndarray, np.ndarray]:
    """RoPE's cos and sin for positions start..start+n, half the head's width, as
    `RotaryEmbed` takes them."""
    inv = 1.0 / theta ** (np.arange(0, head_dim, 2, dtype=np.float64) / head_dim)
    f = np.outer(np.arange(start, start + n, dtype=np.float64), inv)
    return np.cos(f).astype(np.float32), np.sin(f).astype(np.float32)


def causal(past: int, n: int, masked: float, window: int | None = None) -> np.ndarray:
    """[n, past + n]: each new position sees the cache and itself and before, within
    `window` of itself when there is one."""
    q = np.arange(past, past + n)[:, None]
    k = np.arange(past + n)[None, :]
    far = (k > q) | ((q - k >= window) if window else False)
    return np.where(far, masked, 0.0).astype(np.float32)


def run(net, io: dict, inputs: list) -> list:
    """The piece's outputs, as numpy arrays, for `inputs`, numpy arrays or `Mat`s in its
    inputs' order."""
    ex = net.create_extractor()
    for inp, a in zip(io["inputs"], inputs):
        ex.input(inp["name"], a if isinstance(a, ncnn.Mat) else mat(a))
    outs = []
    for out in io["outputs"]:
        ret, m = ex.extract(out["name"])
        assert ret == 0, f"extract {out['name']} failed: {ret}"
        outs.append(np.array(m))
    return outs


class TalkerRun:
    """The talker, run a few positions at a time, its cache carried between runs.

    The cache is kept here, in blocks of BLOCK positions, those not yet written hidden by the
    mask: the talker takes it whole and gives back the new positions' keys and values, written
    in after the others. Its shape changes once a block, not once a step: the guest's device
    goes astray after a hundred or so buffers of new sizes."""

    BLOCK = 128

    def __init__(self, model: Model):
        self.net, self.io = model.nets["talker"], model.io["talker"]
        c = model.config["talker"]
        self.head_dim, self.theta = c["head_dim"], c["rope_theta"]
        self.masked = model.config["masked"]
        self.cache = [np.zeros((c["kv_heads"], self.BLOCK, c["head_dim"]), np.float32) for _ in range(2 * c["layers"])]
        self.length = 0

    def __call__(self, embeds: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        n = embeds.shape[0]
        room = self.cache[0].shape[1]
        if self.length + n > room:
            grown = -(-(self.length + n) // self.BLOCK) * self.BLOCK
            self.cache = [np.concatenate([a, np.zeros((a.shape[0], grown - room, a.shape[2]), np.float32)], axis=1) for a in self.cache]
            room = grown
        cos, sin = angles(self.length, n, self.head_dim, self.theta)
        mask = causal(room, n, self.masked)
        mask[:, self.length : room] = self.masked
        # The cache's arrays are held until the run is over, so wrapped rather than cloned.
        hidden, logits, *new = run(self.net, self.io, [embeds, cos, sin, mask, *map(ncnn.Mat, self.cache)])
        for a, m in zip(self.cache, new):
            a[:, self.length : self.length + n] = m
        self.length += n
        return hidden, logits


# --- The prompt -----------------------------------------------------------------------------


def prompt(model: Model, text: str, instruct: str, language: str) -> np.ndarray:
    """The talker's prefill, as `generate` lays it out with the whole text in it: the
    instruction, the assistant's turn, the codec's tags under pads and then under the text,
    and the codec's start."""
    c = model.config
    tok, codec = c["tokens"], c["codec"]
    T = lambda ids: model.text_embeds(ids)
    C = lambda ids: model.codec_embed[np.asarray(ids)]
    bos, eos, pad = T([tok["tts_bos"], tok["tts_eos"], tok["tts_pad"]])

    if language.lower() == "auto":
        tags = [codec["nothink"], codec["think_bos"], codec["think_eos"]]
    elif language.lower() in codec["languages"]:
        tags = [codec["think"], codec["think_bos"], codec["languages"][language.lower()], codec["think_eos"]]
    else:
        raise BadRequest(f"unknown language {language!r}, not auto or one of {sorted(codec['languages'])}")
    tags += [codec["pad"], codec["bos"]]

    ids = model.tokens(f"<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n")
    words = ids[3:-5]
    if not words:
        raise BadRequest("the text is empty")
    parts = []
    if instruct:
        parts.append(T(model.tokens(f"<|im_start|>user\n{instruct}<|im_end|>\n")))
    parts += [
        T(ids[:3]),
        np.concatenate([np.repeat(pad[None], len(tags) - 2, 0), bos[None]]) + C(tags[:-1]),
        np.concatenate([T(words), eos[None]]) + C([codec["pad"]] * (len(words) + 1)),
        pad[None] + C([codec["bos"]]),
    ]
    return np.concatenate(parts).astype(np.float32)


# --- Sampling -------------------------------------------------------------------------------


def sample(logits: np.ndarray, rng, temperature: float, top_k: int, top_p: float, do_sample: bool) -> int:
    """transformers' temperature, top-k and top-p warpers, then a draw; or the argmax."""
    if not do_sample:
        return int(np.argmax(logits))
    x = logits.astype(np.float64) / temperature
    if 0 < top_k < x.size:
        x[x < np.partition(x, -top_k)[-top_k]] = -np.inf
    p = np.exp(x - x.max())
    p /= p.sum()
    if top_p < 1.0:
        order = np.argsort(p)
        drop = np.cumsum(p[order]) <= 1 - top_p
        drop[-1] = False
        p[order[drop]] = 0
        p /= p.sum()
    return int(rng.choice(p.size, p=p))


class Talker:
    """The talker and the predictor, a frame at a time: its 16 codes and what the talker takes
    next."""

    def __init__(self, model: Model, seed: int):
        self.model = model
        c = model.config
        self.s = c["sampling"]
        self.eos = c["codec"]["eos"]
        self.vocab = c["predictor"]["vocab"]
        self.groups = c["predictor"]["groups"]
        self.suppress_from = c["suppress_from"]
        self.min_new = c["min_new_tokens"]
        self.pad = model.text_embeds([c["tokens"]["tts_pad"]])[0]
        self.rng = np.random.default_rng(seed)
        self.talker = TalkerRun(model)
        self.firsts = []

    def first_code(self, logits: np.ndarray) -> int:
        """The logits processors `generate` gives the talker: the repetition penalty on the
        first codes so far, the prompt's codes suppressed and the end until `min_new_tokens`."""
        x = logits.astype(np.float64).copy()
        if self.firsts:
            seen = np.unique(self.firsts)
            r = self.s["repetition_penalty"]
            x[seen] = np.where(x[seen] > 0, x[seen] / r, x[seen] * r)
        keep_eos = x[self.eos]
        x[self.suppress_from :] = -np.inf
        x[self.eos] = keep_eos if len(self.firsts) >= self.min_new else -np.inf
        return sample(x, self.rng, self.s["temperature"], self.s["top_k"], self.s["top_p"], self.s["do_sample"])

    def rest(self, hidden: np.ndarray, first: int, forced=None) -> tuple[list[int], list[np.ndarray]]:
        """The predictor's 15 codes of the frame whose first code is `first`, from the talker's
        hidden state; `forced` takes the given codes instead of sampling, for checking. The
        log-softmax of each head's logits comes back too."""
        s, m = self.s, self.model
        c = m.config["predictor"]
        net, io = m.nets["predictor"], m.io["predictor"]
        # All of the frame so far, each code: it is 16 positions at most.
        x = [hidden, m.codec_embed[first]]
        codes, logps = [], []
        for g in range(self.groups - 1):
            n = len(x)
            cos, sin = angles(0, n, c["head_dim"], c["rope_theta"])
            (logits,) = run(net, io, [np.stack(x).astype(np.float32), cos, sin, causal(0, n, m.config["masked"])])
            head = logits[-1, g * self.vocab : (g + 1) * self.vocab]
            logps.append(head - head.max() - np.log(np.exp(head - head.max()).sum()))
            code = forced[g] if forced is not None else sample(
                head, self.rng, s["subtalker_temperature"], s["subtalker_top_k"], s["subtalker_top_p"], s["subtalker_dosample"]
            )
            codes.append(int(code))
            x.append(m.predictor_embed[g][code].astype(np.float32))
        return codes, logps

    def next_embed(self, frame: list[int]) -> np.ndarray:
        m = self.model
        e = m.codec_embed[frame[0]] + self.pad
        for g, code in enumerate(frame[1:]):
            e = e + m.predictor_embed[g][code].astype(np.float32)
        return e[None].astype(np.float32)

    def generate(self, prefill: np.ndarray, progress=None) -> list[list[int]]:
        hidden, logits = self.talker(prefill)
        frames = []
        while len(frames) < MAX_FRAMES:
            if not np.isfinite(logits[-1]).all():
                raise FloatingPointError("the talker's logits are not finite")
            first = self.first_code(logits[-1])
            self.firsts.append(first)
            if first == self.eos:
                break
            rest, _ = self.rest(hidden[-1], first)
            frames.append([first] + rest)
            hidden, logits = self.talker(self.next_embed(frames[-1]))
            if progress:
                progress(len(frames))
        return frames


# --- The codec ------------------------------------------------------------------------------


def decode(model: Model, frames: np.ndarray) -> np.ndarray:
    """The waveform of [N, 16] codes, a chunk of frames at a time with the frames before it as
    context, as `chunked_decode` does."""
    c = model.config["decoder"]
    net, io, masked = model.nets["codec"], model.io["codec"], model.config["masked"]
    x = sum(model.rvq[q][frames[:, q]].astype(np.float32) for q in range(frames.shape[1]))
    up, chunk, context = c["upsample"], c["chunk"], c["context"]
    out = []
    for start in range(0, len(x), chunk):
        ctx = min(context, start)
        part = x[start - ctx : start + chunk]
        n = part.shape[0]
        cos, sin = angles(0, n, c["head_dim"], c["rope_theta"])
        (wav,) = run(net, io, [part, cos, sin, causal(0, n, masked, c["sliding_window"])])
        out.append(wav.reshape(-1)[ctx * up :])
    return np.concatenate(out)


def write_wav(path: str, samples: np.ndarray, rate: int):
    pcm = (np.clip(samples, -1, 1) * 32767).astype("<i2")
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm.tobytes())


# --- Running and checking -------------------------------------------------------------------


def read(request: dict, key: str) -> str:
    if key in request and f"{key}_file" in request:
        raise BadRequest(f"give {key} or {key}_file, not both")
    if f"{key}_file" in request:
        try:
            with open(request[f"{key}_file"], encoding="utf-8") as f:
                return f.read().strip()
        except OSError as e:
            raise BadRequest(f"cannot read {key}_file: {e}")
    return str(request.get(key, "")).strip()


def speak(mode: str, request: dict) -> dict:
    text, instruct = read(request, "text"), read(request, "instruct")
    out = request.get("out")
    if not out:
        raise BadRequest("the request has no `out` path")
    language = request.get("language", "auto")
    seed = int(request.get("seed", 0))
    model = Model(mode)
    prefill = prompt(model, text, instruct, language)
    log(f"prompt: {prefill.shape[0]} positions")
    t = time.time()
    tick = lambda n: n % 25 == 0 and log(f"  {n} frames, {n * 0.08:.1f}s of speech in {time.time() - t:.1f}s")
    talker = Talker(model, seed)
    frames = talker.generate(prefill, tick)
    gen = time.time() - t
    del talker
    model.release("talker", "predictor")
    log(f"decoding {len(frames)} frames")
    if not frames:
        raise FloatingPointError("the talker ended before a single frame")
    t = time.time()
    wav = decode(model, np.array(frames))
    if not np.isfinite(wav).all():
        raise FloatingPointError("the waveform is not finite")
    rate = model.config["decoder"]["sample_rate"]
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    write_wav(out, wav, rate)
    seconds = len(wav) / rate
    ended = len(frames) < MAX_FRAMES
    return {
        "mode": mode,
        "out": out,
        "seconds": round(seconds, 2),
        "frames": len(frames),
        "ended": ended,
        "generate_s": round(gen, 1),
        "decode_s": round(time.time() - t, 1),
        "realtime_factor": round((gen + time.time() - t) / seconds, 2),
        "language": language,
        "seed": seed,
    }


def check(mode: str) -> int:
    with open(f"{MODELS}/tts.reference.json") as f:
        ref = json.load(f)
    model = Model(mode)
    codes = ref["codes"]
    talker = Talker(model, 0)
    hidden, logits = talker.talker(prompt(model, ref["text"], ref["instruct"], "auto"))
    diffs, t = [], time.time()
    for f, frame in enumerate(codes):
        x = logits[-1].astype(np.float64)
        first = x[frame[0]] - x.max() - np.log(np.exp(x - x.max()).sum())
        _, logps = talker.rest(hidden[-1], frame[0], forced=frame[1:])
        got = [first] + [lp[c] for lp, c in zip(logps, frame[1:])]
        diffs.append(np.abs(np.array(got) - np.array(ref["logp"][f])))
        hidden, logits = talker.talker(talker.next_embed(frame))
    diffs = np.array(diffs)
    log(f"talker and predictor: {len(codes)} frames in {time.time() - t:.1f}s")
    print(f"log-probability off by {diffs[:, 0].mean():.4f} on average for the first code, {diffs[:, 1:].mean():.4f} for the rest, {diffs.max():.3f} at most", flush=True)
    del talker
    model.release("talker", "predictor")
    wav = decode(model, np.array(codes))
    with wave.open(f"{MODELS}/reference.wav") as w:
        want = np.frombuffer(w.readframes(w.getnframes()), "<i2").astype(np.float32) / 32767 if w.getsampwidth() == 2 else None
    n = min(len(wav), len(want))
    a, b = wav[:n].astype(np.float64), want[:n].astype(np.float64)
    corr = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))
    print(f"waveform: {len(wav)} samples against {len(want)}, correlation {corr:.4f}")
    ok = diffs.mean() <= MAX_LOGP_DIFF[mode] and corr >= MIN_WAVE_COS and np.isfinite(wav).all()
    print(f"tts [{mode}]:", "OK" if ok else "DIFFERS FROM THE REFERENCE")
    return 0 if ok else 5


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "check":
        mode = sys.argv[2]
        assert mode in MODES, f"unknown mode {mode}, not in {list(MODES)}"
        device()
        sys.exit(check(mode))
    if len(sys.argv) != 3:
        log(__doc__)
        sys.exit(4)
    mode = sys.argv[1]
    assert mode in MODES, f"unknown mode {mode}, not in {list(MODES)}"
    device()
    try:
        report = speak(mode, json.loads(sys.argv[2]))
    except (BadRequest, json.JSONDecodeError) as e:
        print(json.dumps({"error": str(e)}, ensure_ascii=False))
        sys.exit(4)
    except FloatingPointError as e:
        print(json.dumps({"error": str(e)}, ensure_ascii=False))
        sys.exit(5)
    print(json.dumps(report, ensure_ascii=False))
