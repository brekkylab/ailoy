"""Answer the reference request with the ncnn Laya model under /models, on the Vulkan device.

Run inside the guest as `python3 -c RUN MODE`, so it has no file of its own there; MODE is one
of MODES below. `LAYA_MODELS` names another directory than /models, to run it on the host.
What the model takes and gives back, and the constants the
steps around it need, are in `laya.json`, which `prepare_model.py` wrote beside it; the request
and laya's own answers to it, from PyTorch on the host, are in `laya.reference.json`.

Each question is one extraction: its sequence is built as `laya.common.build_sequence` builds
it, with `tokenizers` alone, looked up in the embedding table, padded to `max_len` and run
through the model. The scores at the option markers, the temperature and the act head give
the answer, in the shape `laya` answers in. The token ids are checked against the ones laya
built, and the answers against laya's.

Exits 2 if the wheel has no Vulkan, 3 if it finds no device, 4 if a question's tokens are not
laya's, 5 on a non-finite answer or one too far from laya's.
"""

import json
import math
import os
import sys
import time

import numpy as np
import ncnn
from tokenizers import Tokenizer

MODELS = os.environ.get("LAYA_MODELS", "/models")
QTYPES = {"choice": 0, "score": 1, "noul": 2}
# Past this in any probability from laya's, an answer counts as wrong, by mode. fp32 is within
# 0.002 of laya's. bf16 is off by up to ~0.07, and not by the same from one run to the next:
# the residual stream reaches ~30000, where bf16's 8 bits of mantissa step by hundreds.
MAX_DIFF = {"fp32": 0.01, "bf16": 0.1}

# Whether to store in bf16; the arithmetic stays fp32. Quietly dropped by ncnn if the device
# has no `shaderBFloat16Type`, which is fp32 again, and the `bf16-p/s` line on stderr says so.
#
# fp16 is not a mode. ModernBERT's residual stream reaches ~30000 by its last layers: inside
# fp16's range, but not its square, and ncnn's Vulkan LayerNorm squares it in fp16 whatever
# `use_fp16_arithmetic` says -- from the ninth layer on it hands back zeros, and every answer
# comes out uniform. bf16 has fp32's range. Taking the LayerNorms alone out of fp16 storage
# (`31=featmask`) makes the extraction fail outright.
MODES = {"fp32": False, "bf16": True}


def device():
    """The Vulkan device ncnn will run on, or the reason there is none."""
    if not hasattr(ncnn, "get_gpu_count"):
        print("vulkan: NOT BUILT IN (no ncnn.get_gpu_count)")
        sys.exit(2)
    if ncnn.get_gpu_count() == 0:
        print("vulkan: no device")
        sys.exit(3)
    print(f"ncnn {getattr(ncnn, '__version__', '?')} on {ncnn.get_gpu_info(0).device_name()}")


def load(mode: str):
    net = ncnn.Net()
    net.opt.use_vulkan_compute = True
    net.opt.use_fp16_storage = False
    net.opt.use_fp16_packed = False
    net.opt.use_fp16_arithmetic = False
    net.opt.use_bf16_storage = MODES[mode]
    net.opt.use_bf16_packed = MODES[mode]
    net.set_vulkan_device(0)
    t = time.time()
    assert net.load_param(f"{MODELS}/laya.ncnn.param") == 0, "load_param failed"
    assert net.load_model(f"{MODELS}/laya.ncnn.bin") == 0, "load_model failed"
    print(f"laya [{mode}]: loaded in {time.time() - t:.1f}s")
    return net


# --- The sequence, as `laya.common` builds it -------------------------------------------------


def serialize(value) -> str:
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)


def criterion(value) -> str:
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, separators=(", ", ": "), default=str)


def internal(q: dict) -> dict:
    t, crit = q["type"], q.get("criteria")
    if t == "choice" and isinstance(crit, list):
        crit = {c: None for c in crit}
    elif t == "noul" and isinstance(crit, dict):
        crit = {str(k).lower(): v for k, v in crit.items()}
    return {"t": t, "ins": serialize(q["instructions"]), "crit": crit}


def options(q: dict) -> list[str]:
    t, crit = q["t"], q["crit"]
    if t == "choice":
        return [k if v is None or v == "" else f"{k}: {criterion(v)}" for k, v in crit.items()]
    if t == "score":
        return [f"level {i}: {criterion(c)}" for i, c in enumerate(crit)]
    crit = crit or {}
    f, t_ = crit.get("false"), crit.get("true")
    return [
        "false: " + (criterion(f) if f not in (None, "") else "no, the statement does not hold"),
        "true: " + (criterion(t_) if t_ not in (None, "") else "yes, the statement holds"),
    ]


def sequence(tok: Tokenizer, io: dict, state, q: dict) -> tuple[list[int], list[int]]:
    """`[CLS] <type> instructions [SEP] [MASK] opt0 [MASK] opt1 ... [SEP] state [SEP]`."""
    ids_of = lambda text: tok.encode(text, add_special_tokens=False).ids
    special, mask = io["tokens"], io["mask_token"]
    max_len, head_max_len = io["max_len"], io["head_max_len"]
    head = ids_of(f"{q['t']} question: {str(q['ins']).replace(mask, ' ')}")
    opts = [[special["mask"]] + ids_of(" " + o.replace(mask, " "))[:48] for o in options(q)]
    budget = head_max_len - sum(len(o) for o in opts)
    if budget < 16:
        per = max(4, (head_max_len - 16) // max(1, len(opts)))
        opts = [o[:per] for o in opts]
        budget = head_max_len - sum(len(o) for o in opts)
    ids = [special["cls"]] + head[: max(8, budget)] + [special["sep"]]
    markers = []
    for o in opts:
        markers.append(len(ids))
        ids.extend(o)
    ids.append(special["sep"])
    room = max(0, max_len - len(ids) - 1)
    ids = ids + ids_of(serialize(state).replace(mask, " "))[:room] + [special["sep"]]
    return ids[:max_len], [m for m in markers if m < max_len]


# --- The model and what comes after it ---------------------------------------------------------


def feeds(io: dict, table: np.ndarray, types: np.ndarray, ids: list[int], qtype: int) -> dict:
    length, n = io["max_len"], len(ids)
    pos = np.arange(length)
    visible = np.broadcast_to(pos[None] < n, (length, length))
    near = np.abs(pos[:, None] - pos[None]) <= io["window"]
    padded = np.array(ids + [io["tokens"]["pad"]] * (length - n))
    by_role = {
        "embeds": table[padded],
        "global_mask": np.where(visible, 0, io["masked"]),
        "local_mask": np.where(visible & near, 0, io["masked"]),
        "type_embed": types[qtype],
    }
    return {spec["name"]: by_role[spec["role"]] for spec in io["inputs"]}


def extract(net, io: dict, feeds: dict) -> dict:
    ex = net.create_extractor()
    for spec in io["inputs"]:
        # Named, not inlined: `ncnn.Mat` borrows the array's memory without holding it.
        x = np.ascontiguousarray(feeds[spec["name"]], np.float32).reshape(spec["ncnn_shape"])
        ex.input(spec["name"], ncnn.Mat(x).clone())
    out = {}
    for spec in io["outputs"]:
        ret, m = ex.extract(spec["name"])
        assert ret == 0, f"extract {spec['name']} failed: {ret}"
        out[spec["role"]] = np.array(m).reshape(spec["ncnn_shape"]).astype(np.float64)
    return out


def softmax(z: np.ndarray) -> np.ndarray:
    e = np.exp(z - z.max())
    return e / e.sum()


def gelu(x: np.ndarray) -> np.ndarray:
    return 0.5 * x * (1 + np.vectorize(math.erf)(x / math.sqrt(2)))


def act_probability(head: dict, logits: np.ndarray, pooled: np.ndarray) -> float:
    """The act head: whether to act on the answer rather than escalate it."""
    p = softmax(logits)
    k = max(len(p), 2)
    top = np.sort(p)[::-1]
    top2 = top[1] if len(top) > 1 else 0.0
    ent = -(p * np.log(np.clip(p, 1e-9, None))).sum() / math.log(k)
    x = np.concatenate([pooled, [top[0], top[0] - top2, ent, k / 255.0]])
    z = head["w1"] @ gelu(head["w0"] @ x + head["b0"]) + head["b1"]
    return float(softmax(z)[0])


def temperature(io: dict, qtype: str, k: int) -> float:
    size = "2" if k <= 2 else "3-5" if k <= 5 else "6-10" if k <= 10 else "11+"
    return io["temperature_by_options"].get(f"{qtype}:{size}", io["temperature"][QTYPES[qtype]])


def answer(io: dict, head: dict, q: dict, scores: np.ndarray, pooled: np.ndarray, markers: list[int]) -> dict:
    logits = scores[markers]
    k = len(logits)
    p = softmax(logits / temperature(io, q["t"], k))
    confidence = 1.0 if k < 2 else float(np.clip(1 + (p * np.log(np.clip(p, 1e-12, 1))).sum() / math.log(k), 0, 1))
    common = {"confidence": round(confidence, 4), "action": {"act_probability": round(act_probability(head, logits, pooled), 4)}}
    if q["t"] == "choice":
        keys = list(q["crit"])
        return {"type": "choice", "choice": keys[int(p.argmax())],
                "probabilities": {c: round(float(v), 4) for c, v in zip(keys, p)}, **common}
    if q["t"] == "score":
        return {"type": "score", "score": round(float((np.arange(k) * p).sum()), 4),
                "probabilities": {str(i): round(float(v), 4) for i, v in enumerate(p)}, **common}
    return {"type": "noul", "noul": round(float(p[1]), 4), **common, "confidence": round(max(p[1], 1 - p[1]), 4)}


def probabilities(a: dict) -> list[float]:
    return list(a["probabilities"].values()) if "probabilities" in a else [a["noul"]]


mode = sys.argv[1]
assert mode in MODES, f"unknown mode {mode}, not in {list(MODES)}"

device()
with open(f"{MODELS}/laya.json") as f:
    io = json.load(f)
with open(f"{MODELS}/laya.reference.json") as f:
    reference = json.load(f)
tok = Tokenizer.from_file(f"{MODELS}/tokenizer.json")
table = np.load(f"{MODELS}/laya.embeddings.npy")
types = np.load(f"{MODELS}/laya.type_embed.npy")
head = dict(np.load(f"{MODELS}/laya.act_head.npz"))
net = load(mode)

state = reference["state"]
print(f"state: {serialize(state)}")
code, answers = 0, {}
for (qid, qdef), expected in zip(reference["questions"].items(), reference["encoded"]):
    q = internal(qdef)
    ids, markers = sequence(tok, io, state, q)
    if (ids, markers) != (expected["ids"], expected["markers"]):
        print(f"  {qid}: TOKENS DIFFER FROM LAYA'S")
        code = code or 4
        continue
    t = time.time()
    out = extract(net, io, feeds(io, table, types, ids, QTYPES[q["t"]]))
    a = answers[qid] = answer(io, head, q, out["scores"], out["pooled"], markers)
    ms = (time.time() - t) * 1000
    want = reference["answers"][qid]
    got_p, want_p = probabilities(a), probabilities(want)
    diff = max(abs(x - y) for x, y in zip(got_p, want_p))
    line = f"  {qid} ({q['t']}): {a[q['t']]} p={got_p} | laya {want[q['t']]} p={want_p} | {ms:.0f} ms"
    if not all(math.isfinite(x) for x in got_p):
        line += " NON-FINITE"
        code = code or 5
    elif diff > MAX_DIFF[mode]:
        line += f" OFF BY {diff:.3f}"
        code = code or 5
    print(line)

print(json.dumps(answers, indent=2, ensure_ascii=False))
print(f"laya [{mode}]:", "OK" if code == 0 else "MISMATCHED OR NON-FINITE ANSWERS")
sys.exit(code)
