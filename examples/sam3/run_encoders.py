"""Run the ncnn SAM3 encoders under /models on the Vulkan device.

Run inside the guest as `python3 run_encoders.py MODE REQUEST`, from the sam3 skill mounted at
/skills/sam3; MODE is one of MODES below. `SAM3_MODELS` names another directory than /models,
to run it on the host. What each model takes and gives back is in its `.json`, which
`prepare_model.py` wrote beside it, with the CLIP tokenizer the language encoder's prompts are
tokenized with.

REQUEST is a JSON `{"images": [path, ...], "texts": [prompt, ...], "out": path}`. Each image
goes through the image encoder, each prompt through the language encoder, and what they give
back is saved to `out` as one `.npz`, with the inputs the decoder takes beside them: the image
encoder's position encodings, which are the same for every image, and each prompt's padding
mask. stdout is one JSON object saying what went in and came out, an input that could not be
taken answered with `{"error": ...}`, and all else goes to stderr.

Run as `python3 run_encoders.py MODEL MODE [REFERENCE]`, it checks MODEL instead: it runs once
on a fixed input, keeps the outputs in `/tmp/sam3/MODEL.MODE.npz`, and with a REFERENCE mode
compares them to that one's, output by output, by cosine similarity.

One mode a process, because ncnn does not take a bf16 net after an fp32 one in the same
process: extracting from the second segfaults, where either alone, or bf16 first, is fine.
fp16 is not a mode. On Venus the language encoder's `text_memory` comes back NaN or wrong
with it (cos 0.38 to onnxruntime), and the image encoder's backbone NaN or a crash; in fp32
both match onnxruntime to cos 0.9999 and better.

Exits 2 if the wheel has no Vulkan, 3 if it finds no device, 5 on a non-finite output or, when
checking, one too far from the reference's.
"""

import json
import os
import sys
import time

import numpy as np
import ncnn

MODELS = os.environ.get("SAM3_MODELS", "/models")
IMAGE, LANGUAGE = "sam3_image_encoder", "sam3_language_encoder"
OUT = "/tmp/sam3"
# Below this against the reference, a mode counts as wrong.
MIN_COS = 0.99
# SAM3's prompts are CLIP's, cut to this many tokens.
CONTEXT = 32
SOT, EOT = 49406, 49407

# Whether to store in bf16. The arithmetic stays fp32, bar Gemm and SDPA, which go to
# bf16 x bf16 -> fp32 cooperative matrices when the device has them: `use_cooperative_matrix`
# is on by default and the wheel binds no way to turn it off.
MODES = {"fp32": False, "bf16": True}


def device():
    """The Vulkan device ncnn will run on, or the reason there is none."""
    # A wheel built without NCNN_VULKAN has no GPU API at all.
    if not hasattr(ncnn, "get_gpu_count"):
        print("vulkan: NOT BUILT IN (no ncnn.get_gpu_count)", file=sys.stderr)
        sys.exit(2)
    if ncnn.get_gpu_count() == 0:
        # The wheel has Vulkan but found no device: the loader or the guest driver is missing.
        print("vulkan: no device", file=sys.stderr)
        sys.exit(3)
    print(f"ncnn {getattr(ncnn, '__version__', '?')} on {ncnn.get_gpu_info(0).device_name()}", file=sys.stderr)


def load(name: str, mode: str):
    with open(f"{MODELS}/{name}.json") as f:
        io = json.load(f)
    net = ncnn.Net()
    net.opt.use_vulkan_compute = True
    net.opt.use_fp16_storage = False
    net.opt.use_fp16_packed = False
    net.opt.use_fp16_arithmetic = False
    # Quietly dropped by ncnn if the device has no `shaderBFloat16Type`, which is fp32 again.
    net.opt.use_bf16_storage = MODES[mode]
    net.opt.use_bf16_packed = MODES[mode]
    net.set_vulkan_device(0)
    t = time.time()
    assert net.load_param(f"{MODELS}/{name}.ncnn.param") == 0, f"{name}: load_param failed"
    assert net.load_model(f"{MODELS}/{name}.ncnn.bin") == 0, f"{name}: load_model failed"
    print(f"{name} [{mode}]: loaded in {time.time() - t:.1f}s", file=sys.stderr)
    return net, io


# --- What goes in -------------------------------------------------------------------------------


def tokens(tok, text: str) -> np.ndarray:
    """`text` as CLIP tokenizes it, start and end of text included, zero-padded to CONTEXT.

    `tokenizer.json` gives the same ids as CLIP's own `SimpleTokenizer`, which SAM3 uses. A
    longer prompt is cut and still ends in the end of text, where SAM3 truncates.
    """
    ids = tok.encode(text).ids
    if len(ids) > CONTEXT:
        ids = ids[: CONTEXT - 1] + [EOT]
    out = np.zeros((1, CONTEXT), np.int64)
    out[0, : len(ids)] = ids
    return out


def language_feeds(io, ids: np.ndarray) -> dict:
    # The token embeddings `prepare_model.py` peeled off the model, looked up here.
    return {n: tables[t["table"]][ids] for n, t in io["tables"].items()}


def image_feeds(io, path: str) -> tuple[dict, list[int]]:
    """The image as the encoder takes it, and its size before: RGB, stretched to the input's
    square as SAM3 does, as floats in 0..255, channels first."""
    from PIL import Image

    spec = io["inputs"][0]
    h, w = spec["shape"][-2:]
    with Image.open(path) as im:
        size = list(im.size)
        pixels = np.asarray(im.convert("RGB").resize((w, h)), np.float32)
    return {spec["name"]: pixels.transpose(2, 0, 1)[None]}, size


# --- The model ----------------------------------------------------------------------------------


def run(name: str, net, io, feeds: dict) -> dict:
    ex = net.create_extractor()
    for spec in io["inputs"]:
        # Named, not inlined: `ncnn.Mat` borrows the array's memory without holding it, so an
        # array made in the same expression is freed before `clone` copies it.
        x = np.ascontiguousarray(feeds[spec["name"]], np.float32).reshape(spec["ncnn_shape"])
        ex.input(spec["name"], ncnn.Mat(x).clone())
    out = {}
    for spec in io["outputs"]:
        ret, m = ex.extract(spec["name"])
        assert ret == 0, f"{name}: extract {spec['name']} failed: {ret}"
        out[spec["name"]] = np.array(m).reshape(spec["shape"])
    return out


def summary(out: dict) -> dict:
    return {n: {"shape": list(v.shape), "mean": round(float(v.mean()), 4), "std": round(float(v.std()), 4)} for n, v in out.items()}


def finite(out: dict) -> bool:
    return all(bool(np.isfinite(v).all()) for v in out.values())


def encode(mode: str, request: dict) -> tuple[dict, int]:
    """Run `request` through the encoders, save what they give back, and say what that was."""
    images, texts = request.get("images") or [], request.get("texts") or []
    assert images or texts, "the request has neither images nor texts"
    assert request.get("out"), "the request has no `out` to save to"
    arrays, report, code = {}, {"mode": mode, "images": [], "texts": []}, 0

    # One model at a time, so the guest never holds both.
    if images:
        net, io = load(IMAGE, mode)
        for n, value in io["constants"].items():
            arrays[n] = np.load(f"{MODELS}/{value}")
        for i, path in enumerate(images):
            try:
                feeds, size = image_feeds(io, path)
                t = time.time()
                out = run(IMAGE, net, io, feeds)
                arrays.update({f"image{i}.{n}": v for n, v in out.items()})
                ok = finite(out)
                code = code or (0 if ok else 5)
                report["images"].append({"key": f"image{i}", "path": path, "size": size, "seconds": round(time.time() - t, 2), "outputs": summary(out), "finite": ok})
            except Exception as e:
                report["images"].append({"key": f"image{i}", "path": path, "error": str(e) or type(e).__name__})
        del net

    if texts:
        from tokenizers import Tokenizer

        tok = Tokenizer.from_file(f"{MODELS}/tokenizer.json")
        net, io = load(LANGUAGE, mode)
        for i, text in enumerate(texts):
            try:
                ids = tokens(tok, text)
                t = time.time()
                out = run(LANGUAGE, net, io, language_feeds(io, ids))
                # The padding mask the model's graph had, `tokens == 0`: see `prepare_model.py`.
                arrays.update({f"text{i}.{n}": v for n, v in out.items()})
                arrays[f"text{i}.{io['dropped'][0]}"] = ids == 0
                ok = finite(out)
                code = code or (0 if ok else 5)
                report["texts"].append({"key": f"text{i}", "text": text, "tokens": int((ids != 0).sum()), "seconds": round(time.time() - t, 2), "outputs": summary(out), "finite": ok})
            except Exception as e:
                report["texts"].append({"key": f"text{i}", "text": text, "error": str(e) or type(e).__name__})
        del net

    os.makedirs(os.path.dirname(os.path.abspath(request["out"])), exist_ok=True)
    np.savez(request["out"], **arrays)
    report["out"] = request["out"]
    return report, code


# --- Checking one model against another mode ---------------------------------------------------


def check_feeds(name: str, io) -> dict:
    if name == LANGUAGE:
        # CLIP's start and end of text and nothing between: the tokenizer is not the point here.
        ids = np.zeros((1, CONTEXT), np.int64)
        ids[0, :2] = [SOT, EOT]
        return language_feeds(io, ids)
    image = np.random.default_rng(0).integers(0, 256, (1, 3, 1008, 1008)).astype(np.float32)
    return {io["inputs"][0]["name"]: image}


def cos(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.ravel().astype(np.float64), b.ravel().astype(np.float64)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))


def check(name: str, mode: str, reference: str | None) -> int:
    net, io = load(name, mode)
    t = time.time()
    out = run(name, net, io, check_feeds(name, io))
    print(f"{name} [{mode}]: ran in {time.time() - t:.1f}s")
    os.makedirs(OUT, exist_ok=True)
    np.savez(f"{OUT}/{name}.{mode}.npz", **out)
    ref = np.load(f"{OUT}/{name}.{reference}.npz") if reference else None

    ok = True
    for n, v in out.items():
        line = f"  {n} {list(v.shape)} mean={v.mean():.4f} std={v.std():.4f}"
        good = bool(np.isfinite(v).all())
        if not good:
            line += " NON-FINITE"
        elif ref is not None:
            c = cos(v, ref[n])
            line += f" cos={c:.5f} to {reference}"
            good = c >= MIN_COS
        print(line)
        ok &= good

    print(f"{name} [{mode}]:", "OK" if ok else "NON-FINITE OR MISMATCHED OUTPUT")
    return 0 if ok else 5


class Tables(dict):
    """The language encoder's embedding tables, each read from /models the first time."""

    def __missing__(self, path: str) -> np.ndarray:
        self[path] = np.load(f"{MODELS}/{path}")
        return self[path]


tables = Tables()

if sys.argv[1] in (IMAGE, LANGUAGE):
    name, mode, *reference = sys.argv[1:]
    assert mode in MODES, f"unknown mode {mode}, not in {list(MODES)}"
    device()
    sys.exit(check(name, mode, reference[0] if reference else None))

mode, request = sys.argv[1], json.loads(sys.argv[2])
assert mode in MODES, f"unknown mode {mode}, not in {list(MODES)}"
device()
report, code = encode(mode, request)
print(json.dumps(report, ensure_ascii=False))
sys.exit(code)
