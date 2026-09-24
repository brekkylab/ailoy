"""Load one ncnn SAM3 encoder under /models and run it once on the Vulkan device, in one mode.

Run inside the guest as `python3 -c RUN MODEL MODE [REFERENCE]`, so it has no file of its own
there. What each model takes and gives back is in its `.json`, which `prepare_model.py` wrote
beside it. The outputs are kept in `/tmp/sam3/MODEL.MODE.npz`, and with a REFERENCE mode they
are compared to that one's, output by output, by cosine similarity.

One model in one mode a process, because ncnn does not take a bf16 net after an fp32 one in
the same process: extracting from the second segfaults, where either alone, or bf16 first, is
fine. fp16 is not a mode. On Venus the language encoder's `text_memory` comes back NaN or wrong
with it (cos 0.38 to onnxruntime), and the image encoder's backbone NaN or a crash; in fp32
both match onnxruntime to cos 0.9999 and better.

Exits 2 if the wheel has no Vulkan, 3 if it finds no device, 5 on a non-finite output or one
too far from the reference's.
"""

import json
import os
import sys
import time

import numpy as np
import ncnn

MODELS = "/models"
OUT = "/tmp/sam3"
# Below this against the reference, a mode counts as wrong.
MIN_COS = 0.99

# Whether to store in bf16. The arithmetic stays fp32, bar Gemm and SDPA, which go to
# bf16 x bf16 -> fp32 cooperative matrices when the device has them: `use_cooperative_matrix`
# is on by default and the wheel binds no way to turn it off.
MODES = {"fp32": False, "bf16": True}


def device():
    """The Vulkan device ncnn will run on, or the reason there is none."""
    # A wheel built without NCNN_VULKAN has no GPU API at all.
    if not hasattr(ncnn, "get_gpu_count"):
        print("vulkan: NOT BUILT IN (no ncnn.get_gpu_count)")
        sys.exit(2)
    if ncnn.get_gpu_count() == 0:
        # The wheel has Vulkan but found no device: the loader or the guest driver is missing.
        print("vulkan: no device")
        sys.exit(3)
    print(f"ncnn {getattr(ncnn, '__version__', '?')} on {ncnn.get_gpu_info(0).device_name()}")


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
    print(f"{name} [{mode}]: loaded in {time.time() - t:.1f}s")
    return net, io


def feeds(name: str, io) -> dict:
    if name == "sam3_language_encoder":
        # CLIP's start and end of text and nothing between: the tokenizer is not the point here.
        tokens = np.zeros((1, 32), np.int64)
        tokens[0, :2] = [49406, 49407]
        return {n: np.load(f"{MODELS}/{t['table']}")[tokens] for n, t in io["tables"].items()}
    image = np.random.default_rng(0).integers(0, 256, (1, 3, 1008, 1008)).astype(np.float32)
    return {io["inputs"][0]["name"]: image}


def run(name: str, mode: str, net, io, feeds: dict) -> dict:
    ex = net.create_extractor()
    for spec in io["inputs"]:
        # Named, not inlined: `ncnn.Mat` borrows the array's memory without holding it, so an
        # array made in the same expression is freed before `clone` copies it.
        x = np.ascontiguousarray(feeds[spec["name"]], np.float32).reshape(spec["ncnn_shape"])
        ex.input(spec["name"], ncnn.Mat(x).clone())
    out, t = {}, time.time()
    for spec in io["outputs"]:
        ret, m = ex.extract(spec["name"])
        assert ret == 0, f"{name}: extract {spec['name']} failed: {ret}"
        out[spec["name"]] = np.array(m).reshape(spec["shape"])
    print(f"{name} [{mode}]: ran in {time.time() - t:.1f}s")
    return out


def cos(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.ravel().astype(np.float64), b.ravel().astype(np.float64)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))


name, mode, *reference = sys.argv[1:]
assert mode in MODES, f"unknown mode {mode}, not in {list(MODES)}"

device()
net, io = load(name, mode)
out = run(name, mode, net, io, feeds(name, io))
os.makedirs(OUT, exist_ok=True)
np.savez(f"{OUT}/{name}.{mode}.npz", **out)
ref = np.load(f"{OUT}/{name}.{reference[0]}.npz") if reference else None

ok = True
for n, v in out.items():
    line = f"  {n} {list(v.shape)} mean={v.mean():.4f} std={v.std():.4f}"
    good = bool(np.isfinite(v).all())
    if not good:
        line += " NON-FINITE"
    elif ref is not None:
        c = cos(v, ref[n])
        line += f" cos={c:.5f} to {reference[0]}"
        good = c >= MIN_COS
    print(line)
    ok &= good

print(f"{name} [{mode}]:", "OK" if ok else "NON-FINITE OR MISMATCHED OUTPUT")
sys.exit(0 if ok else 5)
