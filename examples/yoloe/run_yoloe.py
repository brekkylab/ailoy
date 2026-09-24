"""Find the classes YOLOE was converted with in the image under /models, on the Vulkan device.

Run inside the guest as `python3 -c RUN MODE`, so it has no file of its own there; MODE is one
of MODES below. `YOLOE_MODELS` names another directory than /models, to run it on the host.
The class names, the input size and the NMS thresholds are in `yoloe.json`, which
`prepare_model.py` wrote beside the model; what Ultralytics found in the image with the
PyTorch model is in `yoloe.reference.json`.

The steps around the model are Ultralytics' own, in numpy and OpenCV: the image letterboxed
to the square the model takes, and the boxes of the anchors that clear the confidence
threshold kept by class-wise NMS and scaled back to the image. Each detection is then matched
to the reference's.

Exits 2 if the wheel has no Vulkan, 3 if it finds no device, 5 on a non-finite output, or on a
detection the reference does not have or one it has that is missing or too far off.
"""

import json
import os
import sys
import time

import cv2
import ncnn
import numpy as np

MODELS = os.environ.get("YOLOE_MODELS", "/models")

# Storage precision; the arithmetic follows it in fp16 and stays fp32 in bf16. ncnn quietly
# drops fp16 or bf16 if the device has no such type, which is fp32 again, and the
# `fp16-p/s/u/a` and `bf16-p/s` lines on stderr say so.
MODES = {"fp32": None, "fp16": "fp16", "bf16": "bf16"}

# A detection matches the reference's of the same class at this IoU, within this of its
# confidence, by mode. bf16's boxes move by a few pixels, which on a small one -- the glasses in
# `bus.jpg` are 39x13 -- is an IoU of 0.7, and can leave a second box on the same object that
# NMS no longer takes for the first: one that overlaps a detection of the reference's as much
# is a duplicate, not something the reference missed. Below MARGIN over the threshold, one may
# be found on one side and not the other.
MIN_IOU = {"fp32": 0.95, "fp16": 0.9, "bf16": 0.6}
MAX_CONF_DIFF = {"fp32": 0.01, "fp16": 0.05, "bf16": 0.15}
MARGIN = 0.1


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
    half = MODES[mode]
    net = ncnn.Net()
    net.opt.use_vulkan_compute = True
    net.opt.use_fp16_storage = net.opt.use_fp16_packed = net.opt.use_fp16_arithmetic = half == "fp16"
    net.opt.use_bf16_storage = net.opt.use_bf16_packed = half == "bf16"
    net.set_vulkan_device(0)
    t = time.time()
    assert net.load_param(f"{MODELS}/yoloe.ncnn.param") == 0, "load_param failed"
    assert net.load_model(f"{MODELS}/yoloe.ncnn.bin") == 0, "load_model failed"
    print(f"yoloe [{mode}]: loaded in {time.time() - t:.1f}s")
    return net


def letterbox(image: np.ndarray, size: int) -> tuple[np.ndarray, float, tuple[int, int]]:
    """Ultralytics' `LetterBox(auto=False)`: scaled to fit, centered, padded with 114."""
    h, w = image.shape[:2]
    r = min(size / h, size / w)
    unpad = int(round(w * r)), int(round(h * r))
    dw, dh = (size - unpad[0]) / 2, (size - unpad[1]) / 2
    if (w, h) != unpad:
        image = cv2.resize(image, unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
    return image, r, (left, top)


def iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    lt = np.maximum(box[:2], boxes[:, :2])
    rb = np.minimum(box[2:], boxes[:, 2:])
    inter = np.prod(np.clip(rb - lt, 0, None), axis=1)
    area = lambda b: (b[..., 2] - b[..., 0]) * (b[..., 3] - b[..., 1])
    return inter / (area(box) + area(boxes) - inter + 1e-9)


def nms(boxes: np.ndarray, scores: np.ndarray, classes: np.ndarray, threshold: float) -> list[int]:
    """Class-wise NMS, as Ultralytics does it: each class's boxes moved apart from the others'."""
    shifted = boxes + classes[:, None] * 7680.0
    order, keep = list(np.argsort(-scores)), []
    while order:
        i = order.pop(0)
        keep.append(i)
        if order:
            rest = np.array(order)
            order = list(rest[iou(shifted[i], shifted[rest]) <= threshold])
    return keep


def detect(net, io: dict, image: np.ndarray) -> tuple[list[dict], float]:
    size, names = io["imgsz"], io["names"]
    boxed, r, (left, top) = letterbox(image, size)
    x = np.ascontiguousarray(boxed[:, :, ::-1].transpose(2, 0, 1), np.float32) / 255.0
    ex = net.create_extractor()
    ex.input(io["input"], ncnn.Mat(x).clone())
    t = time.time()
    ret, out = ex.extract(io["outputs"]["predictions"])
    ms = (time.time() - t) * 1000
    assert ret == 0, f"extract failed: {ret}"
    # [4 box + classes + 32 mask coefficients, anchors], the box as center and size in pixels.
    pred = np.array(out)
    if not np.isfinite(pred).all():
        return None, ms
    scores_all = pred[4 : 4 + len(names)].T
    classes = scores_all.argmax(1)
    scores = scores_all.max(1)
    kept = scores > io["conf"]
    cxcywh, classes, scores = pred[:4].T[kept], classes[kept], scores[kept]
    boxes = np.concatenate([cxcywh[:, :2] - cxcywh[:, 2:] / 2, cxcywh[:, :2] + cxcywh[:, 2:] / 2], 1)
    found = []
    h, w = image.shape[:2]
    for i in nms(boxes, scores, classes, io["iou"]):
        b = (boxes[i] - [left, top, left, top]) / r
        b = np.clip(b, 0, [w, h, w, h])
        found.append({"class": names[classes[i]], "confidence": round(float(scores[i]), 4), "box": [round(float(v), 1) for v in b]})
    return found, ms


def compare(found: list[dict], reference: dict, conf: float, min_iou: float, max_diff: float) -> bool:
    """Every detection clear of the threshold on either side matched on the other, but for
    duplicates of one that is."""
    ok, matched = True, set()
    for d in found:
        same = [(k, r) for k, r in enumerate(reference["detections"]) if r["class"] == d["class"] and k not in matched]
        best = max(same, key=lambda kr: iou(np.array(d["box"]), np.array([kr[1]["box"]]))[0], default=None)
        line = f"  {d['class']} {d['confidence']:.3f} {d['box']}"
        if best is not None:
            k, r = best
            overlap = iou(np.array(d["box"]), np.array([r["box"]]))[0]
            if overlap >= min_iou:
                matched.add(k)
                diff = abs(d["confidence"] - r["confidence"])
                line += f" | reference {r['confidence']:.3f} iou={overlap:.3f}"
                if diff > max_diff:
                    line += f" CONFIDENCE OFF BY {diff:.3f}"
                    ok = False
                print(line)
                continue
        of_class = [r for r in reference["detections"] if r["class"] == d["class"]]
        if any(iou(np.array(d["box"]), np.array([r["box"]]))[0] >= min_iou for r in of_class):
            line += " (a duplicate)"
        elif d["confidence"] >= conf + MARGIN:
            line += " NOT IN THE REFERENCE"
            ok = False
        print(line)
    for k, r in enumerate(reference["detections"]):
        if k not in matched:
            missing = r["confidence"] >= conf + MARGIN
            print(f"  (reference) {r['class']} {r['confidence']:.3f} {r['box']}{' MISSING' if missing else ''}")
            ok &= not missing
    return ok


mode = sys.argv[1]
assert mode in MODES, f"unknown mode {mode}, not in {list(MODES)}"

device()
with open(f"{MODELS}/yoloe.json") as f:
    io = json.load(f)
with open(f"{MODELS}/yoloe.reference.json") as f:
    reference = json.load(f)
image = cv2.imread(f"{MODELS}/image.jpg")
print(f"looking for {io['names']} in a {image.shape[1]}x{image.shape[0]} image")
net = load(mode)

found, ms = detect(net, io, image)
if found is None:
    print(f"yoloe [{mode}]: NON-FINITE OUTPUT")
    sys.exit(5)
print(f"yoloe [{mode}]: {len(found)} found in {ms:.0f} ms")
ok = compare(found, reference, io["conf"], MIN_IOU[mode], MAX_CONF_DIFF[mode])
print(f"yoloe [{mode}]:", "OK" if ok else "DETECTIONS DIFFER FROM THE REFERENCE")
sys.exit(0 if ok else 5)
