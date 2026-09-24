"""Download YOLOE, set its classes from text, and convert it into an ncnn model.

    uv run prepare_model.py [DATA_DIR]

DATA_DIR is `data/` beside this file by default. YOLOE is an open-vocabulary detector: what it
looks for is a list of class names in plain text, which MobileCLIP turns into embeddings and
YOLOE folds into its head. That happens here, on the host, so the ncnn model finds the classes
it was converted with and no others; change them and this converts again.

* `YOLOE_CLASSES` -- the class names, comma-separated. CLASSES below by default.
* `YOLOE_IMAGE` -- the image the reference is taken on. Ultralytics' own `bus.jpg` by default.

The weights (`yoloe-11l-seg.pt`, the large one) and the text encoder (`mobileclip_blt.ts`, ~600 MB)
go into `DATA_DIR/weights`, from the Ultralytics assets release this `ultralytics` pins, and
this writes into `DATA_DIR/ncnn`

* `yoloe.ncnn.param`, `yoloe.ncnn.bin` -- the model at 640x640, weights in fp16;
* `yoloe.json` -- its class names, input size and outputs;
* `image.jpg` -- the image, and `yoloe.reference.json` -- what Ultralytics finds in it with the
  PyTorch model, the classes and the image it was taken on.

**Ultralytics' own export.** The conversion is `model.export(format="ncnn")`, which prepares the
model as Ultralytics does -- fused, its head in export mode with the text embeddings folded in
-- and traces it for pnnx. Only the pnnx it runs is not its own choice: it downloads the latest
pnnx release unless there is a `pnnx` in the working directory, so the one from this project's
pinned `pnnx` package is linked there first.

The ncnn model is YOLOE-seg's: boxes, class scores and mask coefficients for 8400 anchors, and
the mask prototypes. The caller decodes the boxes and runs NMS; the masks are there if it wants
them.
"""

import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

# Ultralytics pip-installs what it finds missing, into whatever environment it runs in; what
# this needs is in pyproject.toml instead.
os.environ["YOLO_AUTOINSTALL"] = "false"

import ultralytics  # noqa: E402
from ultralytics import YOLOE  # noqa: E402

WEIGHTS = "yoloe-11l-seg.pt"
IMGSZ = 640
CLASSES = ["person", "bus", "glasses", "backpack", "traffic light"]
# Ultralytics' predict defaults, which the caller's NMS uses too.
CONF, IOU = 0.25, 0.7


def request() -> dict:
    classes = [c.strip() for c in os.environ.get("YOLOE_CLASSES", ",".join(CLASSES)).split(",") if c.strip()]
    image = Path(os.environ.get("YOLOE_IMAGE") or Path(ultralytics.__file__).parent / "assets" / "bus.jpg")
    return {"weights": WEIGHTS, "classes": classes, "image": image, "image_sha256": hashlib.sha256(image.read_bytes()).hexdigest()}


def converted(ncnn_dir: Path, req: dict) -> bool:
    """Whether `ncnn_dir` holds a model for these classes and a reference for this image."""
    try:
        ref = json.loads((ncnn_dir / "yoloe.reference.json").read_text())
    except (OSError, ValueError):
        return False
    same = all(ref.get(k) == req[k] for k in ("weights", "classes", "image_sha256"))
    return same and (ncnn_dir / "yoloe.ncnn.bin").exists()


def main():
    data = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent / "data"
    weights, ncnn_dir = (data / "weights").resolve(), (data / "ncnn").resolve()
    req = request()
    if converted(ncnn_dir, req):
        print(f"yoloe: already converted for {req['classes']}")
        return
    print(f"yoloe: converting for {req['classes']}", flush=True)
    weights.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(ncnn_dir, ignore_errors=True)
    ncnn_dir.mkdir(parents=True)

    import pnnx

    link = weights / "pnnx"
    link.unlink(missing_ok=True)
    link.symlink_to(Path(pnnx.__file__).parent / "pnnx")
    # Ultralytics downloads the weights and the text encoder into, and looks for pnnx in, the
    # working directory; and it writes the export beside the weights.
    os.chdir(weights)

    model = YOLOE(WEIGHTS)
    model.set_classes(req["classes"], model.get_text_pe(req["classes"]))

    # The reference: Ultralytics' predict with the PyTorch model, letterboxed to the square
    # the ncnn model takes (`rect=False`) rather than to the smallest one that fits.
    result = model.predict(str(req["image"]), imgsz=IMGSZ, rect=False, conf=CONF, iou=IOU, verbose=False)[0]
    boxes = result.boxes
    detections = [
        {"class": req["classes"][int(c)], "confidence": round(float(p), 4), "box": [round(float(v), 1) for v in xyxy]}
        for c, p, xyxy in zip(boxes.cls, boxes.conf, boxes.xyxy)
    ]

    exported = Path(model.export(format="ncnn", imgsz=IMGSZ, half=True))
    shutil.move(exported / "model.ncnn.param", ncnn_dir / "yoloe.ncnn.param")
    shutil.copy(req["image"], ncnn_dir / "image.jpg")
    io = {
        "names": req["classes"],
        "imgsz": IMGSZ,
        "conf": CONF,
        "iou": IOU,
        # pnnx drops the batch dim: [4 box + classes + 32 mask coefficients, anchors], and the
        # mask prototypes [32, 160, 160].
        "outputs": {"predictions": "out0", "prototypes": "out1"},
        "input": "in0",
    }
    (ncnn_dir / "yoloe.json").write_text(json.dumps(io, indent=2) + "\n")
    reference = {
        "weights": WEIGHTS,
        "classes": req["classes"],
        "image_sha256": req["image_sha256"],
        "size": list(result.orig_shape),
        "detections": detections,
    }
    (ncnn_dir / "yoloe.reference.json").write_text(json.dumps(reference, indent=2) + "\n")
    # The `.bin` is what says the model is done, so it appears once the rest is in place.
    shutil.move(exported / "model.ncnn.bin", ncnn_dir / "yoloe.ncnn.bin")
    shutil.rmtree(exported)
    Path(WEIGHTS).with_suffix(".torchscript").unlink(missing_ok=True)

    for d in detections:
        print(f"  {d['class']} {d['confidence']:.3f} {d['box']}")


if __name__ == "__main__":
    main()
