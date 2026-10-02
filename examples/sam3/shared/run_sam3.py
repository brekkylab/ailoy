"""Segment with the ncnn SAM3 pieces under /models, on the Vulkan device.

Run inside the guest as `python3 run_sam3.py MODE REQUEST`, from the sam3 skill mounted at
/skills/sam3; MODE is one of MODES below and REQUEST a JSON object whose `task` is one of

* `detect` -- every instance of a text prompt, or of example boxes, in an image: the detector;
* `segment` -- one object an image's points, box or mask pick out, for each object given: the
  tracker, SAM 2's mask decoder;
* `track` -- objects through a video's frames, from points, a box, a mask or a text prompt on
  one frame: the tracker on features conditioned on a memory of earlier frames.

`SAM3_MODELS` names another directory than /models, to run it on the host. What each piece takes
and gives back, and the settings the steps around them need, are in `sam3.json`, which
`prepare_model.py` wrote beside them, with the weights those steps use in `sam3.npz`. The steps
are transformers' `Sam3Model`, `Sam3TrackerModel` and `Sam3TrackerVideoModel`, redone in numpy
around the pieces: the prompt encoder, resizing, thresholds, and which memories a frame attends
to. Where transformers' image tracker adds `no_memory_embedding` to a level of features it does
not use, this adds it to the one it does, as SAM3's own `sam1_task_predictor` does.

stdout is one JSON object saying what was found and where it was written, and all else goes to
stderr. Masks are written as PNGs under the request's `out` directory, with an overlay of them on
the image or on each frame.

Exits 2 if the wheel has no Vulkan, 3 if it finds no device, 4 on a request it cannot take, 5 on
a non-finite output.
"""

import json
import math
import os
import sys
import time
from pathlib import Path

import ncnn
import numpy as np
from PIL import Image

MODELS = os.environ.get("SAM3_MODELS", "/models")

# Whether to store in bf16. The arithmetic stays fp32, bar Gemm and SDPA, which go to bf16 x bf16
# -> fp32 cooperative matrices when the device has them. Quietly dropped by ncnn if the device
# has no `shaderBFloat16Type`, which is fp32 again, and the `bf16-p/s` line on stderr says so.
# fp16 is not a mode: SAM3's activations leave its range.
MODES = {"fp32": False, "bf16": True}

# What a mask logit is where the tracker sees no object.
NO_OBJ_SCORE = -1024.0


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


class Pieces:
    """The ncnn pieces, each loaded the first time it is run."""

    def __init__(self, mode: str):
        with open(f"{MODELS}/sam3.json") as f:
            self.config = json.load(f)
        self.io = self.config["pieces"]
        self.mode = mode
        self.nets = {}
        self.consts = dict(np.load(f"{MODELS}/sam3.npz"))

    def net(self, name: str):
        if name not in self.nets:
            net = ncnn.Net()
            net.opt.use_vulkan_compute = True
            net.opt.use_fp16_storage = False
            net.opt.use_fp16_packed = False
            net.opt.use_fp16_arithmetic = False
            net.opt.use_bf16_storage = MODES[self.mode]
            net.opt.use_bf16_packed = MODES[self.mode]
            net.set_vulkan_device(0)
            t = time.time()
            assert net.load_param(f"{MODELS}/sam3_{name}.ncnn.param") == 0, f"{name}: load_param failed"
            assert net.load_model(f"{MODELS}/sam3_{name}.ncnn.bin") == 0, f"{name}: load_model failed"
            log(f"sam3_{name} [{self.mode}]: loaded in {time.time() - t:.1f}s")
            self.nets[name] = net
        return self.nets[name]

    def drop(self, *names: str):
        """Free pieces no longer needed: the guest holds each one's weights while it is loaded."""
        for name in names:
            self.nets.pop(name, None)

    def run(self, name: str, **feeds) -> dict:
        """Run a piece on its inputs by role, without their batch of one, and give back its
        outputs by role."""
        io = self.io[name]
        ex = self.net(name).create_extractor()
        for spec in io["inputs"]:
            # Named, not inlined: `ncnn.Mat` borrows the array's memory without holding it, so an
            # array made in the same expression is freed before `clone` copies it.
            x = np.ascontiguousarray(feeds[spec["role"]], np.float32)
            ex.input(spec["name"], ncnn.Mat(x).clone())
        out = {}
        for spec in io["outputs"]:
            ret, m = ex.extract(spec["name"])
            assert ret == 0, f"{name}: extract {spec['role']} failed: {ret}"
            a = np.array(m)
            if not np.isfinite(a).all():
                log(f"{name}: {spec['role']} is not finite")
                sys.exit(5)
            out[spec["role"]] = a
        return out


# --- Resizing, as torch's `interpolate(mode="bilinear", align_corners=False)` ----------------


def resize_matrix(n_in: int, n_out: int, antialias: bool) -> np.ndarray:
    """The [n_out, n_in] matrix that resizes one axis, antialiased or not."""
    m = np.zeros((n_out, n_in), np.float64)
    scale = n_in / n_out
    if antialias:
        # `_upsample_bilinear2d_aa`: a triangle filter, as wide as the scale when shrinking
        support = scale if scale >= 1 else 1.0
        inv = 1 / scale if scale >= 1 else 1.0
        for i in range(n_out):
            center = scale * (i + 0.5)
            lo = max(int(center - support + 0.5), 0)
            hi = min(int(center + support + 0.5), n_in)
            j = np.arange(lo, hi)
            w = np.maximum(0.0, 1.0 - np.abs((j - center + 0.5) * inv))
            m[i, lo:hi] = w / w.sum() if w.sum() > 0 else 0
    else:
        src = np.maximum((np.arange(n_out) + 0.5) * scale - 0.5, 0)
        i0 = np.minimum(np.floor(src).astype(int), n_in - 1)
        i1 = np.minimum(i0 + 1, n_in - 1)
        lam = src - i0
        rows = np.arange(n_out)
        np.add.at(m, (rows, i0), 1 - lam)
        np.add.at(m, (rows, i1), lam)
    return m.astype(np.float32)


_matrices = {}


def resize(x: np.ndarray, size: tuple[int, int], antialias: bool = False) -> np.ndarray:
    """[..., H, W] to [..., h, w]."""
    h, w = size
    key_h, key_w = (x.shape[-2], h, antialias), (x.shape[-1], w, antialias)
    for key in (key_h, key_w):
        if key not in _matrices:
            _matrices[key] = resize_matrix(*key)
    return _matrices[key_h] @ x.astype(np.float32) @ _matrices[key_w].T


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def bf16(x: np.ndarray) -> np.ndarray:
    """x rounded to bfloat16, as the tracker keeps its memories."""
    u = np.ascontiguousarray(x, np.float32).view(np.uint32)
    u = (u + 0x7FFF + ((u >> 16) & 1)) & 0xFFFF0000
    return u.view(np.float32)


# --- Images ---------------------------------------------------------------------------------


def load_image(path: str, size: int):
    """The image as RGB, and as the vision piece takes it: stretched to a square, as SAM3
    does, and rounded back to 0..255, as its processor resizes 8-bit pixels."""
    with Image.open(path) as im:
        rgb = np.asarray(im.convert("RGB"))
    x = rgb.transpose(2, 0, 1).astype(np.float32)
    x = np.clip(np.round(resize(x, (size, size), antialias=True)), 0, 255)
    return rgb, x


def features(p: Pieces, pixels: np.ndarray) -> dict:
    t = time.time()
    out = p.run("vision", pixels=pixels)
    log(f"vision: {time.time() - t:.1f}s")
    return out


# --- The detector ---------------------------------------------------------------------------


class Text:
    def __init__(self, p: Pieces):
        from tokenizers import Tokenizer

        self.p = p
        self.tok = Tokenizer.from_file(f"{MODELS}/tokenizer.json")
        self.n = p.config["tokens"]

    def __call__(self, text: str):
        """The prompt's features for the detector, and which of its tokens are not padding."""
        c = self.p.consts
        ids = self.tok.encode(text).ids
        if len(ids) > self.n:
            ids = ids[: self.n - 1] + ids[-1:]
        valid = np.zeros(self.n, np.float32)
        valid[: len(ids)] = 1
        eot = ids[-1]
        ids = ids + [eot] * (self.n - len(ids))
        embeds = c["text_tokens"][ids].astype(np.float32) + c["text_positions"]
        return self.p.run("text", embeds=embeds)["features"], valid


def sine_1d(x: np.ndarray, n: int = 128, temperature: float = 10000.0) -> np.ndarray:
    """`encode_1d_positions`: [K] in 0..1 to [K, n], sin and cos interleaved."""
    dim_t = temperature ** (2 * (np.arange(n) // 2) / n)
    a = (x[:, None] * 2 * math.pi) / dim_t
    out = np.empty_like(a)
    out[:, 0::2], out[:, 1::2] = np.sin(a[:, 0::2]), np.cos(a[:, 1::2])
    return out


def roi_align(feat: np.ndarray, boxes: np.ndarray, size: int = 7) -> np.ndarray:
    """torchvision's `roi_align(feat, boxes, size)`, aligned=False and adaptive sampling:
    [C, H, W] and xyxy boxes in feature cells to [K, C, size, size]."""
    c, h, w = feat.shape
    flat = feat.reshape(c, -1)
    out = np.zeros((len(boxes), c, size, size), np.float32)
    for k, (x1, y1, x2, y2) in enumerate(boxes):
        rw, rh = max(x2 - x1, 1.0), max(y2 - y1, 1.0)
        bw, bh = rw / size, rh / size
        gw, gh = math.ceil(rw / size), math.ceil(rh / size)
        weights = np.zeros((size * size, h * w), np.float32)
        for ph in range(size):
            for pw in range(size):
                row = weights[ph * size + pw]
                for iy in range(gh):
                    y = y1 + ph * bh + (iy + 0.5) * bh / gh
                    for ix in range(gw):
                        x = x1 + pw * bw + (ix + 0.5) * bw / gw
                        if y < -1 or y > h or x < -1 or x > w:
                            continue
                        y, x = max(y, 0.0), max(x, 0.0)
                        yl, xl = int(y), int(x)
                        if yl >= h - 1:
                            yl = yh = h - 1
                            y = float(yl)
                        else:
                            yh = yl + 1
                        if xl >= w - 1:
                            xl = xh = w - 1
                            x = float(xl)
                        else:
                            xh = xl + 1
                        ly, lx = y - yl, x - xl
                        for yy, xx, wt in ((yl, xl, (1 - ly) * (1 - lx)), (yl, xh, (1 - ly) * lx), (yh, xl, ly * (1 - lx)), (yh, xh, ly * lx)):
                            row[yy * w + xx] += wt
                row /= max(gh * gw, 1)
        out[k] = (flat @ weights.T).reshape(c, size, size)
    return out


def geometry(p: Pieces, fpn2: np.ndarray, boxes: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Box prompts, cxcywh in 0..1 with labels 1 or 0, to prompt features [N + 1, 256]."""
    c, eps = p.consts, p.config["geometry_norm_eps"]
    x = fpn2.transpose(1, 2, 0)
    mu, var = x.mean(-1, keepdims=True), x.var(-1, keepdims=True)
    normed = ((x - mu) / np.sqrt(var + eps) * c["geometry_norm_weight"] + c["geometry_norm_bias"]).transpose(2, 0, 1)
    g = p.config["grid"]
    cx, cy, w, h = boxes.T
    xyxy = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], -1) * g
    pooled = roi_align(normed, xyxy).reshape(len(boxes), -1)
    box_pos = np.concatenate([sine_1d(cy), sine_1d(cx), h[:, None], w[:, None]], axis=1)
    label_embed = c["geometry_label_embed"][labels.astype(int)]
    return p.run("geometry", boxes=boxes, box_pos=box_pos, labels=label_embed, pooled=pooled, fpn2=fpn2)["prompt"]


def detect_prompt(p: Pieces, feats: dict, text, text_prompt: str, boxes=None, labels=None) -> dict:
    """The detector's 200 queries for one prompt: scores, xyxy boxes in 0..1, mask logits."""
    prompt, valid = text(text_prompt)
    if boxes is not None and len(boxes):
        geo = geometry(p, feats["det_fpn2"], boxes, labels)
        prompt = np.concatenate([prompt, geo], axis=0)
        valid = np.concatenate([valid, np.ones(len(geo), np.float32)])
    t = time.time()
    out = p.run("detector", fpn0=feats["det_fpn0"], fpn1=feats["det_fpn1"], fpn2=feats["det_fpn2"], prompt=prompt, valid=valid)
    log(f"detector: {time.time() - t:.1f}s")
    return {
        "scores": sigmoid(out["logits"]) * sigmoid(out["presence"][0]),
        "boxes": out["boxes"].reshape(-1, 4),
        "masks": out["masks"],
        "presence": float(sigmoid(out["presence"][0])),
    }


# --- The tracker's prompt encoder and mask decoder ------------------------------------------


class Prompts:
    """`Sam3TrackerPromptEncoder`'s sparse embeddings, from points and boxes at IMAGE scale."""

    def __init__(self, p: Pieces):
        c = p.consts
        self.gaussian, self.point_embed = c["prompt_gaussian"], c["point_embed"]
        self.not_a_point = c["not_a_point_embed"]
        self.size = p.config["image_size"]

    def pe(self, coords):
        x = (2 * (coords / self.size) - 1) @ self.gaussian * (2 * np.pi)
        return np.concatenate([np.sin(x), np.cos(x)], axis=-1)

    def points(self, points, labels, pad: bool):
        points, labels = np.asarray(points, np.float32).reshape(-1, 2) + 0.5, np.asarray(labels, int).reshape(-1)
        if pad:
            points = np.concatenate([points, np.zeros((1, 2), np.float32)])
            labels = np.concatenate([labels, [-1]])
        e = self.pe(points)
        e[labels == -1] = self.not_a_point
        e[labels == -10] = 0
        e[labels >= 0] += self.point_embed[labels[labels >= 0]]
        return e

    def box(self, box):
        corners = np.asarray(box, np.float32).reshape(2, 2) + 0.5
        e = self.pe(np.concatenate([corners, np.zeros((1, 2), np.float32)]))
        e[0] += self.point_embed[2]
        e[1] += self.point_embed[3]
        e[2] = self.not_a_point
        return e

    def __call__(self, points=None, labels=None, box=None):
        if points is None and box is None:
            points, labels = np.zeros((1, 2), np.float32), np.array([-1])
        parts = []
        if points is not None and len(points):
            parts.append(self.points(points, labels, pad=box is None))
        if box is not None:
            parts.append(self.box(box))
        return np.concatenate(parts, axis=0)


def stability(masks: np.ndarray, delta: float) -> np.ndarray:
    flat = masks.reshape(len(masks), -1)
    inter, union = (flat > delta).sum(-1), (flat > -delta).sum(-1)
    return np.where(union > 0, inter / np.maximum(union, 1), 1.0)


def decode(p: Pieces, pix, dense, feats, sparse, multimask: bool) -> dict:
    """The mask decoder, and the mask it gives: the one of the three with the best predicted IoU
    for multimask output, else token 0's, or the best of the three where token 0's is unstable,
    as `_dynamic_multimask_via_stability` has it."""
    tc = p.config["tracker"]
    out = p.run("tracker", pix=pix, dense=dense, s0=feats["trk_s0"], s1=feats["trk_s1"], sparse=sparse)
    masks, iou, pointers = out["masks"], out["iou"].reshape(-1), out["pointers"].reshape(len(out["masks"]), -1)
    score = float(out["score"].reshape(-1)[0])
    if multimask:
        best = 1 + int(np.argmax(iou[1:]))
        token = best
    else:
        best, token = 0, 0
        if stability(masks[:1], tc["stability_delta"])[0] < tc["stability_thresh"]:
            best = 1 + int(np.argmax(iou[1:]))
    return {"all_masks": masks, "all_iou": iou, "mask": masks[best], "iou": float(iou[best]), "score": score, "pointer": pointers[token]}


def mask_to_dense(p: Pieces, logits_288: np.ndarray) -> np.ndarray:
    return p.run("mask_prompt", mask=logits_288[None])["dense"]


def no_mask_dense(p: Pieces) -> np.ndarray:
    g = p.config["grid"]
    return np.broadcast_to(p.consts["no_mask_embed"][:, None, None], (len(p.consts["no_mask_embed"]), g, g))


# --- Writing results ------------------------------------------------------------------------

COLORS = np.array(
    [[230, 25, 75], [60, 180, 75], [255, 225, 25], [0, 130, 200], [245, 130, 48], [145, 30, 180],
     [70, 240, 240], [240, 50, 230], [210, 245, 60], [250, 190, 212], [0, 128, 128], [170, 110, 40]],
    np.float32,
)


def box_of(mask: np.ndarray):
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


def overlay(rgb: np.ndarray, masks: list, boxes: list, path: Path):
    from PIL import ImageDraw

    out = rgb.astype(np.float32)
    for i, m in enumerate(masks):
        out[m] = out[m] * 0.45 + COLORS[i % len(COLORS)] * 0.55
    im = Image.fromarray(out.clip(0, 255).astype(np.uint8))
    draw = ImageDraw.Draw(im)
    for i, b in enumerate(boxes):
        if b is not None:
            draw.rectangle(b, outline=tuple(int(v) for v in COLORS[i % len(COLORS)]), width=3)
    im.save(path)


def save_mask(mask: np.ndarray, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask.astype(np.uint8) * 255).save(path)


def upscale(logits: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Mask logits at any size to the image's, as the processors' `post_process_masks` do."""
    return resize(logits, size)


# --- Tasks ----------------------------------------------------------------------------------


def detect(p: Pieces, req: dict) -> dict:
    image, out = req.get("image"), Path(req.get("out") or "")
    if not image or not req.get("out"):
        raise BadRequest("detect needs `image` and `out`")
    texts = req.get("text")
    texts = [texts] if isinstance(texts, str) else list(texts or [])
    boxes_px = np.asarray(req.get("boxes") or [], np.float32).reshape(-1, 4)
    labels = np.asarray(req.get("box_labels") or [1] * len(boxes_px), np.float32)
    if not texts and not len(boxes_px):
        raise BadRequest("detect needs a `text`, or `boxes` to find more of")
    if not texts:
        texts = ["visual"]  # what SAM3 takes as the text of a prompt of boxes alone
    threshold = float(req.get("threshold", p.config["detector"]["score_threshold"]))

    rgb, pixels = load_image(image, p.config["image_size"])
    h, w = rgb.shape[:2]
    feats = features(p, pixels)
    p.drop("vision")
    boxes = None
    if len(boxes_px):
        x1, y1, x2, y2 = boxes_px.T
        boxes = np.stack([(x1 + x2) / 2 / w, (y1 + y2) / 2 / h, (x2 - x1) / w, (y2 - y1) / h], -1)
    text = Text(p)
    out.mkdir(parents=True, exist_ok=True)
    found, masks, rects = [], [], []
    for prompt in texts:
        d = detect_prompt(p, feats, text, prompt, boxes, labels)
        keep = np.nonzero(d["scores"] > threshold)[0]
        keep = keep[np.argsort(-d["scores"][keep])]
        for q in keep:
            mask = upscale(sigmoid(d["masks"][q]), (h, w)) > 0.5
            i = len(found)
            name = f"mask_{i:03d}.png"
            save_mask(mask, out / name)
            box = (d["boxes"][q] * [w, h, w, h]).round(1).tolist()
            found.append({"prompt": prompt, "score": round(float(d["scores"][q]), 4), "box": box, "area": int(mask.sum()), "mask": str(out / name)})
            masks.append(mask)
            rects.append(box)
        log(f"{prompt!r}: {len(keep)} above {threshold}, presence {d['presence']:.3f}")
    overlay(rgb, masks, rects, out / "overlay.png")
    return {"task": "detect", "image": image, "size": [w, h], "threshold": threshold, "instances": found, "overlay": str(out / "overlay.png")}


def read_mask(path: str, size: tuple[int, int]) -> np.ndarray:
    """A mask image as floats 0 or 1, resized to `size` as the video processor does."""
    with Image.open(path) as im:
        m = (np.asarray(im.convert("L")) > 127).astype(np.float32)
    if m.shape != size:
        m = (resize(m, size, antialias=True) >= 0.5).astype(np.float32)
    return m


def scaled(points, w: int, h: int, size: int) -> np.ndarray:
    return np.asarray(points, np.float32).reshape(-1, 2) * [size / w, size / h]


def segment(p: Pieces, req: dict) -> dict:
    image, out = req.get("image"), req.get("out")
    objects = req.get("objects") or ([req] if any(k in req for k in ("points", "box", "mask")) else [])
    if not image or not out or not objects:
        raise BadRequest("segment needs `image`, `out` and `objects`, each with `points`, a `box` or a `mask`")
    out = Path(out)
    size, g = p.config["image_size"], p.config["grid"]
    rgb, pixels = load_image(image, size)
    h, w = rgb.shape[:2]
    feats = features(p, pixels)
    p.drop("vision")
    prompts = Prompts(p)
    # SAM3's image predictor adds `no_mem_embed` to the last level, as its video tracker does on
    # a first frame.
    pix = feats["trk_fpn2"] + p.consts["no_memory_embedding"][:, None, None]
    out.mkdir(parents=True, exist_ok=True)
    results, masks, rects = [], [], []
    for i, obj in enumerate(objects):
        points = obj.get("points")
        labels = obj.get("point_labels", [1] * len(points or []))
        box = obj.get("box")
        sp = scaled(points, w, h, size) if points else None
        bx = scaled(box, w, h, size).reshape(-1) if box else None
        if obj.get("mask"):
            m = read_mask(obj["mask"], (h, w))
            dense = mask_to_dense(p, (resize(m, (4 * g, 4 * g), antialias=True) * 20 - 10).astype(np.float32))
        else:
            dense = no_mask_dense(p)
        npts = 0 if sp is None else len(sp)
        multimask = obj.get("multimask", bx is None and npts <= 1)
        r = decode(p, pix, dense, feats, prompts(sp, labels, bx), multimask)
        mask = upscale(r["mask"], (h, w)) > 0
        name = f"mask_{i:03d}.png"
        save_mask(mask, out / name)
        b = box_of(mask)
        results.append(
            {"object": obj.get("id", i), "iou": round(r["iou"], 4), "object_score": round(r["score"], 3), "box": b, "area": int(mask.sum()),
             "candidates_iou": [round(float(v), 4) for v in (r["all_iou"][1:] if multimask else r["all_iou"][:1])], "mask": str(out / name)}
        )
        masks.append(mask)
        rects.append(b)
    overlay(rgb, masks, rects, out / "overlay.png")
    return {"task": "segment", "image": image, "size": [w, h], "objects": results, "overlay": str(out / "overlay.png")}


class Video:
    """`Sam3TrackerVideoModel`'s inference session, over frames read one at a time.

    Each object keeps its conditioning frames' outputs -- the frames it was prompted on -- and
    the outputs of the frames it was tracked through. An output is the low-res mask logits, the
    object pointer and score, and the memory the memory encoder made of them. Only the current
    frame's image features are kept.
    """

    def __init__(self, p: Pieces, frames: list[str]):
        self.p, self.frames = p, frames
        self.cfg = p.config["tracker"]
        self.c = p.consts
        self.size, self.grid = p.config["image_size"], p.config["grid"]
        self.prompts = Prompts(p)
        self.cond: dict[int, dict] = {}
        self.non_cond: dict[int, dict] = {}
        self.feats_frame, self.feats = None, None
        self.rgb = None

    def frame(self, f: int) -> dict:
        if self.feats_frame != f:
            self.rgb, pixels = load_image(self.frames[f], self.size)
            self.feats, self.feats_frame = features(self.p, pixels), f
        return self.feats

    # -- one frame for one object --

    def high_res(self, low: np.ndarray) -> np.ndarray:
        return resize(low, (self.size, self.size))

    def pointer(self, r: dict) -> np.ndarray:
        return r["pointer"] if r["score"] > 0 else self.c["no_object_pointer"]

    def from_mask(self, feats: dict, mask_1008: np.ndarray) -> dict:
        """`_use_mask_as_output`: the mask given is the output, and the decoder, prompted with
        it, only makes the object pointer."""
        g = self.grid
        high = mask_1008 * 20.0 - 10.0
        low = resize(high, (4 * g, 4 * g), antialias=True)
        w, b = self.c["mask_downsample_weight"], float(self.c["mask_downsample_bias"][0])
        n = self.size // 4
        down = (mask_1008.reshape(n, 4, n, 4) * w[None, :, None, :]).sum((1, 3)) + b
        dense = mask_to_dense(self.p, resize(down, (4 * g, 4 * g), antialias=True).astype(np.float32))
        r = decode(self.p, feats["trk_fpn2"], dense, feats, self.prompts(), multimask=True)
        # the object is where the mask says it is, whatever the decoder's own score
        appearing = bool((mask_1008 > 0).any())
        pointer = self.pointer(r) if appearing else self.c["no_object_pointer"]
        return {"low": low, "high": high, "pointer": pointer, "score": 10.0 if appearing else -10.0}

    def multimask(self, init: bool, npts: int) -> bool:
        c = self.cfg
        return c["multimask_output_in_sam"] and (init or c["multimask_output_for_tracking"]) and c["multimask_min_pt_num"] <= npts <= c["multimask_max_pt_num"]

    def conditioned(self, obj: int, f: int, feats: dict, reverse: bool) -> np.ndarray:
        """`_prepare_memory_conditioned_features` for a frame that is not a first one."""
        c, g = self.cfg, self.grid
        cond = self.cond[obj]
        # the conditioning frames closest to this one, at most max_cond_frame_num
        if len(cond) <= c["max_cond_frame_num"]:
            selected = dict(cond)
        else:
            selected = {}
            before = [t for t in cond if t < f]
            after = [t for t in cond if t >= f]
            if before:
                selected[max(before)] = cond[max(before)]
            if after:
                selected[min(after)] = cond[min(after)]
            rest = sorted((t for t in cond if t not in selected), key=lambda t: abs(t - f))
            for t in rest[: c["max_cond_frame_num"] - len(selected)]:
                selected[t] = cond[t]
        unselected = {t: v for t, v in cond.items() if t not in selected}
        memories = [(0, out) for out in selected.values()]
        for offset in range(c["num_maskmem"] - 1, 0, -1):
            prev = f + offset if reverse else f - offset
            memories.append((offset, self.non_cond[obj].get(prev, unselected.get(prev))))
        mem, mem_pos = [], []
        for offset, out in memories:
            if out is None:
                continue
            mem.append(out["memory"])
            # an offset of 0, a conditioning frame's, takes the last temporal encoding
            mem_pos.append(self.c["memory_pos"] + self.c["memory_temporal"][offset - 1])

        # object pointers: the conditioning frames' up to this one, then the frames before it
        sign = -1 if reverse else 1
        max_ptr = min(len(self.frames), c["max_object_pointers"])
        offsets, pointers = [], []
        for t, out in cond.items():
            if (t >= f) if reverse else (t <= f):
                offsets.append((f - t) * sign)
                pointers.append(out["pointer"])
        for d in range(1, max_ptr):
            ref = f + d if reverse else f - d
            if ref < 0 or ref >= len(self.frames):
                break
            out = self.non_cond[obj].get(ref)
            if out is not None:
                offsets.append(d)
                pointers.append(out["pointer"])
        dim = len(self.c["no_object_pointer"])
        mem_dim = c["mem_dim"]
        splits = dim // mem_dim
        diffs = np.asarray(offsets, np.float32) / max(max_ptr - 1, 1)
        half = dim // 2
        dim_t = 10000 ** (2 * (np.arange(half) // 2) / half)
        pe = diffs[:, None] / dim_t
        pe = np.concatenate([np.sin(pe), np.cos(pe)], -1) @ self.c["pointer_pos_weight"].T + self.c["pointer_pos_bias"]
        ptr = np.stack(pointers).reshape(-1, splits, mem_dim).reshape(-1, mem_dim)
        ptr_pos = np.repeat(pe, splits, axis=0)

        current = feats["trk_fpn2"].reshape(dim, -1).T
        out = self.p.run(
            "memory_attention",
            current=current,
            memory=np.concatenate(mem),
            memory_pos=np.concatenate(mem_pos),
            pointers=ptr,
            pointers_pos=ptr_pos,
        )["conditioned"]
        return out.T.reshape(dim, g, g)

    def infer(self, obj: int, f: int, inputs, reverse: bool) -> dict:
        feats = self.frame(f)
        # a frame prompted before the object was tracked through it starts it afresh
        init = inputs is not None and f not in self.tracked[obj]
        if inputs is not None and inputs[0] == "mask":
            return self.from_mask(feats, inputs[1])
        if init:
            pix = feats["trk_fpn2"] + self.c["no_memory_embedding"][:, None, None]
        else:
            pix = self.conditioned(obj, f, feats, reverse)
        points, labels = (inputs[1], inputs[2]) if inputs is not None else (None, None)
        npts = 0 if points is None else len(points)
        r = decode(self.p, pix, no_mask_dense(self.p), feats, self.prompts(points, labels), self.multimask(init, npts))
        # the mask kept for memories is a hard choice between object and none
        low = r["mask"] if r["score"] > 0 else np.full_like(r["mask"], NO_OBJ_SCORE)
        return {"low": low, "high": self.high_res(low), "pointer": self.pointer(r), "score": r["score"]}

    def encode(self, f: int, outs: dict, from_inputs: bool):
        """`_batch_encode_memories`: each object's memory of this frame."""
        feats = self.frame(f)
        m = 16 * self.grid
        for out in outs.values():
            high = resize(out["high"], (m, m), antialias=True)
            mask = (high > 0).astype(np.float32) if from_inputs else sigmoid(high)
            mask = mask * self.cfg["sigmoid_scale_for_mem_enc"] + self.cfg["sigmoid_bias_for_mem_enc"]
            mem = self.p.run("memory_encoder", pix=feats["trk_fpn2"], mask=mask[None].astype(np.float32))["features"]
            if out["score"] <= 0:
                mem = mem + self.c["occlusion_embedding"][:, None, None]
            out["memory"] = bf16(mem.reshape(len(mem), -1).T)
            del out["high"]

    # -- the session --

    def run(self, objects: dict, start: int, end: int, reverse: bool, on_frame):
        """`objects`: id -> {frame: inputs}. Prompted frames first, then each frame from `start`
        to `end`, in order, calling on_frame(f, {id: out})."""
        self.tracked = {o: set() for o in objects}
        for o in objects:
            self.cond.setdefault(o, {})
            self.non_cond.setdefault(o, {})
        prompted = sorted({f for inp in objects.values() for f in inp})
        for f in prompted:
            outs = {}
            for o, inp in objects.items():
                if f in inp:
                    outs[o] = self.infer(o, f, inp[f], reverse=False)
            self.encode(f, outs, from_inputs=True)
            for o, out in outs.items():
                self.cond[o][f] = out
        order = range(start, end - 1, -1) if reverse else range(start, end + 1)
        for f in order:
            outs, fresh = {}, {}
            for o in objects:
                if f in self.cond[o]:
                    outs[o] = self.cond[o][f]
                    continue
                first = min(self.cond[o]) if self.cond[o] else None
                if first is None or (f < first if not reverse else f > first):
                    continue  # not prompted yet on this pass
                fresh[o] = self.infer(o, f, None, reverse)
            if fresh:
                self.encode(f, fresh, from_inputs=False)
            for o, out in fresh.items():
                self.non_cond[o][f] = out
                self.tracked[o].add(f)
                outs[o] = out
            on_frame(f, outs)


def frames_of(spec, stride: int = 1) -> list[str]:
    """A list of frame images, a folder of them in the order of their names, or a video file,
    whose frames are written out under /tmp first."""
    if isinstance(spec, list):
        return spec[::stride]
    d = Path(spec)
    if d.is_dir():
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        return sorted(str(x) for x in d.iterdir() if x.suffix.lower() in exts)[::stride]
    if d.is_file():
        import av

        out = Path("/tmp/sam3_frames") / d.stem
        out.mkdir(parents=True, exist_ok=True)
        paths = []
        with av.open(str(d)) as container:
            for i, frame in enumerate(container.decode(video=0)):
                if i % stride == 0:
                    path = out / f"{i:05d}.jpg"
                    frame.to_image().save(path, quality=95)
                    paths.append(str(path))
        log(f"{d.name}: {len(paths)} frames")
        return paths
    raise BadRequest(f"`frames` is not a folder, a video file or a list of images: {spec}")


def track(p: Pieces, req: dict) -> dict:
    if not req.get("frames") or not req.get("out"):
        raise BadRequest("track needs `frames` and `out`")
    frames = frames_of(req["frames"], int(req.get("stride", 1)))
    if not frames:
        raise BadRequest("no frames")
    out = Path(req["out"])
    size = p.config["image_size"]
    with Image.open(frames[0]) as im:
        w, h = im.size
    video = Video(p, frames)

    # each object's inputs, by frame: ("points", points, labels) or ("mask", mask at IMAGE)
    objects: dict = {}
    for i, obj in enumerate(req.get("objects") or []):
        oid, f = obj.get("id", i + 1), int(obj.get("frame", 0))
        if obj.get("mask"):
            objects.setdefault(oid, {})[f] = ("mask", read_mask(obj["mask"], (size, size)))
            continue
        pts = scaled(obj["points"], w, h, size) if obj.get("points") else np.zeros((0, 2), np.float32)
        lbl = np.asarray(obj.get("point_labels", [1] * len(pts)), int)
        if obj.get("box"):
            # the video processor puts a box in as its two corners, labeled 2 and 3, before points
            pts = np.concatenate([scaled(obj["box"], w, h, size), pts])
            lbl = np.concatenate([[2, 3], lbl])
        if not len(pts):
            raise BadRequest(f"object {oid} has no `points`, `box` or `mask`")
        objects.setdefault(oid, {})[f] = ("points", pts, lbl)
    if req.get("text"):
        # the detector on the prompt frame; each instance it finds is an object, prompted with its
        # mask, as SAM3's video model adds new detections
        f = int(req.get("text_frame", 0))
        feats = video.frame(f)
        d = detect_prompt(p, feats, Text(p), req["text"])
        threshold = float(req.get("threshold", p.config["detector"]["score_threshold"]))
        next_id = max([o for o in objects if isinstance(o, int)] + [0]) + 1
        for q in np.nonzero(d["scores"] > threshold)[0][np.argsort(-d["scores"][d["scores"] > threshold])]:
            m = (resize(sigmoid(d["masks"][q]), (size, size)) > 0.5).astype(np.float32)
            objects[next_id] = {f: ("mask", m)}
            next_id += 1
        p.drop("text", "detector", "geometry")
        log(f"{req['text']!r}: {len(objects)} objects on frame {f}")
    if not objects:
        raise BadRequest("track needs `objects` or a `text` that finds some")

    first = min(f for inp in objects.values() for f in inp)
    reverse = bool(req.get("reverse", False))
    n = int(req.get("max_frames", len(frames)))
    start = int(req.get("start", first))
    end = max(start - n + 1, 0) if reverse else min(start + n - 1, len(frames) - 1)
    ids = list(objects)
    summary = {o: {"frames": 0, "first": None, "last": None} for o in ids}
    per_frame = []

    def on_frame(f, outs):
        masks, rects, entry = [], [], {"frame": f, "image": frames[f], "objects": {}}
        for o in ids:
            if o not in outs:
                continue
            mask = upscale(outs[o]["low"], (h, w)) > 0
            b = box_of(mask)
            if mask.any():
                save_mask(mask, out / "masks" / f"obj{o}" / f"{f:05d}.png")
                s = summary[o]
                s["frames"] += 1
                s["first"] = f if s["first"] is None else min(s["first"], f)
                s["last"] = f if s["last"] is None else max(s["last"], f)
            entry["objects"][str(o)] = {"score": round(float(outs[o]["score"]), 3), "area": int(mask.sum()), "box": b}
            masks.append(mask)
            rects.append(b)
        (out / "overlay").mkdir(parents=True, exist_ok=True)
        overlay(video.rgb, masks, rects, out / "overlay" / f"{f:05d}.png")
        per_frame.append(entry)
        log(f"frame {f}: " + ", ".join(f"obj{o} {v['area']}px" for o, v in entry["objects"].items()))

    out.mkdir(parents=True, exist_ok=True)
    video.run(objects, start, end, reverse, on_frame)
    (out / "frames.json").write_text(json.dumps(per_frame, indent=1) + "\n")
    return {
        "task": "track",
        "frames": len(frames),
        "tracked": [per_frame[0]["frame"], per_frame[-1]["frame"]] if per_frame else [],
        "objects": {str(o): v for o, v in summary.items()},
        "per_frame": str(out / "frames.json"),
        "masks": str(out / "masks"),
        "overlays": str(out / "overlay"),
    }


TASKS = {"detect": detect, "segment": segment, "track": track}


def main():
    if len(sys.argv) != 3 or sys.argv[1] not in MODES:
        log(f"usage: run_sam3.py {{{'|'.join(MODES)}}} REQUEST")
        sys.exit(4)
    mode = sys.argv[1]
    try:
        req = json.loads(sys.argv[2])
        task = TASKS.get(req.get("task"))
        if task is None:
            raise BadRequest(f"`task` must be one of {list(TASKS)}")
        device()
        t = time.time()
        result = task(Pieces(mode), req)
    except (BadRequest, json.JSONDecodeError, FileNotFoundError) as e:
        print(json.dumps({"error": str(e) or type(e).__name__}))
        sys.exit(4)
    result["mode"] = mode
    result["seconds"] = round(time.time() - t, 1)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
