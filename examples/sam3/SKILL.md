---
name: sam3
description: Segment images and videos with SAM3, a machine learning model for segmentation. Use it to find and mask every instance of something named in text, to mask one object picked out by points or a box, or to follow objects through the frames of a video.
---

# SAM3

SAM3 finds and segments objects. It does three tasks.

* `detect` finds every instance of what a short text names in an image, such as "cat" or
  "red car", with a score, a box and a mask each. It can also find more of what example boxes
  hold.
* `segment` masks one object in an image for each set of points, box or mask given.
* `track` follows objects through the frames of a video. Each object starts from points, a
  box or a mask on one frame, or from what a text finds on one frame.

## Running it

Run `run_sam3.py` from this directory with the mode and the request as its two arguments.

```sh
python3 run_sam3.py fp32 '{"task": "detect", "image": "/context/street.jpg", "text": "car", "out": "/artifacts/street-cars"}'
```

The mode is `fp32`, or `bf16` for a run that is less exact. The models are read from
`/models`, or from the directory `SAM3_MODELS` names.

Every run loads the image model again and runs it once for each image or frame, which takes
several seconds each. So put every text for one image in a single request, and track every
object of one video in a single request.

The result is printed to stdout as one JSON object. The masks are written as PNG files under
the request's `out` directory, white where the object is, at the image's own size. An
`overlay.png` shows them colored over the image, with their boxes. A request it cannot take
comes back as `{"error": ...}` and exits 4. What ncnn logs about the device goes to stderr.

Coordinates are in pixels of the original image, x to the right and y down. A box is
`[x1, y1, x2, y2]`, its top-left and bottom-right corners, and a point is `[x, y]`.

## detect

```json
{"task": "detect", "image": "/context/kitchen.jpg", "text": ["mug", "spoon"], "out": "/artifacts/kitchen"}
```

* `text` is a short noun phrase, or a list of them. Each is run on its own.
* `boxes` are example boxes of the thing to find, and `box_labels` say for each whether it is
  an example (1) or a counterexample (0). All are examples if left out. With boxes and no text,
  SAM3 finds more of what the boxes hold.
* `threshold` is the score an instance needs to be kept, 0.5 by default.

The result lists `instances`, best first. Each has the `prompt` that found it, its `score`
between 0 and 1, its `box`, its `area` in pixels and the path of its `mask`. No instance
means nothing scored above the threshold. Lower it only when the user asks for weaker
matches.

## segment

```json
{"task": "segment", "image": "/context/desk.jpg", "objects": [{"points": [[410, 220]]}, {"box": [30, 40, 200, 380]}], "out": "/artifacts/desk"}
```

Each entry of `objects` is one object, and gets one mask. It has any of the following.

* `points`, with `point_labels` saying for each whether it is on the object (1) or off it (0).
  All are on it if left out.
* `box`, around the object.
* `mask`, the path of a rough mask image to refine.
* `multimask`, whether to choose among three masks. It is on for a single point, which could
  mean the part or the whole, and off otherwise.

Each object's result has its `iou`, the model's own estimate of how good the mask is, and its
`box`, `area` and `mask`. A low `iou` means a vague prompt. More points, or a box, pin it down.

## track

```json
{"task": "track", "frames": "/context/clip", "objects": [{"id": 1, "frame": 0, "box": [120, 60, 300, 400]}], "out": "/artifacts/clip-tracks"}
```

* `frames` is a video file, a folder of frame images taken in the order of their names, or a
  list of paths. `stride` keeps every so many frames, 1 by default: tracking takes several
  seconds a frame, so a long video goes faster with a stride. Frame numbers count the frames
  kept.
* `objects` are the objects to follow. Each has an `id`, the `frame` it is prompted on, 0 by
  default, and `points` with `point_labels`, a `box` or a `mask`, as in `segment`.
* `text` finds its instances on `text_frame`, 0 by default, with `threshold`. Each instance it
  finds is an object to follow, numbered after the given ones.
* `reverse` follows the objects back from the first prompted frame instead of forward.
* `start` is the frame to start from, the first prompted one by default, and `max_frames` how
  many frames to go.

The masks are written to `masks/objID/FRAME.png` for each frame an object is on, and an overlay
of each frame to `overlay/FRAME.png`. For each object, the result gives the number of `frames`
it was found on, and the `first` and `last` of them. `frames.json` lists each frame's objects
with their `score`, `area` and `box`. A negative score means the object is not on that frame,
such as when it is hidden or has left.

## Exit codes

The run exits 2 if the wheel has no Vulkan, and 3 if there is no device to run on. It exits 4 on
a request it cannot take, and 5 if the model gives a non-finite output.
