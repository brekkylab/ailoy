---
name: yoloe
description: Detect objects in images with YOLOE, a machine learning model for object detection, and get a box, a class and a confidence for each. Use it to find, count or locate things in images, to draw boxes on them, or to check that the detector runs on this machine.
---

# YOLOE

YOLOE is an open-vocabulary object detector. It is given what to look for as class names in
plain text, and finds them in an image as boxes, each with a class and a confidence. The
classes were set when the model was converted, and this model finds those and no others. They
are listed in `/models/yoloe.json` under `names`, and in every answer under `classes`. When
the user asks for something that is not one of them, say so rather than answering with the
nearest class.

## Running it

Run `run_yoloe.py` from this directory with the mode and the request as its two arguments.

```sh
python3 run_yoloe.py fp32 '{"images": ["/context/street.jpg"], "classes": ["person"], "annotate": "/artifacts/street"}'
```

The mode is `fp32`, `fp16` or `bf16`. fp16 and bf16 store the weights in half the memory and
are less exact, and bf16 the least: its boxes can move by a few pixels. The model is read
from `/models`, or from the directory `YOLOE_MODELS` names.

Each run loads the model again, which takes a second or two, and an image takes a fraction of
a second. So put every image of one task in a single run.

## The request

The request is a JSON object.

* `images` is a list of paths to image files, in any format OpenCV reads. Each is scaled to
  fit 640x640, as YOLOE takes it, keeping its aspect ratio.
* `classes` is optional, the classes to report, from those the model finds. The others are
  left out of the answer and the drawn images. A class the model does not find is listed
  under `unknown_classes` in the answer.
* `conf` is optional, the confidence below which a detection is dropped. It is 0.25 by
  default. Lower it to find more, at the cost of more false detections.
* `annotate` is optional, a directory to write each image to with its boxes drawn and
  labelled. Put it under `/artifacts` when the user is to see them.

## The answer

The run prints one JSON object to stdout. It has the `mode`, the `classes` the model finds,
the `conf` threshold, and a list of `images` in the order they were given. Each entry has the
`path`, the `size` of the image as width and height, the `ms` the model took, and its
`detections`. With `annotate`, it also has the path of the `annotated` image.

Each detection has its `class`, its `confidence` between 0 and 1, and its `box` as
`[x0, y0, x1, y1]` in the image's pixels, from the top left. Detections are sorted by
confidence, highest first. Two boxes of one class on the same object are rare but possible,
most of all in bf16.

An image that could not be taken, such as a missing file, comes back with an `error` field
instead, and the others are still looked in.

What ncnn logs about the device goes to stderr, with the load time. Its `fp16-p/s/u/a` and
`bf16-p/s` lines say whether fp16 or bf16 were really used. Where the device has no such type,
the run is fp32.

## Checking the model

To check that the model runs, and how close a mode comes to the PyTorch model, run it with
`check` first.

```sh
python3 run_yoloe.py check bf16
```

It looks in the image the reference was taken on, at `/models/image.jpg`, and matches each
detection to what Ultralytics found there with the PyTorch model. Each is printed with the
reference's confidence and its overlap with the reference's box. The last line says `OK` or
that the detections differ from the reference.

## Exit codes

The run exits 2 if the wheel has no Vulkan, and 3 if there is no device to run on. It exits
5 if an output is non-finite or, when checking, the detections differ from the reference.
Otherwise it exits 0, even when some images came back with an `error`.
