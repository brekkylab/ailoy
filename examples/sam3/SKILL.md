---
name: sam3
description: Encode images and text prompts with SAM3's image and language encoders, a machine learning model for segmentation, and save the features its decoder takes. Use it to turn images and prompts into SAM3 features, or to check that the encoders run on this machine.
---

# SAM3

SAM3 segments what a text prompt names in an image, such as "cat" or "a red car". Only its
two encoders are here. The image encoder turns an image into features at three scales, and
the language encoder turns a prompt into text features. The decoder that would make boxes
and masks from them is not, so this skill gives no masks, boxes or scores. What it gives is
the features, saved to a file, and a summary of them.

## Running it

Run `run_encoders.py` from this directory with the mode and the request as its two arguments.

```sh
python3 run_encoders.py fp32 '{"images": ["/context/street.jpg"], "texts": ["a red car"], "out": "/artifacts/street.npz"}'
```

The mode is `fp32`, or `bf16` for a run that stores the weights in half the memory and is
less exact. The models are read from `/models`, or from the directory `SAM3_MODELS` names.

Each run loads the models again. The image encoder takes a second to load and tens of
seconds an image, and the language encoder a fraction of a second a prompt. So put every
image and prompt of one task in a single run.

## The request

The request is a JSON object with three fields.

* `images` is a list of paths to image files, in any format Pillow reads. Each is converted
  to RGB and stretched to 1008x1008, as SAM3 does, whatever its aspect ratio.
* `texts` is a list of prompts. Each is a short noun phrase, as SAM3 takes. A prompt is cut
  at 32 tokens, start and end included.
* `out` is the path of the `.npz` file to write. Put it under `/artifacts` when the user is
  to get it, and name it after what was encoded.

Either `images` or `texts` can be left out, but not both.

## The answer

The run prints one JSON object to stdout. It has the `mode`, the `out` path, and a list each
of `images` and `texts` in the order they were given. Each entry has its `key` in the file,
such as `image0` or `text1`. It has the `seconds` it took and the `shape`, `mean` and `std`
of each output, and `finite`, which is false when an output has NaN or infinite values. An
image entry also has the `size` it had before it was stretched, as width and height. A text
entry also has the number of `tokens` it came to.

An input that could not be taken, such as a missing file, comes back with an `error` field
instead, and the others are still encoded.

The file holds these arrays, named after the entry's key.

* `imageN.backbone_fpn_0`, `imageN.backbone_fpn_1` and `imageN.backbone_fpn_2` are the image
  features, 256 channels at 288x288, 144x144 and 72x72.
* `textN.text_memory` is the text features, 32 tokens by 256. `textN.text_embeds` is the text
  embeddings before the projection, 32 by 1024. `textN.text_attention_mask` is true at the
  padding tokens.
* `vision_pos_enc_0`, `vision_pos_enc_1` and `vision_pos_enc_2` are the position encodings of
  the three scales. They are the same for every image, and are there when any image is.

What ncnn logs about the device goes to stderr, with the load times. Its `bf16-p/s` line says
whether bf16 storage was really used. Where it says `bf16-p/s=1/0` or `0/0`, a `bf16` run is
fp32.

## Checking the encoders

To check that an encoder runs, and how close bf16 comes to fp32, run it with the model name
first. Run fp32 first, and then bf16 against it, in separate runs.

```sh
python3 run_encoders.py sam3_language_encoder fp32
python3 run_encoders.py sam3_language_encoder bf16 fp32
```

The model is `sam3_language_encoder` or `sam3_image_encoder`. The input is fixed: an empty
prompt, or a random image. Each output is printed with its cosine similarity to the fp32
run's, and below 0.99 it counts as wrong. The last line says `OK` or that an output was
non-finite or mismatched.

## Exit codes

The run exits 2 if the wheel has no Vulkan, and 3 if there is no device to run on. It exits
5 if an output is non-finite or, when checking, too far from the reference's. Otherwise it
exits 0, even when some inputs came back with an `error`.
