---
name: tts
description: Speak a text aloud with Qwen3-TTS, a machine learning model for text to speech, in a voice described in words, and get a WAV file. Use it to read a text out, to make narration or a voice-over, or to check that the model runs on this machine.
---

# Qwen3-TTS

Qwen3-TTS turns text into speech. This is its VoiceDesign model: there are no named voices to
pick from, and the voice is what an instruction describes in plain words, such as "a calm woman
in her thirties, speaking slowly and warmly, like a radio host". The instruction can say who is
speaking -- sex, age, timbre, pitch -- and how -- pace, mood, emotion, emphasis. It can be in
any language the model speaks, and need not be the text's.

It speaks Chinese, English, Japanese, Korean, German, French, Russian, Portuguese, Spanish and
Italian.

## Running it

Run `run_tts.py` from this directory with the mode and the request as its two arguments.

```sh
python3 run_tts.py fp16 '{"text_file": "/context/text.txt", "instruct_file": "/context/instruct.txt", "out": "/artifacts/speech.wav"}'
```

The mode is `fp16`, which is faster and sounds as good, so use it. Use `fp32`, which is slower
and a little more exact, only when an fp16 run's speech is clearly wrong, or when you are asked
for exact output. The models are read from `/models`, or from the directory `TTS_MODELS`
names.

Each run loads the models again, which takes several seconds, and then makes the speech a
frame at a time, 12.5 frames a second of speech, and decodes it: about five seconds for each
second of speech in all. It prints its progress to stderr as it goes. So speak a whole text in
one run rather than a sentence at a time, unless it is too long for one: see below. Give a
command that runs it a timeout of at least ten times the speech it is to make.

## The request

The request is a JSON object.

* `text` is the text to speak, or `text_file` a file to read it from.
* `instruct` is the instruction, or `instruct_file` a file to read it from. It is optional:
  without one, the model picks a voice itself.
* `out` is the path to write the WAV file to. Put it under `/artifacts` when the user is to
  have it.
* `language` is optional, `auto` by default, which lets the model tell. Naming it, such as
  `korean` or `english`, helps with a short text or one that mixes languages.
* `seed` is optional, 0 by default. The speech is sampled, so another seed says the same text
  a little differently, and the same seed says it the same way again.

The text is spoken as written. Numbers, abbreviations and symbols are read as the model
guesses, so spell out what has to be read one way.

A run speaks at most about ten minutes. Over a minute or two, the voice can drift and the
model can skip or repeat, so split a long text at paragraph or sentence ends into runs, each
with the same instruction and seed, write each to its own file, and say so.

## The answer

The run prints one JSON object to stdout. It has the `mode`, the path the speech was written
to as `out`, how many `seconds` of speech it is and in how many `frames`, whether the model
`ended` the speech itself, the `language` and the `seed`. `generate_s` and `decode_s` are the
seconds the two stages took, and `realtime_factor` the seconds a second of speech took.

The WAV file is 16-bit mono PCM at 24 kHz.

`ended` false means the model did not stop by the limit, which happens when it goes astray:
the speech is there, but check the text and try another seed.

A request it cannot take, such as one without a text or with an unknown language, comes back
as `{"error": ...}` and exits 4.

What ncnn logs about the device goes to stderr, with the load times. Its `fp16-p/s/u/a` line
says whether fp16 was really used. Where the device has no such type, the run is fp32.

## Checking the model

To check that the model runs, and how close a mode comes to the PyTorch model, run it with
`check` first.

```sh
python3 run_tts.py check fp16
```

It runs the model on the speech the PyTorch model made of an example, and compares the
probability each gave every one of its codes, and the waveform. The last line says `OK` or
that it differs from the reference.

## Exit codes

The run exits 2 if the wheel has no Vulkan, and 3 if there is no device to run on. It exits 4
on a request it cannot take, and 5 if an output is non-finite or, when checking, the model
differs from the reference. Otherwise it exits 0.
