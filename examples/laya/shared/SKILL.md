---
name: laya
description: Answer typed questions about a text with Laya, a machine learning model for decisions. Use it to classify, score or answer yes or no about a message, an email or a ticket, with calibrated probabilities.
---

# Laya

Laya is a machine learning model for decisions. Given a text, such as a message, an email or a
ticket, it answers typed questions about it with calibrated probabilities. It picks one of
several options, rates on an ordered scale, or answers yes or no, and it generates no text.
You put what is to be decided to it as questions, and write the reply yourself from its
answers.

## Running it

Run `run_laya.py` from this directory with the mode and the request as its two arguments.

```sh
python3 run_laya.py bf16 '{"state": "...", "questions": {...}}'
```

The mode is `bf16`, which is faster and good enough for almost every request, so use it. Use
`fp32`, which is exact but slower, only when a bf16 run's answers are clearly wrong, or when you
are asked for exact output. The model is read from `/models`, or from the directory
`LAYA_MODELS` names. Each run loads the model again, which
takes about a second, so put every question about one text in a single run.

The answers are printed to stdout as one JSON object, keyed by the ids of the questions. What
ncnn logs about the device goes to stderr.

## The request

The request is a JSON object with two fields.

The `state` is the text the questions are about, passed verbatim. Do not summarize it or
decide in its place. The state and each question share a window of 512 tokens, so a state
longer than about 300 tokens is cut off.

The `questions` field is an object from an id of your choosing, such as `department` or
`urgency`, to a question. Each question has a `type`, its `instructions` in one sentence, and
`criteria` that depend on the type.

A `choice` picks one of the criteria and gives a probability for each. Its criteria are a
list of labels, or an object from each label to what it covers.

```json
{"type": "choice", "instructions": "Which department should handle this request?",
 "criteria": {"billing": "invoices, payments, refunds", "technical": "bugs, outages",
              "sales": "pricing, new contracts", "other": "everything else"}}
```

A `score` picks a level on an ordered scale and is answered as its expected value. Its
criteria are the levels as a list of descriptions, from lowest to highest.

```json
{"type": "score", "instructions": "How urgent is this request?",
 "criteria": ["not urgent", "soon", "critical deadline or blocking issue"]}
```

A `noul` is a yes or no question and is answered as the probability of yes. Its criteria are
optional, an object that says what `true` and `false` mean.

```json
{"type": "noul", "instructions": "Does the user threaten to cancel or leave?"}
```

## The answers

A choice answer has the `choice` it picked and the `probabilities` of every label. A score
answer has the `score`, the `probabilities` of every level and a `legend` of what each level
means. A noul answer has `noul`, the probability of yes.

Every answer also has a `confidence` between 0 and 1 that says how peaked the distribution
is, and an `action.act_probability` that says whether to act on it rather than escalate.
Where the confidence is low or the probabilities are close, report that Laya was unsure
rather than rounding it to a verdict.

A question Laya cannot take, such as a choice with a single option, comes back as an object
with an `error` field, and the other questions are still answered.

If the run exits 2 or 3, there is no Vulkan device to run on.
