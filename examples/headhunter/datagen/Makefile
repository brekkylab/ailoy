# The generation pipeline. Order matters: structure first, prose after.

DATA := ../data
PY := .venv/bin/python

.PHONY: gen narrate validate clean

# Deterministic, so it can be rerun at any time. truth.py consumes gen.py's assemble(),
# which is why the answer key comes out of the same target.
gen:
	$(PY) -m headhunter.gen
	$(PY) -m headhunter.truth

# Calls an LLM and the result is committed, so this runs once.
narrate:
	$(PY) -m headhunter.narrate

validate:
	$(PY) -m headhunter.validate

clean:
	rm -f $(DATA)/candidates.json $(DATA)/ground_truth.json
