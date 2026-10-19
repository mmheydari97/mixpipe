.PHONY: install install-chords stems analyze mix mix-llm chords chords-ui split split-ui test clean

# Use the project venv when there is one, so targets work without activating it.
PY := $(if $(wildcard .venv/bin/python),.venv/bin/python,python3)
# venvs made by uv have no pip; install with uv when it's available.
PIP := $(if $(shell command -v uv 2>/dev/null),uv pip install --python $(PY),$(PY) -m pip install)
RUN := PYTHONPATH=src $(PY)
# extra CLI flags for the song tools, e.g. make split-ui ARGS="--port 8767 --no-browser"
ARGS ?=

install:
	$(PIP) -r requirements.txt

install-chords:
	$(PIP) -r requirements.txt -r requirements-chords.txt

stems:
	$(RUN) scripts/make_test_stems.py

analyze:
	$(RUN) -m mixpipe.cli analyze --stems stems/

mix:
	$(RUN) -m mixpipe.cli mix --stems stems/ --out out/mix_master.wav

mix-llm:
	$(RUN) -m mixpipe.cli mix --stems stems/ --advisor llm \
		--notes "$(NOTES)" --out out/mix_master_llm.wav

chords:
	@test -n "$(FILE)" || { echo 'usage: make chords FILE="song.mp3 or https://..." [LANG_CODE=en]'; exit 2; }
	$(RUN) -m mixpipe.cli chords "$(FILE)" --lang $(or $(LANG_CODE),auto) $(ARGS)

chords-ui:
	$(RUN) -m mixpipe.cli chords-ui $(ARGS)

split:
	@test -n "$(FILE)" || { echo 'usage: make split FILE="song.mp3 or https://..."'; exit 2; }
	$(RUN) -m mixpipe.cli split "$(FILE)" $(ARGS)

split-ui:
	$(RUN) -m mixpipe.cli split-ui $(ARGS)

test:
	$(RUN) -m pytest tests/ -q

# pipeline outputs only: the chord finder and splitter song libraries are kept
clean:
	find out -mindepth 1 -maxdepth 1 ! -name chords ! -name splitter ! -name .gitkeep -exec rm -rf {} +
	rm -rf stems/*.wav __pycache__ src/mixpipe/__pycache__ .pytest_cache
	touch out/.gitkeep
