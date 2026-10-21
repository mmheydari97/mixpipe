.PHONY: install install-chords stems demo-session verify-mix analyze mix mix-llm chat mix-ui mix-studio install-acestep acestep chords chords-ui split split-ui test clean

# Use the project venv when there is one, so targets work without activating it.
PY := $(if $(wildcard .venv/bin/python),.venv/bin/python,python3)
# venvs made by uv have no pip; install with uv when it's available.
PIP := $(if $(shell command -v uv 2>/dev/null),uv pip install --python $(PY),$(PY) -m pip install)
RUN := PYTHONPATH=src $(PY)
# extra CLI flags for the song tools, e.g. make split-ui ARGS="--port 8767 --no-browser"
ARGS ?=

install:
	$(PIP) -r requirements.txt

chat:
	$(RUN) -m mixpipe.cli chat --stems $(or $(STEMS),stems/) $(ARGS)

mix-ui:
	$(RUN) -m mixpipe.cli mix-ui $(ARGS)

# AI edits: a local ACE-Step 1.5 server, in its own checkout and venv (about 11 GB with the model)
ACESTEP_DIR ?= ../ACE-Step-1.5
install-acestep:
	test -d $(ACESTEP_DIR) || git clone --depth 1 https://github.com/ace-step/ACE-Step-1.5.git $(ACESTEP_DIR)
	cd $(ACESTEP_DIR) && uv sync

# the mixer page and the ACE-Step server together; Ctrl+C stops both
mix-studio:
	ACESTEP_DIR=$(ACESTEP_DIR) scripts/mix_studio.sh $(ARGS)

acestep:
	cd $(ACESTEP_DIR) && ACESTEP_LM_BACKEND=mlx ACESTEP_INIT_LLM=false TOKENIZERS_PARALLELISM=false \
		uv run acestep-api --host 127.0.0.1 --port 8001

install-chords:
	$(PIP) -r requirements.txt -r requirements-chords.txt

stems:
	$(RUN) scripts/make_test_stems.py

# eight synthetic stems with deliberate flaws (hum, clicks, sibilance, a late bass)
demo-session:
	$(RUN) scripts/make_demo_session.py --out stems_demo/

# run every mixer feature on a stems folder and check the result, e.g.
# make verify-mix STEMS=out/splitter/<id>/stems ARGS="--start 60 --duration 45"
verify-mix:
	$(RUN) scripts/verify_mix.py $(or $(STEMS),stems_demo/) $(ARGS)

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
