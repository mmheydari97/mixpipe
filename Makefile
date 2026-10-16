.PHONY: install install-chords stems analyze mix mix-llm chords chords-ui split split-ui test clean

install:
	pip install -r requirements.txt

install-chords:
	pip install -r requirements.txt -r requirements-chords.txt

stems:
	PYTHONPATH=src python scripts/make_test_stems.py

analyze:
	PYTHONPATH=src python -m mixpipe.cli analyze --stems stems/

mix:
	PYTHONPATH=src python -m mixpipe.cli mix --stems stems/ --out out/mix_master.wav

mix-llm:
	PYTHONPATH=src python -m mixpipe.cli mix --stems stems/ --advisor llm \
		--notes "$(NOTES)" --out out/mix_master_llm.wav

chords:
	PYTHONPATH=src python -m mixpipe.cli chords "$(FILE)" --lang $(or $(LANG_CODE),auto)

chords-ui:
	PYTHONPATH=src python -m mixpipe.cli chords-ui

split:
	PYTHONPATH=src python -m mixpipe.cli split "$(FILE)"

split-ui:
	PYTHONPATH=src python -m mixpipe.cli split-ui

test:
	PYTHONPATH=src python -m pytest tests/ -q

clean:
	rm -rf out/* stems/*.wav __pycache__ src/mixpipe/__pycache__ .pytest_cache
	touch out/.gitkeep
