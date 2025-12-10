PYTHON := python

.PHONY: all setup install-deps download-data clean

all: setup

setup: install-deps download-data

install-deps:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt

download-data:
	$(PYTHON) Code/data-extraction/download_gdrive_folder.py

clean:
	rm -rf __pycache__ .pytest_cache