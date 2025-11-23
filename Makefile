.PHONY: setup help
.DEFAULT_GOAL := help

setup:  ## Install development dependencies
	@# check if uv is installed
	@uv --version >/dev/null 2>&1 || (echo "uv is not installed, please install it" && exit 1)

	@# install dependencies
	uv sync --group dev
	uv run pre-commit install

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'
