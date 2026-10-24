.PHONY: all cuda rust python test clean

all: cuda rust python

cuda:
	cmake -S . -B build
	cmake --build build -j

rust:
	@if [ -f Cargo.toml ]; then cargo build; else echo "No Cargo.toml found, skipping Rust build."; fi

python:
	uv sync

test: python
	uv run python -m ml.kan.demo
	@if [ -f Cargo.toml ]; then cargo test; else echo "No Cargo.toml found, skipping Rust tests."; fi

clean:
	rm -rf build .venv
	@if [ -f Cargo.toml ]; then cargo clean; else rm -rf target; fi
