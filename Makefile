.PHONY: all cuda rust python test tauri clean

all: cuda rust python

cuda:
	cmake -S . -B build
	cmake --build build -j

rust:
	@if [ -f Cargo.toml ]; then cargo build; else echo "No Cargo.toml found, skipping Rust build."; fi

tauri:
	cd tauri-src/src-tauri && cargo build

python:
	uv sync --extra dev --extra data

test: cuda python
	uv run pytest -q
	@if [ -f build/CTestTestfile.cmake ]; then ctest --test-dir build --output-on-failure; else echo "No CTest file found, skipping C++ tests."; fi
	@if [ -f Cargo.toml ]; then cargo test; fi

clean:
	rm -rf build .venv
	@if [ -f Cargo.toml ]; then cargo clean; else rm -rf target; fi
