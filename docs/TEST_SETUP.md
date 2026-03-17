# Test Setup

This project supports configurable model/trace locations for integration and numeric trace tests.

## Configuration Priority

Path resolution order:

1. Environment variables
2. CMake definitions
3. Built-in defaults

## Environment Variables

- `VOXCPM_MODEL_PATH`: GGUF model path
- `VOXCPM_TRACE_DIR`: trace directory path
- `VOXCPM_TEST_THREADS`: CPU test thread count (already supported)

Example:

```bash
export VOXCPM_MODEL_PATH=/data/models/voxcpm1.5.gguf
export VOXCPM_TRACE_DIR=/data/voxcpm-traces
ctest --output-on-failure
```

## CMake Defaults

You can override compile-time defaults:

```bash
cmake -B build \
  -DVOXCPM_DEFAULT_MODEL_PATH=/data/models/voxcpm1.5.gguf \
  -DVOXCPM_DEFAULT_TRACE_DIR=/data/voxcpm-traces
cmake --build build
```

## Behavior When Assets Are Missing

Integration/trace tests are designed to skip gracefully when model or trace files are absent.
This allows open-source contributors to run the test suite without private or large test assets.

## Trace Files

`trace_*.jsonl` files are used for numeric parity validation between PyTorch and ggml implementations.
Keep trace generation reproducible (fixed model version, fixed export scripts, fixed seeds) when updating fixtures.
