# CS3210 Assignment 2 – DNA Virus String Matching (CUDA)

Authors: Chang Jia Jun / Hieu Trung

Course: CS3210 – High Performance Computing

Repository: https://github.com/jiajunchang2002g/CS3210-Assignment2

## Table of contents

1. Project overview
2. Motivation
3. Features
4. Repository layout
5. Prerequisites
6. Build & run
7. Usage examples
8. Design notes
9. Benchmarking
10. Limitations & future work
11. License

## Project overview

This repository contains a classroom assignment that implements GPU-accelerated substring (DNA signature) matching using CUDA. The goal is to match a set of viral signature sequences against large DNA sample sequences efficiently by offloading work to the GPU and comparing to a CPU baseline.

## Motivation

- DNA sequence databases are large; naive CPU search can be slow.
- GPUs can accelerate massive, data-parallel workloads such as substring searching.
- The assignment explores memory layout, GPU kernel design, and trade-offs between CPU and GPU approaches.

## Features

- Utilities to generate synthetic DNA sample sequences (`gen_sample.cc`).
- Utilities to generate signature lists (`gen_sig.cc`).
- Shared code and types in `common.cc` / `common.h` and `device_seq_t.h`.
- CUDA kernel skeleton in `kernel_skeleton.cu` for GPU matching.
- Scripts and folders for benchmarking on different hardware (`bench-a100`, `bench-h100`).
- Download helper for FASTA files (`download_fasta.sh`).

## Repository layout

Top-level files and directories (important ones):

- `gen_sample.cc`         — sample DNA generator (synthetic datasets)
- `gen_sig.cc`            — signature (viral substring) generator
- `common.cc`, `common.h` — shared helpers and CPU reference implementation
- `device_seq_t.h`        — device/host sequence types and helpers
- `kernel_skeleton.cu`    — CUDA kernel(s) for GPU matching (entry point / skeleton)
- `download_fasta.sh`     — script to download real FASTA data (optional)
- `Makefile`              — build rules (project-dependent; see below)
- `bench-a100/`, `bench-h100/` — benchmark data / scripts for specific GPUs
- `kseq/`                 — lightweight FASTA parser / utilities used in project

If you need a full file map, run `ls -la` in the repository root; the Makefile lists build targets.

## Prerequisites

- Linux (this repo was developed/tested on Linux).
- GNU Make and a C++ compiler (g++).
- CUDA toolkit (nvcc) and a CUDA-capable GPU for the GPU implementation.
- Optional: `python3`, `awk`, or other tools to help parse outputs for benchmarking.

Notes / assumptions:

- I assume the provided `Makefile` contains sensible targets to build the generators and the CUDA binary. If you prefer to build manually, `nvcc` can compile `kernel_skeleton.cu`.

## Build & run

Typical workflow:

1. Build everything (uses the repository `Makefile`):

```bash
make
```

2. If `make` builds specific binaries, inspect the Makefile or list targets with:

```bash
grep -E '^[a-zA-Z0-9_.-]+:' Makefile
```

3. Generate data (example):

```bash
# generate synthetic signatures
./gen_sig  # may print usage/options; check `./gen_sig --help` if supported

# generate a synthetic DNA sample
./gen_sample
```

4. Run the GPU kernel (example):

```bash
# The exact name of the GPU executable depends on the Makefile.
# A likely pattern is a kernel binary named `kernel` or `match_gpu`. Inspect outputs after `make`.
./kernel sample.fasta sigs.txt
```

If your Makefile doesn't produce the expected binary names, build the CUDA file manually (example):

```bash
# compile a simple CUDA binary (adjust flags as needed)
nvcc -O3 kernel_skeleton.cu common.cc -o kernel -lstdc++
```

## Usage examples

Below are illustrative examples — adapt them to the actual flags your compiled binaries accept.

1) Generate 1M-length synthetic sample and 100 signatures:

```bash
./gen_sample --length 1000000 > sample.fasta
./gen_sig --count 100 > sigs.txt
```

2) Run CPU baseline (if provided):

```bash
./cpu_match sample.fasta sigs.txt  # example; actual binary name may differ
```

3) Run GPU version and measure time (example):

```bash
./kernel sample.fasta sigs.txt
```

Check program output for match counts and timings — the code prints timings for CPU and GPU where implemented.

## Design & implementation notes

- The code separates data generation (`gen_sample`, `gen_sig`), CPU reference implementation (`common.cc`), and GPU implementation (`kernel_skeleton.cu`).
- `device_seq_t.h` contains device-side types and helpers for storing sequences compactly for coalesced memory access.
- The kernel skeleton demonstrates mapping sequence offsets and signatures to threads/blocks to find substring matches in parallel. Typical optimisations to consider:
	- Use shared memory for short signatures to reduce global loads.
	- Coalesce memory reads for samples split across threads.
	- Use bit-packed representations (2 bits per base) to reduce memory footprint.

Edge cases to be aware of:

- Very short signatures (length < warp size) — ensure no out-of-bounds thread reads.
- Overlapping matches — if counting overlapping matches is required, ensure algorithm allows it.
- Large sample inputs that exceed GPU memory — use chunked streaming or process in windows.

## Benchmarking

- The `bench-a100` and `bench-h100` folders contain benchmarking scripts/data collected on particular GPUs (NVIDIA A100 and H100). Use these as templates to run your own benchmarks.
- When benchmarking, run multiple iterations and take median times. Also measure data transfer times (host<->device) separately from kernel execution.

Recommended measurement steps:

1. Warm up the GPU with a dummy kernel invocation.
2. Measure H2D (host->device) transfer time, kernel time, and D2H time separately.
3. Repeat runs (e.g., 5–10) and report median.

## Limitations & future work

- This is an educational assignment and focuses on clarity over production-grade robustness.
- Current limitations that could be improved:
	- Streaming large datasets that don't fit in device memory.
	- More advanced string-match algorithms (Aho–Corasick, suffix arrays) adapted for GPUs.
	- Memory packing and vectorised comparisons for shorter signatures.

## License

This repo is provided for coursework. Check the project root for an explicit LICENSE file. If none is present, assume the code is for educational use only and contact the authors for permission before reuse.

## Acknowledgements

- Instructor and course staff for CS3210.
- The kseq FASTA parser included in `kseq/` (lightweight parser commonly used in bioinformatics examples).

---

If you'd like, I can:

- Inspect the `Makefile` in this repo and add exact build/run commands to this README.
- Add short example data and a minimal Makefile target to produce a runnable demo and a small test.

Tell me which you'd prefer and I'll update the README accordingly.
