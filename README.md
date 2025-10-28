# CS3210 Assignment 2 – DNA Virus String Matching (CUDA)  
**Author(s):** [Your Name(s)]  
**Course:** CS3210 – High Performance Computing  
**Semester:** [Term / Year]  
**Repository:** [jiajunchang2002g/CS3210-Assignment2](https://github.com/jiajunchang2002g/CS3210-Assignment2)

---

## Table of Contents  
1. [Project Overview](#project-overview)  
2. [Motivation](#motivation)  
3. [Features](#features)  
4. [Repository Structure](#repository-structure)  
5. [Prerequisites](#prerequisites)  
6. [Building & Running](#building-and-running)  
7. [Usage Examples](#usage-examples)  
8. [Performance & Benchmarking](#performance-&-benchmarking)  
9. [Design & Implementation Details](#design-&-implementation-details)  
10. [Limitations & Future Work](#limitations-&-future-work)  
11. [License](#license)  
12. [Acknowledgements](#acknowledgements)

---

## Project Overview  
This project implements a **string-matching algorithm** optimized for GPUs via CUDA to search for virus DNA signature patterns in large DNA sequences. The goal is to accelerate matching of known viral substrings (signatures) in large reference or sample DNA data, leveraging parallel processing.

---

## Motivation  
- Modern genomic datasets are huge; searching them for specific viral markers is computationally demanding.  
- Using GPU (CUDA) provides massive parallelism to speed up substring matching tasks.  
- This assignment explores how to map string-matching onto CUDA threads, optimise memory access, and compare performance between CPU and GPU versions.

---

## Features  
- Generation of synthetic sample DNA sequences (`gen_sample.cc`).  
- Generation of viral signature sequences (`gen_sig.cc`).  
- A baseline CPU version of matching (perhaps in `common.cc` or related).  
- A CUDA kernel implementation (`kernel_skeleton.cu`) that executes the matching in parallel on the GPU.  
- Benchmarking scripts/folders (e.g., `bench-a100`, `bench-h100`) to record performance on different GPU hardware.  
- Script to download DNA/FASTA data (`download_fasta.sh`).  
- Modular code with header files (`common.h`, `device_seq_t.h`) to promote reuse and clarity.

---

## Repository Structure  
