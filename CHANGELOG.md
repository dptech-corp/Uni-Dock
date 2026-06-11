# Changelog

All notable changes to Uni-Dock (the V1 GPU docking engine in `unidock/`) are documented here.

## [1.2.0] - 2026-06-11

Hardware support for Blackwell GPUs plus a batch of correctness and crash fixes.
The grid-buffer and macrocycle fixes **change docked results** for the affected
inputs (they correct previously-wrong/undefined behavior) — see notes below.

### Added
- **Blackwell / CUDA 12.8 support** — build for `sm_100` (B100/B200/GB200) and `sm_120` (RTX 50/B40); CUDA bumped to 12.8 (#184).

### Fixed — correctness (results change for affected inputs)
- **Grid buffer sized for the default box** — `MAX_NUM_OF_GRID_POINT` 80³ → 81³ (531,441), so the default 30 Å / 0.375 Å box fits exactly. Previously the bounds check was a release-stripped `assert` and the grid silently overflowed `grid_cuda_t::m_data`, corrupting ~20% of results for boxes ≥ 30 Å and crashing (SIGSEGV) on larger boxes (#195, #177; closes #174, #14).
- **Large boxes now error cleanly** — boxes exceeding the GPU grid buffer print a clear message and exit instead of silently overflowing / SIGSEGV (#177).
- **Macrocycle (Meeko `CG`/`G`) affinities** — the ring-closure "glue" energy no longer leaks into the reported affinity (it's a sampling artifact used only to close the ring during optimization). Fixes absurd, non-reproducible scores for macrocycle ligands (#196; closes #102, #192).
- **Out-of-bounds poses** are skipped instead of being reported with `FLT_MAX` (3.4e38) affinities (#182; closes #144).

### Fixed — crashes
- **CUDA error 700 (illegal address) in `--gpu_batch`** from a `cudaMemset` pointer bug (`&ig_cuda_gpu` → `ig_cuda_gpu`) (#178; addresses #115/#118/#136).
- **Segfault when a ligand fails to parse in a batch** — the bad ligand is now skipped with a warning instead of crashing the batch (#181; closes #112).
- **SIGSEGV on unsupported atom types in SDF** (e.g. boron) — SDF atoms are now validated like the PDBQT path and unsupported ones are skipped (#180; closes #138).
- **Intermittent SIGSEGV from a NaN / non-unit axis in `angle_to_quaternion`** — guarded with a renormalize / identity-quaternion fallback (#179; closes #160).
- **CUDA error 700 with flexible residues (`--flex`)** — ≤ 1 flex torsion now docks on GPU; > 1 exits with a clear "supports at most 1" message instead of crashing (#176; closes #159).

### Build / CI
- **Fix `-DFETCH_BOOST=ON` against system Boost ≥ 1.90** — link the fetched `Boost::math` into the CUDA target so nvcc doesn't pick up the system headers (#189).
- Repair the (manually-dispatched) benchmark workflow: call the current `run_test.py` entry point and download the dataset (#194).
- Pin Python 3.12 for the tools/CI images (openbabel compatibility) (#183, #188); dependency bumps (#185, #186, #187).

### Changed / Removed
- License unified to **Apache-2.0** (#153, #155).
- Removed the built-in receptor processor (it didn't support most user proteins) (#164).

### Known limitations
- Macrocycle **sampling**: the broken ring isn't held as tightly closed as AutoDock during the GPU search (glue equilibrium ~1.5 Å vs ~0.03 Å), so absolute affinities for macrocycles can be weaker than AutoDock even though scores are now sane and reproducible (tracked in #37).
- GPU flexible-receptor docking supports at most 1 flex torsion (#175 prototype for more).

[1.2.0]: https://github.com/dptech-corp/Uni-Dock/compare/1.1.3...1.2.0
