# cSVR — Fast Multi-Stack Slice-to-Volume Reconstruction via Multi-Scale Unrolled Optimization

cSVR reconstructs a 3D fetal-brain volume from three orthogonal, motion-corrupted 2D
stacks. A multi-scale network predicts the rigid pose of every slice in a single forward
pass, and the posed slices are fused into a volume by gradient-descent super-resolution
(default) or an implicit neural representation. Stacks need no orientation labels.

---

## Requirements

- Linux, NVIDIA GPU (developed and tested on an A6000, 48 GB).
- A CUDA 11.7 toolkit reachable through `CUDA_HOME`. Two NeSVoR kernels are JIT-compiled
  on first use; without a toolkit they fall back to pure PyTorch and reconstruction is
  roughly 6× slower.
- Python 3.11 (installed by uv).

---

## Installation

### Option A — uv (recommended)

The environment is pinned in `pyproject.toml` / `uv.lock` (Python 3.11, PyTorch
2.0.1+cu117, MONAI 1.3.1). With [uv](https://docs.astral.sh/uv/) installed:

```bash
git clone <this repository> cSVR && cd cSVR
uv sync
source .venv/bin/activate     # or prefix commands with: uv run
./download_checkpoints.sh     # fetches the network weights into model_checkpoints/
```

`tinycudann` (used by `--inr-recon`) is not on PyPI and needs `nvcc`, so it is an opt-in
group. Build it for every GPU architecture you will run on:

```bash
TCNN_CUDA_ARCHITECTURES="75;86" CUDA_HOME=/path/to/cuda-11.7 uv sync --group tcnn
```

Notes:

- A plain `uv sync` removes `tinycudann` again (`uv run` does not). Without it the INR
  reconstruction falls back to a slower pure-PyTorch encoding; the default GD
  reconstruction does not use it.
- Pass the toolkit through `CUDA_HOME` rather than by prepending its `bin/` to `PATH`:
  uv picks the first `python3.11` on `PATH`, and a toolkit inside a conda environment
  would make that interpreter the base of `.venv`. Check with `grep home .venv/pyvenv.cfg`.
- On shared file systems, keep uv's cache off quota'd home directories:
  `~/.config/uv/uv.toml` with `cache-dir = "/big/disk/.cache/uv"` and
  `export UV_PYTHON_INSTALL_DIR=/big/disk/.local/uv/python`.

### Option B — conda

A conda export (`env_cSVR.yml`) with the same package versions is in the
[Google Drive folder](https://drive.google.com/drive/folders/14lG-uKZLcrR_jPe-SXfNCmOJfhzCV7mQ?usp=sharing).
Create it with `conda env create -f env_cSVR.yml` and activate it instead of `.venv`.

### CUDA toolkit for the JIT kernels

Before running anything:

```bash
export CUDA_HOME=/usr/local/cuda-11.7          # or a conda env that ships nvcc 11.7
export TORCH_CUDA_ARCH_LIST="7.5;8.6"          # compute capabilities of your GPUs
```

`TORCH_CUDA_ARCH_LIST` matters when you alternate between GPU models: torch's extension
cache does not key on the architecture, so without it a run on a different card triggers
a ~1 min recompile.

The brain-masking checkpoint (MONAIfbs) is downloaded automatically from
[Zenodo](https://zenodo.org/record/4282679) the first time `--preprocess` runs.

---

## Input data

One directory per subject with its three stacks (axial, sagittal, coronal) as `.nii` or
`.nii.gz`, under any filenames. Scanner exports work as they are:

```
sub01/
├── 0005_T2_HASTE.nii.gz
├── 0007_T2_HASTE.nii.gz
└── 0009_T2_HASTE.nii.gz
```

An ordering MLP decides which stack is which and whether each has to be rotated 180°
in-plane; the pipeline re-orients and re-orders the stacks accordingly. Each subject's log
records the decision, per input stack in sorted-filename order:

```
ORDER REVERSE: [1, 0, 1]    # 1 = rotated 180° in-plane
STACK ORDER: [2, 0, 1]      # 0 = sagittal, 1 = coronal, 2 = axial
```

Rules:

- Exactly three stacks per subject directory. Files with `mask` in their name, `._*`
  files and subdirectories are ignored.
- The three stacks must come from the same subject with consistent voxel sizes.
- If the three stacks have an odd total number of brain-containing slices, the last slice
  of the third stack in sorted-filename order is dropped.
- Orientation suffixes (`_sag` / `_cor` / `_axi` before the extension) are optional. They
  are only used to pick three stacks from a directory that holds more, by `--no-mlp`, and
  by the legacy input without `--preprocess`.

With `--preprocess` (recommended) that is all you need: masks, reorientation, N4 bias
correction and normalization are computed in memory. Without `--preprocess` the pipeline
expects stacks that are already masked, bias-corrected and normalized, plus a binary mask
per stack, paired by the `_axi`/`_sag`/`_cor` suffix (e.g. `sub01_axi.nii.gz` +
`mask_axi.nii.gz`).

---

## Running the pipeline

Activate the environment and set `CUDA_HOME` first. All commands run from the repository
root.

Two entry points share the same options. Serve mode is the fast one: the start-up
cost (loading ~13 GB of checkpoints and, on the very first run, JIT-compiling the two
NeSVoR CUDA kernels) is paid once, and every subject after that takes tens of seconds.
The one-shot script pays that cost on every invocation, so a single subject takes
several minutes.

### Option 1 — Serve mode (fast)

```bash
python run_pipeline_cSVR_serve.py --serve /data/sub01 --suffix run1 --preprocess --gd-recon --dest-folder /data/sub01/cSVR_run1
```

The models are loaded and warmed up once, the subjects given on the command line are
processed, and then each line typed or piped at the `cSVR>` prompt is a further job with
the same arguments (`--run-cSVR` is implied). Each job finishes in tens of seconds:

```
cSVR> /data/sub01 --suffix run1 --preprocess --gd-recon --dest-folder /data/sub01/cSVR_run1
cSVR> /data/sub02 --suffix run1 --preprocess --gd-recon --dest-folder /data/sub02/cSVR_run1
cSVR> quit
```

Non-interactively:

```bash
echo "/data/sub01 --suffix run1 --preprocess --gd-recon --dest-folder /data/sub01/cSVR_run1" \
  | python run_pipeline_cSVR_serve.py --serve
```

`run_pipeline_cSVR_serve.py` also works like `run_pipeline_cSVR.py` when called without
`--serve`, and additionally writes a `<subject>_timings_<suffix>.json` with per-stage times.

### Option 2 — One subject, one shot (slow)

```bash
python run_pipeline_cSVR.py /data/sub01 \
    --suffix run1 \
    --run-cSVR --preprocess --gd-recon \
    --dest-folder cSVR_run1
```

This loads the models, preprocesses `sub01`, estimates slice poses, reconstructs with
gradient descent and writes everything to `/data/sub01/cSVR_run1/`. Expect several
minutes for the first subject because of the checkpoint load and kernel compilation;
each further subject in the same process takes 20–30 s. Several subject directories can
be given at once, or listed one per line in a file passed with `--dir-list subjects.txt`;
the models are loaded once for all of them. For more than one or two subjects, prefer
serve mode.

### Reconstruction back-end

| Flag | Back-end |
|---|---|
| `--gd-recon` | Gradient-descent SVR (NeSVoR `svr`, 0.8 mm output). Default choice. |
| `--inr-recon` | Implicit neural representation (NeSVoR `reconstruct`). Uses `tinycudann` when installed. |

Both may be given; each writes its own volume.

### All options

Common to both scripts:

| Option | Meaning |
|---|---|
| `directories` / `--dir-list FILE` | Subject directories to process (positional, and/or one per line in `FILE`; `#` lines are skipped). |
| `--suffix S` | Run name appended to every output filename (default `run1`). |
| `--run-cSVR` | Run pose estimation. Without it only the standardization step runs. Implied in serve mode. |
| `--preprocess` | In-memory masking + reorientation + N4 + normalization from the raw stacks (recommended). |
| `--gd-recon` / `--inr-recon` | Reconstruction back-end(s). |
| `--dest-folder D` | Output directory. `run_pipeline_cSVR.py` resolves it relative to the subject directory; `run_pipeline_cSVR_serve.py` uses it as given. Default: `<subject>/cSVR_files` (`cSVR_files_inr` for INR). |
| `--save-slices-to-disk` | Also write the posed slices as one NIfTI per slice into `<dest>/<subject>_slices/`. |
| `--save-folder D` | Custom directory for those slices. |
| `--output-volume D` | Directory for the reconstructed volume, if different from `--dest-folder`. |
| `--no-mlp` | Skip the ordering MLP and trust the filenames: needs `_sag`/`_cor`/`_axi` suffixes (or three files whose sorted order is already sagittal, coronal, axial). The 180° in-plane flip is then not predicted either. |
| `--normalize-mean` | Normalize stacks by mean intensity instead of the second histogram mode. |
| `--clin` | Clinical slice-spacing flag passed to the slice writer. |
| `--synth` | Layout of the synthetic-motion benchmark (needs its ground-truth files); not for clinical data. |

Only in `run_pipeline_cSVR.py`:

| Option | Meaning |
|---|---|
| `--bias-field-norm` | Without `--preprocess`, read `*_axi_bias_field` / `*_sag_bias_field` / `*_cor_bias_field` stacks instead of the raw ones. |
| `--no-trim-odd` | Keep all slices (by default a stack with an odd slice count loses its last slice). |

Only in `run_pipeline_cSVR_serve.py`:

| Option | Meaning |
|---|---|
| `--serve` | Interactive / stdin job loop, models loaded once. |
| `--timing-mode` | Measure compute only: skips the standardization dumps, the per-slice output and the simulated slices. |
| `--preprocess-chain {norm_bias,bias_norm}` | Order of normalization and N4 in `--preprocess`. `norm_bias` (default) leaves the three stacks intensity-matched; `bias_norm` reproduces the training preprocessing. |
| `--stack-normalize {auto,per_stack,joint,none}` | Rescaling applied after cropping. `auto` (default) picks `none` for `norm_bias` and `per_stack` otherwise. |
| `--dump-preprocessed` | Write the `--preprocess` intermediates into `<dest>/preprocessed/`. |

---

## Outputs

For subject `sub01`, suffix `run1`, output directory `<dest>`:

| File | Content |
|---|---|
| `<dest>/sub01_run1.nii.gz`, `sub01_mask_run1.nii.gz` | Standardized, masked input stacks stacked into one volume (network input), and the mask. |
| `<dest>/sub01_run1.pt`, `init_stack_sub01_run1.pt` | The same as tensors. |
| `<dest>/sub01_slices/` | One NIfTI per slice with its estimated pose in the header (`--save-slices-to-disk`). |
| `<dest>/sub01_cSVR_gd_reconrun1.nii.gz` | Gradient-descent reconstruction. |
| `<dest>/sub01_cSVR_inr_reconrun1.nii.gz` | INR reconstruction (`--inr-recon`). |
| `<dest>/sub01_sim_slices/` | Slices simulated from the reconstruction at the estimated poses. |
| `<dest>/sub01_timings_run1.json` | Per-stage times (serve script only). |

Typical per-subject cost on an A6000 with the CUDA kernels active: ~12 s preprocessing,
2–9 s pose estimation, ~6 s GD reconstruction. On top of that, each process pays a
one-time start-up cost of loading the ~13 GB of checkpoints (from ~30 s on a local SSD to
several minutes on network storage) and, on the first run only, JIT-compiling the CUDA
kernels. In serve mode this start-up cost is paid once for all subjects.

---

## Acknowledgements

- `nesvor_local/` is a modified copy of [NeSVoR](https://github.com/daviddmc/NeSVoR)
  (Xu et al.): slice-acquisition model, GD and INR reconstruction, preprocessing tools.
- Brain masking uses [MONAIfbs](https://github.com/gift-surg/MONAIfbs) (Ranzini et al.).
- `cornucopia/` is a vendored copy of [cornucopia](https://github.com/balbasty/cornucopia)
  (Yaël Balbastre, MIT license, see `cornucopia/LICENSE`) at upstream commit `58369ea`
  with local modifications to `geometric.py` and `utils/warps.py`.
- The INR encoding uses [tiny-cuda-nn](https://github.com/NVlabs/tiny-cuda-nn).

## Contact

Questions, extensions or applications: feel free to reach out by e-mail.
