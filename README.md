# cSVR — Fast Multi-Stack Slice-to-Volume Reconstruction via Multi-Scale Unrolled Optimization

cSVR reconstructs a 3D fetal-brain volume from three orthogonal, motion-corrupted 2D
stacks (axial, sagittal, coronal). A learned multi-scale network predicts the rigid pose of
every slice in one forward pass; the posed slices are then fused into a volume with either
gradient-descent super-resolution (default) or an implicit neural representation (INR).
The stacks need no orientation labels: a network recognizes which one is which.

The pipeline, per subject:

1. **Preprocessing** (`--preprocess`, in memory): brain masking (MONAIfbs), reorientation
   to a canonical frame, N4 bias-field correction, intensity normalization, and
   standardization of the three stacks into the network's 128×128 slice tensors.
2. **Pose estimation**: a small MLP predicts, for each stack, whether it is the sagittal,
   coronal or axial one and whether it must be rotated 180° in-plane; the stacks are
   re-oriented and re-ordered accordingly. Then the multi-scale SVR network predicts a
   rigid transform for every slice.
3. **Reconstruction**: the posed slices are handed in memory to the reconstruction step —
   `--gd-recon` (gradient-descent SVR, 0.8 mm isotropic output) or `--inr-recon` (NeSVoR
   INR).

This branch contains the files needed to run the pipeline, plus the training script and
the configs that produced the released checkpoints (see *Training* below).

---

## Contents

| Path | Purpose |
|---|---|
| `run_pipeline_cSVR.py` | One-shot entry point: preprocess → pose estimation → reconstruction for one or more subjects. |
| `run_pipeline_cSVR_serve.py` | Same pipeline plus a **serve mode** that loads the models once and processes subjects fed on stdin; also writes per-stage timings. Used by the SLURM script. |
| `preprocess_in_memory.py` | In-memory preprocessing chain (masking, reorientation, N4, normalization). |
| `nifti_utils.py` | Stack loading / standardization into network tensors. |
| `run_cSVR_fast.py`, `slice_saver.py` | Pose-estimation inference and slice export. |
| `gd_recon.py`, `inr_recon.py` | Thin wrappers that call the reconstruction back-ends in `nesvor_local`. |
| `get_TRE_from_file_outputs_og.py` | Evaluation: slice similarity (NCC / SSIM / PSNR) between the input slices and slices simulated from the reconstruction. |
| `models/`, `datasets/` | Network definitions and tensor utilities. |
| `nesvor_local/` | Modified copy of [NeSVoR](https://github.com/daviddmc/NeSVoR): slice-acquisition CUDA kernels, GD/INR reconstruction, MONAIfbs masking, N4. |
| `cornucopia/` | Vendored copy of [cornucopia](https://github.com/balbasty/cornucopia) (MIT) with local modifications; provides the affine-flow / slice-simulation utilities the networks use. Imported in place of the PyPI package. |
| `job_eval_slurm.sh` | SLURM array-job example (one subject per task, then evaluation). |
| `train_from_bulk_wdb_yaml.py`, `options.py`, `experiments/` | Training script, its argparse defaults, and the configs of the released checkpoints (see `experiments/README.md`). |
| `job_train_slurm.sh`, `generate_resume_yaml.py` | SLURM training job and resume-config helper. |
| `pyproject.toml`, `uv.lock`, `.python-version` | Pinned Python environment (uv). |
| `model_checkpoints/`, `nesvor_local/checkpoints/` | Where the network weights go (not in git, see below). |

---

## Requirements

- Linux, NVIDIA GPU. Developed and tested on an NVIDIA A6000 (48 GB); the SVR network
  checkpoint alone is 7.9 GB on disk.
- A **CUDA 11.7 toolkit** (`nvcc`, headers, `libcudart`) reachable through `CUDA_HOME`.
  PyTorch itself comes with its own CUDA runtime, but two NeSVoR kernels
  (`slice_acq_cuda`, `transform_convert_cuda`) are JIT-compiled by `torch.utils.cpp_extension`
  on first use. Without a toolkit they silently fall back to pure-PyTorch implementations
  that make reconstruction roughly 6× slower.
- Python 3.11 (installed automatically by uv).

---

## Installation

### Option A — uv (recommended)

The environment is pinned in `pyproject.toml` / `uv.lock` (Python 3.11, PyTorch
2.0.1+cu117, MONAI 1.3.1, …). With [uv](https://docs.astral.sh/uv/) installed:

```bash
git clone <this repository> cSVR && cd cSVR
uv sync                       # creates ./.venv with every pinned dependency
source .venv/bin/activate     # or prefix commands with: uv run
```

`tinycudann` (the fast hash-grid encoding used by `--inr-recon`) is not on PyPI and needs
`nvcc`, so it is an opt-in dependency group. Build it for every GPU architecture you will run
on; `.cuda_home` here stands for any CUDA 11.7 toolkit directory:

```bash
TCNN_CUDA_ARCHITECTURES="75;86" CUDA_HOME=/path/to/cuda-11.7 uv sync --group tcnn
```

Notes:

- A plain `uv sync` removes `tinycudann` again (`uv run` does not). When it is missing, the
  INR reconstruction automatically falls back to a slower pure-PyTorch encoding; the default
  GD reconstruction does not use it at all.
- tiny-cuda-nn links against `libnvrtc`, so `$CUDA_HOME/lib64/libnvrtc.so` must resolve.
  Pass the toolkit through `CUDA_HOME` rather than by prepending its `bin/` to `PATH`:
  uv picks the first `python3.11` it finds on `PATH`, and a toolkit that lives inside a
  conda environment would make that environment's interpreter the base of `.venv`. Check
  with `grep home .venv/pyvenv.cfg` — it should point at a uv-managed CPython
  (`uv python list`).
- On shared file systems, keep uv's cache off slow/quota'd home directories:
  `~/.config/uv/uv.toml` with `cache-dir = "/big/disk/.cache/uv"` and
  `export UV_PYTHON_INSTALL_DIR=/big/disk/.local/uv/python`.

### Option B — conda

A conda environment export (`env_cSVR.yml`) with the same package versions is available in
the [Google Drive folder](https://drive.google.com/drive/folders/14lG-uKZLcrR_jPe-SXfNCmOJfhzCV7mQ?usp=sharing).
Create it with `conda env create -f env_cSVR.yml` and activate it instead of `.venv`.

### CUDA toolkit for the JIT kernels

Point `CUDA_HOME` at a CUDA 11.7 installation before running anything:

```bash
export CUDA_HOME=/usr/local/cuda-11.7          # or a conda env that ships nvcc 11.7
export TORCH_CUDA_ARCH_LIST="7.5;8.6"          # compute capabilities of your GPUs
```

`TORCH_CUDA_ARCH_LIST` matters when you alternate between GPU models: torch's extension
cache (`~/.cache/torch_extensions/`) does not key on the architecture, so without a pinned
list a run on a different card triggers a ~1 min recompile.

---

## Checkpoints

Three weight files are needed. They are not tracked in git.

| File | Size | Used by | Where to put it |
|---|---|---|---|
| `cSVR_SVR.ckpt` | 7.9 GB | pose estimation (multi-scale SVR network, `models.flow_SNet3d2_1024_multi_crop`) | `model_checkpoints/` |
| `cSVR_MLP.ckpt` | 5.7 GB | stack orientation / order prediction (`models.flow_SNet3d2_1024_MLP`) | `model_checkpoints/` |
| `checkpoint_dynUnet_DiceXent.pt` | 366 MB | MONAIfbs brain masking (`--preprocess`) | `nesvor_local/checkpoints/` |

- The two cSVR checkpoints download from the command line with

  ```bash
  ./download_checkpoints.sh
  ```

  which fetches them from the [Hugging Face Hub](https://huggingface.co/mafirenze/csvr_v2)
  into `model_checkpoints/` (or `$CSVR_CHECKPOINT_DIR` if set) under the names above,
  verifying sizes and sha256 checksums. Interrupted downloads resume where they left
  off on the next run. Checkpoints already in place are verified as well, so a file left
  over from an earlier release is reported instead of silently reused. They can also be
  fetched manually from that repo page, e.g. with
  `hf download mafirenze/csvr_v2 cSVR_SVR.ckpt cSVR_MLP.ckpt --local-dir model_checkpoints`.
- Each release has its own Hub repo. `csvr_v2` (this release) pairs the original SVR
  network with a retrained ordering MLP (see *Training*); the first release's checkpoints
  are still available at [mafirenze/csvr_v1](https://huggingface.co/mafirenze/csvr_v1).
- The MONAIfbs checkpoint is **downloaded automatically** from
  [Zenodo](https://zenodo.org/record/4282679) into `nesvor_local/checkpoints/` the first
  time `--preprocess` runs, if it is not already there.
- To keep the cSVR checkpoints elsewhere, set `CSVR_CHECKPOINT_DIR=/path/to/dir`; the two
  filenames stay the same.

---

## Input data

One directory per subject containing its **three stacks** (axial, sagittal, coronal) as
`.nii` or `.nii.gz`, under any filenames. Scanner exports work as they are:

```
sub01/
├── 0005_T2_HASTE.nii.gz
├── 0007_T2_HASTE.nii.gz
└── 0009_T2_HASTE.nii.gz
```

By default the ordering MLP determines which stack is the sagittal, coronal and axial one,
and whether each has to be rotated 180° in-plane; the pipeline re-orients and re-orders
the stacks accordingly. The filenames play no part in this, and the files can come in any
order. Each subject's log records the decision, per input stack in sorted-filename order:

```
ORDER REVERSE: [1, 0, 1]    # 1 = rotated 180° in-plane
STACK ORDER: [2, 0, 1]      # 0 = sagittal, 1 = coronal, 2 = axial
```

Rules (see `find_raw_stacks` in `preprocess_in_memory.py`):

- Exactly three stacks per subject directory. Files with `mask` in their name, AppleDouble
  `._*` files and subdirectories (where the outputs go) are ignored.
- The three stacks must come from the same subject with consistent voxel sizes.
- File order still has one small effect: if the three stacks have an odd total number of
  brain-containing slices, the last slice of the third stack in sorted-filename order is
  dropped to make the total even.
- Orientation suffixes (`_sag` / `_cor` / `_axi` as the last thing before the extension)
  are optional. They are only used to pick three stacks from a directory that holds more
  than three, by `--no-mlp` (which skips the MLP and trusts them), and by the legacy
  input without `--preprocess` described next.

With `--preprocess` (recommended) that is all you need: masks, reorientation, N4 bias
correction and normalization are computed in memory. Without `--preprocess` the pipeline
expects stacks that were already masked, bias-corrected and normalized, plus a binary mask
per stack; this legacy path still finds the stacks, and pairs them with their masks, by
the `_axi`/`_sag`/`_cor` suffix (e.g. `sub01_axi.nii.gz` + `mask_axi.nii.gz`).

---

## Running the pipeline

Activate the environment and set `CUDA_HOME` first (see above). All commands run from the
repository root.

### One subject, one shot

```bash
python run_pipeline_cSVR.py /data/sub01 \
    --suffix run1 \
    --run-cSVR --preprocess --gd-recon \
    --dest-folder cSVR_run1
```

This loads the models (~30 s), preprocesses `sub01`, estimates slice poses, reconstructs
with gradient descent and writes everything to `/data/sub01/cSVR_run1/`. Several subject
directories can be given at once, or listed one per line in a file passed with
`--dir-list subjects.txt`; the models are loaded once for all of them.

Reconstruction back-end:

| Flag | Back-end |
|---|---|
| `--gd-recon` | Gradient-descent SVR (NeSVoR `svr`: 5 outer iterations, 3 reconstruction iterations, no global outlier exclusion, 0.8 mm output). Default choice. |
| `--inr-recon` | Implicit neural representation (NeSVoR `reconstruct`). Uses `tinycudann` when installed. |

Both may be given; each writes its own volume.

### Serve mode (many subjects, models stay loaded)

```bash
python run_pipeline_cSVR_serve.py --serve
```

The models are loaded and warmed up once; then each line typed (or piped) at the `cSVR>`
prompt is a job with the same arguments as above (`--run-cSVR` is implied):

```
cSVR> /data/sub01 --suffix run1 --preprocess --gd-recon --dest-folder /data/sub01/cSVR_run1
cSVR> /data/sub02 --suffix run1 --preprocess --gd-recon --dest-folder /data/sub02/cSVR_run1
cSVR> quit
```

This is what `job_eval_slurm.sh` does non-interactively:

```bash
echo "/data/sub01 --suffix run1 --preprocess --gd-recon --dest-folder /data/sub01/cSVR_run1" \
  | python run_pipeline_cSVR_serve.py --serve
```

`run_pipeline_cSVR_serve.py` also works exactly like `run_pipeline_cSVR.py` when called
without `--serve`, and additionally writes a `<subject>_timings_<suffix>.json` with the
per-stage times.

### All options

Common to both scripts:

| Option | Meaning |
|---|---|
| `directories` / `--dir-list FILE` | Subject directories to process (positional, and/or one per line in `FILE`; `#` lines are skipped). |
| `--suffix S` | Run name appended to every output filename (default `run1`). |
| `--run-cSVR` | Run pose estimation. Without it only the standardization step runs. Implied in serve mode. |
| `--preprocess` | In-memory masking + reorientation + N4 + normalization from the raw stacks (recommended). |
| `--gd-recon` / `--inr-recon` | Reconstruction back-end(s), see above. |
| `--dest-folder D` | Output directory. `run_pipeline_cSVR.py` resolves it **relative to the subject directory**; `run_pipeline_cSVR_serve.py` uses it **as given** (absolute path). Default: `<subject>/cSVR_files` (`cSVR_files_inr` for INR). |
| `--save-slices-to-disk` | Also write the posed slices as one NIfTI per slice into `<dest>/<subject>_slices/`. Needed for evaluation; otherwise slices only live in memory. |
| `--save-folder D` | Custom directory for those slices. |
| `--output-volume D` | Directory for the reconstructed volume, if different from `--dest-folder`. |
| `--no-mlp` | Skip the ordering MLP, which runs by default, and trust the filenames instead: needs `_sag`/`_cor`/`_axi` suffixes (or three files whose sorted order is already sagittal, coronal, axial). The 180° in-plane re-orientation is then not predicted either; a fixed default is used. |
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
| `--timing-mode` | Measure compute only: skips the standardization dumps, the per-slice output and the simulated slices. The volume and the timings JSON are still written; nothing can be evaluated afterwards. |
| `--preprocess-chain {norm_bias,bias_norm}` | Order of normalization and N4 in `--preprocess`. `norm_bias` (default) leaves the three stacks intensity-matched; `bias_norm` reproduces the training preprocessing. |
| `--stack-normalize {auto,per_stack,joint,none}` | Rescaling applied after cropping. `auto` (default) picks `none` for `norm_bias` and `per_stack` otherwise. The ordering MLP always normalizes its own copy of each stack, whatever this is set to. |
| `--dump-preprocessed` | Write the `--preprocess` intermediates (bias-corrected, normalized stacks and masks) into `<dest>/preprocessed/` for inspection. |

---

## Outputs

For subject `sub01`, suffix `run1`, output directory `<dest>`:

| File | Content |
|---|---|
| `<dest>/sub01_run1.nii.gz`, `sub01_mask_run1.nii.gz` | Standardized, masked input stacks stacked into one volume (network input), and the mask. |
| `<dest>/sub01_run1.pt`, `init_stack_sub01_run1.pt` | The same as tensors (skipped in `--timing-mode`). |
| `<dest>/sub01_slices/` | One NIfTI per slice with its estimated pose in the header (`--save-slices-to-disk`). |
| `<dest>/sub01_cSVR_gd_reconrun1.nii.gz` | Gradient-descent reconstruction (the suffix is appended directly to the name). |
| `<dest>/sub01_cSVR_inr_reconrun1.nii.gz` | INR reconstruction (`--inr-recon`). |
| `<dest>/sub01_sim_slices/` | Slices simulated from the reconstruction at the estimated poses; compared with the input slices by the evaluation script. |
| `<dest>/sub01_timings_run1.json` | `time_preprocess_s`, `time_pose_estimation_s`, `time_reconstruction_s` (serve script only). |
| `sub01/*.nii.gz` debug volumes | A few intermediate volumes (original slices, reoriented slices, splatted volume) written next to the subject; disabled in `--timing-mode`. |

Typical per-subject cost on an A6000 with the CUDA kernels active: ~12 s preprocessing,
2–9 s pose estimation, ~6 s GD reconstruction, plus ~30 s one-time model loading.

---

## Evaluation

`get_TRE_from_file_outputs_og.py` scores a run by comparing the posed input slices with the
slices simulated from the reconstruction (no ground truth needed):

```bash
python get_TRE_from_file_outputs_og.py \
    --og_slices  /data/sub01/cSVR_run1/sub01_sim_slices \
    --folder_est /data/sub01/cSVR_run1/sub01_slices \
    --json_path  evaluate_metrics/run1_0.json \
    --img_num 0 --img_name sub01 \
    --timings_json /data/sub01/cSVR_run1/sub01_timings_run1.json \
    --clin True --only_slice_sim
```

The JSON gets one entry per subject with `slice_ncc_mean`, `slice_ncc_med`,
`slice_ssim_mean`, `slice_ssim_med`, `slice_psnr_mean` and, if `--timings_json` is given,
the three stage times. `--only_slice_sim` skips the target-registration-error part, which
needs ground-truth poses and only applies to the synthetic benchmark. The run must have
used `--save-slices-to-disk` (both slice folders have to exist on disk).

---

## SLURM

`job_eval_slurm.sh` runs one subject per array task (serve mode, `--preprocess`,
`--gd-recon`, `--save-slices-to-disk`) and then the evaluation. Submit it from the
repository root; it finds the code and `.venv` through `SLURM_SUBMIT_DIR`:

```bash
mkdir -p train_outs evaluate_metrics
export DATA_DIR=/data/subjects          # one sub-directory per subject
export SUFFIX=run1                      # outputs: $DATA_DIR/<subject>/cSVR_run1/
export SUBJECT_LIST=$DATA_DIR/subjects.txt   # optional; default $DATA_DIR/subjects.txt
sbatch --array=0-17 -p <partition> --gres=gpu:1 job_eval_slurm.sh          # eval mode
sbatch --array=0-17 -p <partition> --gres=gpu:1 job_eval_slurm.sh timing   # timing mode
```

`subjects.txt` lists one subject directory name per line; the array indices select lines.
Logs go to `train_outs/o_cSVR_<jobid>_<index>.out`, metrics to
`evaluate_metrics/<SUFFIX><index>.json`. The script also sets `TORCH_EXTENSIONS_DIR` to
`./.torch_extensions` so the JIT-built kernels are cached per checkout (the first job
spends ~100 s compiling them; later jobs reuse the build). Merge the
per-task JSONs afterwards with e.g.

```bash
python -c "import json,glob; d={}; [d.update(json.load(open(f))) for f in sorted(glob.glob('evaluate_metrics/run1*.json'))]; json.dump(d, open('evaluate_metrics/combined_run1.json','w'), indent=2)"
```

Site-specific settings (partition, account, QoS, GPU type) are passed on the `sbatch`
command line. The script sets `CUDA_HOME` to `./.cuda_home` if that directory exists, so a
symlink to a CUDA 11.7 toolkit at `.cuda_home` is a convenient way to satisfy the JIT
kernels on a cluster.

---

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `Fail to load CUDA extention for slice_acq. Will use pytorch implementation.` | No CUDA 11.7 toolkit found: set `CUDA_HOME`. Reconstruction still works but is ~6× slower. Serve mode prints `nesvor CUDA extensions active` when everything is right. |
| Import hangs at start-up | Stale lock from a killed JIT build: `rm -f ~/.cache/torch_extensions/py311_cu117/*/lock`. |
| `Could not find compatible tinycudann extension for compute capability NN` | `tinycudann` was built for another GPU. Rebuild with `TCNN_CUDA_ARCHITECTURES` covering `NN`, or ignore it: only `--inr-recon` uses it, and it falls back automatically. |
| `expected 3 stacks in … found N` / `… holds N stacks …: keep only the three to reconstruct` | Leave exactly three stacks in the subject directory, or mark the three to use with `_sag`/`_cor`/`_axi` suffixes (see *Input data*). |
| `FileNotFoundError: … model_checkpoints/cSVR_SVR.ckpt` | Run `./download_checkpoints.sh`, or set `CSVR_CHECKPOINT_DIR` to where the checkpoints live. |
| CUDA out of memory | Another process is using the GPU, or the card is too small. Pick a free GPU with `CUDA_VISIBLE_DEVICES`. |
| Evaluation finds no slices | The run did not use `--save-slices-to-disk`, or used `--timing-mode`. |

**Reproducibility.** Preprocessing and pose estimation are deterministic (bit-identical
across runs and across the uv / conda environments). The gradient-descent reconstruction
uses atomic GPU operations, so two runs of the same subject typically differ by
mean |Δ| ≈ 2·10⁻³ of the intensity range (correlation > 0.9997) with slice metrics moving
by ≈ 10⁻³; occasionally the robust outlier weighting settles differently and a run lands
≈ 8·10⁻³ away (metrics ± 6·10⁻³) — observed once in six repeats of the same subject.

---

## Training

> **Note: the MLP training code in this branch is not up to date.** The released ordering
> MLP was trained with a different version of the training code that is not included
> here, so this branch's script cannot reproduce it (see below).

The SVR checkpoint was trained with `train_from_bulk_wdb_yaml.py`:

```bash
python train_from_bulk_wdb_yaml.py --config experiments/feb17_node5_repeat_180_in_plane.yaml  # cSVR_SVR.ckpt
```

The ordering MLP of this release comes from `experiments/feb16_mlp_recreate_merode_tr20.yaml`
(wider through-plane rotations and brain-mask erosion during training), trained with the
February version of the training code rather than this branch's script and selected by
clinical stack-order accuracy; `experiments/feb16_mlp_only_rots_eps0.yaml` is the first
release's MLP. `experiments/README.md` has the details.

Training needs its own environment (pytorch-lightning 1.9 — the pinned uv environment
above is inference-only) and local copies of FeTA 2.1 and the CRL atlas.
See `experiments/README.md` for the environment, data layout, wandb logging,
checkpoints/resume behavior, and `job_train_slurm.sh` for a SLURM template.

---

## Acknowledgements

- `nesvor_local/` is a modified copy of [NeSVoR](https://github.com/daviddmc/NeSVoR)
  (Xu et al.), which provides the slice-acquisition model, the GD and INR reconstruction
  and the preprocessing tools.
- Brain masking uses the [MONAIfbs](https://github.com/gift-surg/MONAIfbs) fetal-brain
  segmentation network (Ranzini et al.).
- `cornucopia/` is a vendored copy of [cornucopia](https://github.com/balbasty/cornucopia)
  (Yaël Balbastre, MIT license, see `cornucopia/LICENSE`) at upstream commit `58369ea`
  with local modifications to `geometric.py` and `utils/warps.py` (a Gaussian
  slice-profile option for the slice-wise affine transform). The networks were trained and
  validated with this version, which is why it is shipped in-tree rather than installed
  from PyPI.
- The INR encoding uses [tiny-cuda-nn](https://github.com/NVlabs/tiny-cuda-nn).

## Contact

Questions, extensions or applications: feel free to reach out by e-mail.
