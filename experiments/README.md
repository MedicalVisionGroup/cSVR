# Training

`train_from_bulk_wdb_yaml.py` (repo root) trains the two networks the pipeline uses. The
configs in this directory are the ones behind the released checkpoints:

| Config | Trains | Released checkpoint |
|---|---|---|
| `feb17_node5_repeat_180_in_plane.yaml` | multi-scale SVR network (`models.flow_SNet3d2_1024_multi_crop`) | `cSVR_SVR.ckpt` (both releases) |
| `feb16_mlp_recreate_merode_tr20.yaml` | stack-ordering MLP (`models.flow_SNet3d2_1024_MLP`) | `cSVR_MLP.ckpt` of this release (`csvr_v2`), see *The v2 ordering MLP* |
| `feb16_mlp_only_rots_eps0.yaml` | stack-ordering MLP (`models.flow_SNet3d2_1024_MLP`) | `cSVR_MLP.ckpt` of the first release (`csvr_v1`) |

> **Note: the MLP training code in this branch is not up to date.** Both released ordering
> MLPs were trained with a different version of the training code that is not included
> here (see *Environment* and *The v2 ordering MLP*); this branch's script does not
> reproduce them. The two MLP configs above document their recipes.

The mapping comes from the run-name convention: every run is named
`{dataset}_{network}_{loss}_{rotations}_{translations}_{noise}_{bulk_rotations_plane}_{bulk_rotations_tr_plane}_lr{lr_start}_{remarks_add}`
and checkpoints are written to `./checkpoints/<run name>/{best,last}.ckpt`. The released
`.ckpt` files are Lightning checkpoints of the `feb17_node5_repeat_180_in_plane` SVR run
and of the MLP runs `_MLP_only_order_eps0` (v1) and `_MLP_only_order_eps0_recreate_merode_tr20`
(v2); `eps0` = label-smoothing `eps=0`, the default of the `classification_multihot_order`
loss. The February script that trained both MLPs wrote `bulk_rotations_tr_plane` into
*both* rotation fields of the name, so their run directories read `..._12_12_...` (v1) and
`..._20_20_...` (v2) although both used `bulk_rotations_plane: 180`.

## Environment

Training targets the **pytorch-lightning 1.9 API** (`Trainer.from_argparse_args`), which
was removed in Lightning 2.x — so the pinned uv environment of this repository
(`pyproject.toml`, PL 2.1.0) runs the *pipeline* but **not** this training script. Use a
separate environment, approximately:

- Python 3.10
- torch 1.13.1 (CUDA 11.7), torchvision 0.14.1
- pytorch-lightning 1.9.2, torchmetrics 0.11.4
- torch-interpol, wandb, nibabel, scipy
- cornucopia: the vendored copy at the repo root is picked up automatically when you run
  from the repository root (do not `pip install cornucopia`).

Both ordering MLPs were trained with the February version of the training code in a
different environment: Python 3.10.18, torch 2.5.1 (CUDA 12.1), pytorch-lightning 1.9.0,
and cornucopia 0.3.0 installed from PyPI instead of the vendored copy. Retrains of the MLP
in the environment above did not reach the released model's accuracy.

## Data

The dataset factory (`feta3d0[_mlp]_multi_stack_svr_final_sb2_crop` in
`datasets/feta.py`) synthesizes motion-corrupted stacks on the fly from segmented
volumes. It expects, relative to the repository root:

- `../public_data/FeTA/feta_2.1_mial/` — [FeTA 2.1](https://feta.grand-challenge.org/)
  volumes, registered to a common frame and intensity-normalized, laid out as
  `<sub>/anat/<sub>_rec-<rec>_T2w_norm_reg.nii` (plus `_T2w_reg.nii` and
  `_dseg_reg.nii`), where `<rec>` is `mial`/`irtk`/`nmic` depending on the subject
  number. Subjects used: `datasets/feta_2.1/train` and `datasets/feta_2.1/val`.
- `../public_data/CRL/CRL_FetalBrainAtlas_2017v3_lia/` — the
  [CRL fetal brain atlas](https://crl.med.harvard.edu/research/fetal_brain_atlas/) as
  `<name>.nii.gz` + `<name>_regional.nii.gz`; names in `datasets/crl/train`. Mixed into
  the training set as extra volumes.
- `../MIAL/lia/sub-01/anat/` — one real acquired multi-stack subject
  (`sub-01_run-{1..6}_T2w_norm.nii.gz` + masks), used as the test loader.

The registration/normalization preprocessing that produced these local copies is not part
of this repository.

## Running

From the repository root, with the training environment active:

```bash
export WANDB_ENTITY=<your entity>      # or WANDB_MODE=offline
python train_from_bulk_wdb_yaml.py --config experiments/feb17_node5_repeat_180_in_plane.yaml
```

or through SLURM (`job_train_slurm.sh`):

```bash
mkdir -p train_outs
export CONFIG=experiments/feb17_node5_repeat_180_in_plane.yaml CONDA_ENV=<pl19 env>
sbatch -p <partition> --gres=gpu:1 job_train_slurm.sh
```

Notes:

- Progress is logged to [wandb](https://wandb.ai) (`WANDB_ENTITY` / `WANDB_PROJECT`
  override the destination, `WANDB_MODE=offline` disables the account requirement). The
  run id is stored in the checkpoint directory so resubmissions resume the same run.
- Checkpoints go to `./checkpoints/<run name>/`: `best.ckpt` (lowest `val_loss`) and
  `last.ckpt`. If `last.ckpt` exists the script resumes from it automatically;
  `generate_resume_yaml.py <config>` writes an explicit resume config instead.
- A run trains for `max_steps: 250000` optimizer steps (batch size 1); on an A6000 that
  is on the order of a week — hence the 7-day time limit in `job_train_slurm.sh`.
- The training losses/metrics (`l22_loss_grid_multiscale`, `classification_multihot_order`,
  `TRELoss`, `MLP_CorrectVec`) live in `models/spherical.py`, `models/spherical_rec.py`
  and `models/metrics.py`.

## The v2 ordering MLP

`cSVR_MLP.ckpt` of this release (`csvr_v2`) comes from `feb16_mlp_recreate_merode_tr20.yaml`,
which changes two things relative to the v1 recipe (`feb16_mlp_only_rots_eps0.yaml`):

- `bulk_rotations_tr_plane: 20` (v1: 12): a wider range of random through-plane rotation
  per stack.
- `mask_erode: 0.5`, `mask_cut_max: 0.35`: for half of the training volumes the brain mask
  is eroded by a random radius of up to 0.35 × half its smallest extent (erosions keeping
  less than a quarter of the mask are skipped), and the image is zeroed where the mask was
  removed, imitating the under-segmented masks of clinical stacks.

It was trained with the February training code that also produced v1, plus the
mask-erosion option: branch `feb17_mlp_recreate`, commit `9126481`, in the MLP environment
described under *Environment*, on one A6000. **This branch's script cannot reproduce it:**
its `datasets/` code has no mask erosion, and the two `mask_*` keys would be swallowed by
`**kwargs` without an error.

The released weights are the checkpoint after 35 epochs (global step 56,700 of the
250,000-step schedule), picked by stack-order accuracy on clinical data rather than by
`val_loss`: on the synthetic validation data this checkpoint was never the `val_loss`
minimum, and the clinical accuracy swung widely between nearby epochs.
