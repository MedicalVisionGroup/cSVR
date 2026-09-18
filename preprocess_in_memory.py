"""In-memory preprocessing for the cSVR pipeline.

Reproduces the disk preprocessing chain (get_masks -> reorder_masks ->
apply_bias_field -> normalize_stacks) entirely in memory, so the pipeline can go
straight from raw stacks to cSVR without writing/reading the intermediate
_re / _re_bias_field / _re_bias_field_norm files.

Each step was validated to reproduce the disk output exactly on sub2:
  - masking (MONAIfbs segmentation): Dice 0.9999 vs disk masks
  - reorient (det<0 flip + mask affine): data + affine identical
  - bias (N4 + brain-mask zeroing): 0.0 max abs diff vs disk _re_bias_field
  - normalize (normalize_by_second_mode): 0.0 max abs diff vs disk _re_bias_field_norm

The two nesvor <-> nii-array bridges mirror load_stack() and save_nii_volume()
exactly, so a nii -> Stack -> nii round-trip is byte-identical to the disk path
(same affine via transformation2affine, same transpose, same qform/sform).
"""

import glob
import os
from argparse import Namespace

import numpy as np
import torch
import nibabel as nib

from nesvor_local.image import Stack
from nesvor_local.image.image_utils import affine2transformation, transformation2affine
from nesvor_local.preprocessing import n4_bias_field_correction, brain_segmentation
from nifti_utils import normalize_by_second_mode

# N4 defaults, matching the `nesvor correct-bias-field` parser defaults used by
# apply_bias_field.py (which passes no overrides).
N4_DEFAULTS = {
    "n_proc_n4": 8, "shrink_factor_n4": 2, "tol_n4": 0.001, "spline_order_n4": 3,
    "noise_n4": 0.01, "n_iter_n4": 50, "n_levels_n4": 4, "n_control_points_n4": 4,
    "n_bins_n4": 200,
}

# Segmentation defaults, matching the `nesvor segment-stack` parser defaults -- which is
# how get_masks.py produced the v1 mask_* files, since it passes no overrides.
# dilation_radius_seg is 1.0 mm there, not 0: at 0 the masks come out at 92-94% of the v1
# volume on every stack of every subject (~1 voxel of radius), which trims the outer rim
# of brain off the stacks before any method sees them.
SEG_DEFAULTS = dict(batch_size_seg=16, no_augmentation_seg=False,
                    dilation_radius_seg=1.0, threshold_small_seg=0.1)


def _strip_ext(name):
    for ext in (".nii.gz", ".nii"):
        if name.endswith(ext):
            return name[: -len(ext)]
    return name


def find_raw_stacks(subject_dir):
    """Return the 3 raw stack paths.

    Filenames need no orientation label: the ordering MLP decides which stack is
    sagittal / coronal / axial (constrained to a permutation, see run_cSVR_fast), so
    the order returned here only decides the outcome with --no-mlp.

    A folder holding exactly three stacks is taken as is: in (sag, cor, axi) order when
    every stack carries a _sag/_cor/_axi suffix (the layout the SVR net's stack slots
    assume, which is what --no-mlp relies on), otherwise sorted by basename. With more
    than three stacks (e.g. the older clin_data_processed_svr layout with
    _re/_bias_field/_norm copies) the suffixes say which three are meant. mask_* and
    AppleDouble (._*) files are excluded throughout.
    """
    files = (glob.glob(os.path.join(subject_dir, "*.nii"))
             + glob.glob(os.path.join(subject_dir, "*.nii.gz")))
    candidates = sorted(
        f for f in files
        if not os.path.basename(f).startswith("._") and "mask" not in os.path.basename(f).lower()
    )

    def by_orientation(cands):
        """(sag, cor, axi) picks by suffix, or None if any is missing/ambiguous."""
        out = []
        for ori in ("sag", "cor", "axi"):
            matches = [f for f in cands
                       if _strip_ext(os.path.basename(f)).endswith("_" + ori)]
            if len(matches) != 1:
                return None
            out.append(matches[0])
        return out

    if len(candidates) == 3:
        return by_orientation(candidates) or sorted(candidates, key=os.path.basename)

    if len(candidates) < 3:
        raise FileNotFoundError(
            f"expected 3 stacks in {subject_dir}, found {len(candidates)}: "
            f"{[os.path.basename(c) for c in candidates]}"
        )

    ordered = by_orientation(candidates)
    if ordered is None:
        raise FileNotFoundError(
            f"{subject_dir} holds {len(candidates)} stacks "
            f"{[os.path.basename(c) for c in candidates]}: keep only the three to "
            f"reconstruct, or mark those three with _sag/_cor/_axi suffixes"
        )
    return ordered


def find_mask_for(stack_path, subject_dir):
    """Find the existing mask_<core>.nii[.gz] for a raw stack (used for validation)."""
    core = os.path.basename(_strip_ext(stack_path))
    for f in (glob.glob(os.path.join(subject_dir, "*.nii"))
              + glob.glob(os.path.join(subject_dir, "*.nii.gz"))):
        b = os.path.basename(f)
        if b.startswith("._"):
            continue
        if "mask" in b.lower() and core in b:
            return f
    return None


def _nii_arrays(img_nii):
    """Replicate load_nii_volume(): return (vol transposed to (D,H,W), pixdim, affine)."""
    vol = img_nii.get_fdata().astype(np.float32)
    while vol.ndim > 3:
        vol = vol.squeeze(-1)
    vol = vol.transpose(2, 1, 0)
    resolutions = np.asarray(img_nii.header["pixdim"][1:4])
    affine = img_nii.affine
    if np.any(np.isnan(affine)):
        affine = img_nii.get_qform()
    return vol, resolutions, affine


def stack_from_nii(img_nii, mask_arr, device):
    """Build a nesvor Stack from an in-memory nii image + mask array (nii layout, or None).

    Mirrors load_stack() exactly.
    """
    vol, resolutions, affine = _nii_arrays(img_nii)
    slices_t = torch.tensor(vol, device=device)
    if mask_arr is None:
        mask_t = torch.ones_like(slices_t, dtype=torch.bool)
    else:
        m = mask_arr.astype(np.float32)
        while m.ndim > 3:
            m = m.squeeze(-1)
        m = m.transpose(2, 1, 0)
        mask_t = torch.tensor(m > 0, device=device, dtype=torch.bool)
    slices_t, mask_t, transformation = affine2transformation(
        slices_t, mask_t, resolutions, affine
    )
    return Stack(
        slices=slices_t.unsqueeze(1),
        mask=mask_t.unsqueeze(1),
        transformation=transformation,
        resolution_x=resolutions[0],
        resolution_y=resolutions[1],
        thickness=resolutions[2],
        gap=resolutions[2],
    )


def _volume_to_nii(vol, masked):
    """Convert a nesvor Volume to an in-memory nii image. Mirrors Image.save/save_nii_volume."""
    affine = transformation2affine(
        vol.image, vol.transformation,
        float(vol.resolution_x), float(vol.resolution_y), float(vol.resolution_z),
    )
    data = vol.image * vol.mask.to(vol.image.dtype) if masked else vol.image
    data = data.detach().cpu().numpy()
    while data.ndim > 3:
        data = data.squeeze(1) if data.shape[1] == 1 else data.squeeze(-1)
    data = data.transpose(2, 1, 0)
    if isinstance(affine, torch.Tensor):
        affine = affine.detach().cpu().numpy()
    img = nib.nifti1.Nifti1Image(data, affine)
    img.header.set_xyzt_units(2)
    img.header.set_qform(affine, code="aligned")
    img.header.set_sform(affine, code="scanner")
    return img


def stack_to_nii(stack, masked=True):
    """Convert a nesvor Stack back to an in-memory nii image (via get_volume)."""
    return _volume_to_nii(stack.get_volume(copy=False), masked=masked)


def _segment_to_mask_nii(raw_nii, device):
    """Segment a raw stack in memory and return its brain mask as a nii image (raw grid)."""
    stack = stack_from_nii(raw_nii, None, device)
    stack = brain_segmentation(
        [stack], device, SEG_DEFAULTS["batch_size_seg"],
        not SEG_DEFAULTS["no_augmentation_seg"], SEG_DEFAULTS["dilation_radius_seg"],
        SEG_DEFAULTS["threshold_small_seg"],
    )[0]
    return _volume_to_nii(stack.get_mask_volume(), masked=False)


def mask_and_reorient(raw_path, device, mask_nii=None):
    """Steps 1-2 for one stack: brain mask, then reorient to right-handed nii space.

    Returns (re_nii, mask_nii). If mask_nii is provided it is used as-is (skips
    segmentation); otherwise the mask comes from in-memory MONAIfbs segmentation.
    """
    raw_nii = nib.load(raw_path)

    # 1. MASK (raw grid)
    if mask_nii is None:
        mask_nii = _segment_to_mask_nii(raw_nii, device)

    # 2. REORIENT to right-handed (nii space), matching reorder_masks_fast.
    raw_affine = raw_nii.affine
    raw_data = np.asarray(raw_nii.dataobj)
    if np.linalg.det(raw_affine) < 0:
        re_nii = nib.Nifti1Image(raw_data[::-1, :, :], mask_nii.affine)
    else:
        re_nii = nib.Nifti1Image(raw_data, raw_affine)

    return re_nii, mask_nii


def correct_bias_field(in_niis, mask_niis, device):
    """N4 over all of a subject's stacks in one call, as `nesvor correct-bias-field`.

    apply_bias_field.py ran exactly one such call per subject; this restores that call
    shape. n4_bias_field_correction fits each stack separately either way, but because
    the fitted field absorbs each stack's overall level, its output is intrinsically
    level-matched across stacks -- which is why running it LAST is what leaves a
    subject's three stacks on one intensity scale.

    Returns the bias-corrected, brain-masked stacks as nii images.
    """
    stacks = [stack_from_nii(in_nii, np.asarray(mask_nii.dataobj), device)
              for in_nii, mask_nii in zip(in_niis, mask_niis)]
    corrected = n4_bias_field_correction(stacks, N4_DEFAULTS)
    return [stack_to_nii(stack, masked=True) for stack in corrected]


def normalize_stack(bias_nii, mask_nii):
    """Step 4 for one stack: normalize_by_second_mode, matching normalize_stacks_fast.

    Per stack by construction, so it does not preserve the cross-stack intensity match
    that correct_bias_field produces. See that function's note.
    """
    norm_arr = normalize_by_second_mode(
        bias_nii.get_fdata(), np.asarray(mask_nii.dataobj), verbose=False
    )
    return nib.Nifti1Image(norm_arr, bias_nii.affine)


# The two orders the v1 disk chain used, and what each produced:
#
#   norm_bias   reorient -> normalize_by_second_mode -> correct-bias-field
#               = the *_norm -> *_bias_field files apply_bias_field.py wrote, which the
#                 v1 nesvor/svort baselines read. N4 runs last, so the three stacks come
#                 out on one intensity scale (mean-of-mask spread max/min ~1.05).
#
#   bias_norm   reorient -> correct-bias-field -> normalize_by_second_mode
#               = the *_re_bias_field -> *_re_bias_field_norm files standerdize_stack
#                 reads, i.e. what the cSVR network was trained on. The per-stack
#                 normalization runs last and undoes the level match, taking the spread
#                 to ~1.44 (up to 1.91).
#
# Both are real v1 chains for their own consumer; they are not interchangeable. Handing
# bias_norm output to nesvor/svort was the v2 regression: that change alone sharply
# lowered NeSVoR's slice NCC, everything else held fixed.
CHAINS = ("norm_bias", "bias_norm")


def preprocess_stacks_in_memory(raw_paths, device, mask_niis=None, chain="norm_bias"):
    """Run the full preprocessing chain for a subject's stacks, in memory.

    Returns (out_niis, mask_niis, stages), where out_niis is the final stage of the
    chosen chain and stages maps every stage name to its nii images, so callers can
    dump the intermediates. See CHAINS for what the two orders mean.
    """
    if chain not in CHAINS:
        raise ValueError(f"chain must be one of {CHAINS}, got {chain!r}")
    if mask_niis is None:
        mask_niis = [None] * len(raw_paths)

    re_niis, out_masks = [], []
    for raw_path, mask_nii in zip(raw_paths, mask_niis):
        re_nii, mask_nii = mask_and_reorient(raw_path, device, mask_nii=mask_nii)
        re_niis.append(re_nii)
        out_masks.append(mask_nii)

    if chain == "norm_bias":
        norm_niis = [normalize_stack(r, m) for r, m in zip(re_niis, out_masks)]
        bias_niis = correct_bias_field(norm_niis, out_masks, device)
        out_niis = bias_niis
    else:
        bias_niis = correct_bias_field(re_niis, out_masks, device)
        norm_niis = [normalize_stack(b, m) for b, m in zip(bias_niis, out_masks)]
        out_niis = norm_niis

    return out_niis, out_masks, {"norm": norm_niis, "bias_field": bias_niis}


def preprocess_stack_in_memory(raw_path, device, mask_nii=None, chain="norm_bias"):
    """Single-stack convenience wrapper. Returns (out_nii, mask_nii, stages).

    Prefer preprocess_stacks_in_memory for a whole subject: correct-bias-field then
    runs as one call over all the stacks, the way apply_bias_field.py invoked it.
    """
    out_niis, mask_niis, stages = preprocess_stacks_in_memory(
        [raw_path], device, mask_niis=[mask_nii], chain=chain
    )
    return out_niis[0], mask_niis[0], {k: v[0] for k, v in stages.items()}


def dump_preprocessed(out_dir, raw_paths, mask_niis, stages):
    """Write every stage of the chain to out_dir, for downstream use and for debugging.

    Per raw stack <core>.nii[.gz], one file per stage plus the mask:
        <core>_norm.nii.gz         second-mode normalized
        <core>_bias_field.nii.gz   correct-bias-field output
        mask_<core>.nii.gz         brain mask (raw grid)

    Which of the two is the chain's final output depends on the order it ran in (see
    CHAINS); both are written either way so they can be compared. Each glob sorts into
    the same stack order -- the mask_ prefix and the _norm/_bias_field suffixes are
    constant, so they don't disturb the relative order -- so a caller can pair stacks
    with masks using two sorted globs.

    Returns {stage: [paths]} including "mask".
    """
    os.makedirs(out_dir, exist_ok=True)
    written = {name: [] for name in list(stages) + ["mask"]}
    for i, raw_path in enumerate(raw_paths):
        core = _strip_ext(os.path.basename(raw_path))
        items = [(name, niis[i], f"{core}_{name}.nii.gz") for name, niis in stages.items()]
        items.append(("mask", mask_niis[i], f"mask_{core}.nii.gz"))
        for name, nii, filename in items:
            path = os.path.join(out_dir, filename)
            nib.save(nii, path)
            written[name].append(path)
    return written


def preprocess_subject_in_memory(subject_dir, device, use_existing_masks=False,
                                 chain="norm_bias", dump_dir=None):
    """Full in-memory preprocessing for a subject.

    Returns (img1, img2, img3, mask1, mask2, mask3, voxel_size1) -- the canonical
    arrays that standerdize_stack(bias_field_norm=True) would load from disk, ready to
    pass to standerdize_stack_from_arrays. Stacks come in find_raw_stacks order:
    (sag, cor, axi) when labeled, matching the sorted-by-filename order standerdize
    uses; otherwise sorted by filename, and the ordering MLP assigns the orientations.

    chain picks the step order (see CHAINS). The default, "norm_bias", is what the
    nesvor/svort baselines read, so all three methods run on the same stacks; its output
    keeps a subject's stacks intensity-matched. "bias_norm" reproduces the
    *_re_bias_field_norm files the cSVR network was trained on instead.

    dump_dir, if given, writes both stages plus the masks there for debugging.
    """
    raw_paths = find_raw_stacks(subject_dir)

    mask_niis = None
    if use_existing_masks:
        mask_niis = []
        for raw_path in raw_paths:
            mp = find_mask_for(raw_path, subject_dir)
            if mp is None:
                raise FileNotFoundError(f"no existing mask for {raw_path}")
            mask_niis.append(nib.load(mp))

    out_niis, out_masks, stages = preprocess_stacks_in_memory(
        raw_paths, device, mask_niis=mask_niis, chain=chain
    )
    if dump_dir:
        dump_preprocessed(dump_dir, raw_paths, out_masks, stages)

    imgs, masks, voxel_sizes = [], [], []
    for img_nii, mask_nii in zip(out_niis, out_masks):
        voxel_sizes.append(tuple(float(z) for z in img_nii.header.get_zooms()[:3]))
        imgs.append(nib.as_closest_canonical(img_nii).get_fdata())
        masks.append(nib.as_closest_canonical(mask_nii).get_fdata())

    # standerdize orders stacks by sorted *_re_bias_field_norm filename; the raw
    # sag/cor/axi share the same core prefixes, so (sag,cor,axi) sorted == the disk order.
    return imgs[0], imgs[1], imgs[2], masks[0], masks[1], masks[2], voxel_sizes[0]
