Place the MONAIfbs fetal-brain segmentation checkpoint here:

    checkpoint_dynUnet_DiceXent.pt

It is used by `--preprocess` to mask the input stacks. If the file is missing it is downloaded
automatically from Zenodo on first use. See the "Checkpoints" section of
the top-level README.md.
