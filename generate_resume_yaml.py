"""Generate a resume YAML config from an existing training YAML.

Usage:
    python generate_resume_yaml.py experiments/april6_....yaml
    python generate_resume_yaml.py experiments/april6_....yaml --output experiments/april6_resume.yaml
"""
import argparse
import yaml
import os
import sys


def generate_resume_yaml(input_path, output_path=None):
    with open(input_path, 'r') as f:
        cfg = yaml.safe_load(f)

    # Set load to last.ckpt for resume
    cfg['load'] = 'last.ckpt'

    # Build the checkpoint dir to verify last.ckpt exists
    remarks = (
        f"{cfg.get('dataset', '')}_{cfg.get('network', '')}_"
        f"{cfg.get('loss', '')}_{cfg.get('rotations', '')}_"
        f"{cfg.get('translations', '')}_{cfg.get('noise', '')}_"
        f"{cfg.get('bulk_rotations_plane', '')}_{cfg.get('bulk_rotations_tr_plane', '')}_"
        f"lr{cfg.get('lr_start', '')}_{cfg.get('remarks_add', '')}"
    )
    ckpt_dir = os.path.join('./checkpoints/', remarks)
    last_ckpt = os.path.join(ckpt_dir, 'last.ckpt')

    if os.path.isfile(last_ckpt):
        print(f"✓ Found checkpoint: {last_ckpt}")
    else:
        print(f"⚠ Checkpoint not found yet: {last_ckpt}")
        print("  (Resume will work once training creates the first checkpoint)")

    # Generate output path
    if output_path is None:
        base = os.path.splitext(input_path)[0]
        output_path = f"{base}_resume.yaml"

    with open(output_path, 'w') as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)

    print(f"✓ Written resume config to: {output_path}")
    return output_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate resume YAML from training config')
    parser.add_argument('input', help='Path to original training YAML')
    parser.add_argument('--output', '-o', default=None, help='Output path (default: <input>_resume.yaml)')
    args = parser.parse_args()
    generate_resume_yaml(args.input, args.output)
