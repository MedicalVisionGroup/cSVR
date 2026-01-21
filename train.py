import os
import sys
import yaml
import torch
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, Callback
from pytorch_lightning import loggers as pl_loggers
import wandb
import argparse
import models
import models.losses
import models.metrics
import models.optimizers
import datasets
import options
import pdb

torch.cuda.reset_peak_memory_stats()
torch.cuda.reset_accumulated_memory_stats()
torch.cuda.empty_cache()
torch.cuda.synchronize()

# --------------------------
# Utility functions
# --------------------------
def get_config(f):
    with open(f, "r") as stream:
        return yaml.load(stream, Loader=yaml.SafeLoader)

def print_model_parameters(model, name="Model"):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{name} Total parameters: {total_params:,}")
    print(f"{name} Trainable parameters: {trainable_params:,}")

class MinLossTracker(Callback):
    def __init__(self):
        self.min_loss = float('inf')

    def on_validation_end(self, trainer, pl_module):
        val_loss = trainer.callback_metrics.get("val_loss")
        if val_loss is not None:
            val_loss_value = val_loss.item()
            if val_loss_value < self.min_loss:
                self.min_loss = val_loss_value
            wandb.log({'min_val_loss': self.min_loss})

# --------------------------
# ADD Weights and Biases Account Here
# --------------------------

# wandb_logger = pl_loggers.WandbLogger(
#     project="",
#     entity="",
#     name=args.remarks,
#     config=vars(args),
#     log_model=False,
#     settings=wandb.Settings(console="off"),
# )


# --------------------------
# Load config
# --------------------------

parser = argparse.ArgumentParser()
parser = options.set_argparse_defs(parser)
parser = options.add_argparse_args(parser)

args = parser.parse_args()

# get the config path from whatever was already defined in options
cfg_file = getattr(args, "config", None)
if cfg_file is None:
    print("Please specify a --config YAML file")
    sys.exit(1)


cfg = get_config(cfg_file)

# override argparse values with YAML
for k, v in cfg.items():
    try:
        setattr(args, k, v)
        print(f"✓ Set {k} = {v}")
    except Exception as e:
        print(f"✗ Failed to set {k}: {e}")

args.remarks = f"{args.dataset}_{args.network}_{args.loss}_{args.rotations}_{args.translations}_{args.noise}_{args.bulk_rotations_plane}_{args.bulk_rotations_tr_plane}_lr{args.lr_start}_{args.remarks_add}"


print(args.remarks)
# --------------------------
# Setup
# --------------------------
args.default_root_dir = os.path.join('./checkpoints/', args.remarks)
os.makedirs(args.results_dir, exist_ok=True)
results_file = os.path.join(args.results_dir, f"{args.remarks}.txt")


wandb_logger.experiment.log_artifact = lambda *args, **kwargs: None
seed_everything(args.seed, workers=True)

loss_fn = models.losses.__dict__[args.loss]

# --------------------------
# Data
# --------------------------
train_data, valid_data, tests_data = datasets.__dict__[args.dataset](**vars(args))
train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=4, pin_memory=True)
valid_loader = DataLoader(valid_data, batch_size=args.valid_batch_size, shuffle=False, num_workers=4, pin_memory=True)
tests_loader = DataLoader(tests_data, batch_size=args.valid_batch_size, shuffle=False, num_workers=4, pin_memory=True)

# --------------------------
# Model & Optimizer
# --------------------------
network = models.__dict__[args.network](
    train_data.__numinput__(), train_data.__numclass__(),
    pretrained=args.pretrained,
    padding=args.padding,
    padding_mode=args.padding_mode,
    drop=args.drop_rate,
    skip=args.global_skip
)
optim = models.optimizers.__dict__[args.optim](
    network.parameters(),
    lr=args.lr_start,
    momentum=args.momentum,
    weight_decay=args.decay,
    nesterov=args.nesterov
)

# --------------------------
# Metrics
# --------------------------

train_metrics = [models.metrics.__dict__[m]() for m in args.train_metrics]
valid_metrics = [models.metrics.__dict__[m]() for m in args.valid_metrics]
tests_metrics = [models.metrics.__dict__[m]() for m in args.tests_metrics]

# --------------------------
# Callbacks & Trainer
# --------------------------
callbacks = [
    ModelCheckpoint(monitor=args.monitor, mode=args.monitor_mode, dirpath=args.default_root_dir, filename='best', save_last=True),
    models.ProgressBar(refresh_rate=5),
    MinLossTracker()
]

trainer = Trainer.from_argparse_args(
    args,
    callbacks=callbacks,
    logger=wandb_logger,
    gradient_clip_val=0.5,
    gradient_clip_algorithm='norm',
    precision=32
)

# --------------------------
# Trainee
# --------------------------
loader = models.__dict__[args.trainee].load_from_checkpoint if args.load != '' else models.__dict__[args.trainee]
checkpt = os.path.join(args.default_root_dir, args.load) if args.load != '' else None

trainee = loader(
    checkpoint_path=checkpt,
    model=network,
    optimizer=optim,
    train_data=train_data,
    valid_data=valid_data,
    tests_data=tests_data,
    loss=loss_fn,
    train_metrics=train_metrics,
    valid_metrics=valid_metrics,
    tests_metrics=tests_metrics,
    schedule=args.schedule,
    monitor=args.monitor,
    strict=False
)

print_model_parameters(trainee, name="Trainee")
print(f"Train: {len(train_loader.dataset)} | Valid: {len(valid_loader.dataset)} | Tests: {len(tests_loader.dataset)}", file=sys.stderr)
print(args.train)

# --------------------------
# Run
# --------------------------
if args.train:
    trainer.fit(trainee, train_loader, valid_loader)
if args.validate:
    trainer.validate(trainee, dataloaders=valid_loader, ckpt_path=checkpt, verbose=False)
if args.test:
    trainer.test(trainee, dataloaders=tests_loader, ckpt_path=checkpt, verbose=False)

wandb.finish()
