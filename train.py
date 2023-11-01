import os
import argparse
import yaml
import pickle
import torch

from virdo import VirdoModule

root_path = os.getcwd()
os.chdir(root_path)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--config",
    required=True,
    help="Path to config file"
)
parser.add_argument(
    "--gpu_id",
    default=-1,
    type=int
)
parser.add_argument(
    "--pretrain_path",
    default="logs",
    required=True
)
parser.add_argument(
    "--from_maintrained",
    default=None,
    type=str,
    help="Path to the pre-maintrained model. If None, start training from scratch",
)
opt = parser.parse_args()

with open(opt.config, 'r') as file:
    config_args = yaml.safe_load(file)
maintraining_args = config_args["maintraining"]

DEVICE = "cpu" if opt.gpu_id < 0 else "cuda"
if DEVICE == "cuda":
    torch.cuda.set_device(opt.gpu_id)

with open(config_args["dataset"]["data_save_path"], "rb") as f:
    data_dict = pickle.load(f)

torch.manual_seed(300)
torch.cuda.manual_seed(300)
torch.cuda.manual_seed_all(300)

VM = VirdoModule(data_dict["train"], network_specs=config_args['network_specs'], DEVICE=DEVICE)
VM.maintraining(opt.pretrain_path, maintraining_args)
