import os
import datetime
import argparse
import numpy as np
import pickle
import time, torch, importlib, math, sys
import numpy as np
import open3d as o3d

from utilities.train_util import data2pairs
from virdo import VirdoModule

root_path = os.getcwd()
os.chdir(root_path)
data_root_path = os.path.join(root_path, "Data")


parser = argparse.ArgumentParser()
parser.add_argument("--name", default="pretrain")
parser.add_argument("--gpu_id", default=-1, type=int)
parser.add_argument("--pretrain_path", default="logs")
parser.add_argument("--epoch", default=40001, type=int)
parser.add_argument(
    "--from_pretrained",
    default=None,
    type=str,
    help="Path to the pretrained model. If None, start training from scratch",
)
opt = parser.parse_args()
#


DEVICE = "cpu" if opt.gpu_id < 0 else "cuda"
if DEVICE == "cuda":
    torch.cuda.set_device(opt.gpu_id)


with open("data/virdo_simul_dataset.pickle", "rb") as f:
    data_dict = pickle.load(f)

data_index = data2pairs(data_dict["train"])


torch.manual_seed(300)
torch.cuda.manual_seed(300)
torch.cuda.manual_seed_all(300)

VM = VirdoModule(data_dict["train"])

model_directory = os.path.join(root_path, "logs", opt.name)
checkpoints_dir = os.path.join(model_directory, "checkpoints")

VM.maintraining(opt.pretrain_path, checkpoints_dir, 4001 )
