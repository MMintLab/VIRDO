# VIRDO
<a href="https://github.com/psf/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
[![CodeQL](https://github.com/MMintLab/VIRDO/actions/workflows/codeql-analysis.yml/badge.svg)](https://github.com/MMintLab/VIRDO/actions/workflows/codeql-analysis.yml)

This is a github repository of a [Visio-tactile Implicit Representations of Deformable Objects](https://arxiv.org/abs/2202.00868) (ICRA 2022).
Codes are based on [siren](https://github.com/vsitzmann/siren) and [pointnet](https://github.com/charlesq34/pointnet) repositories.

## Quick Start
**Reconstruction & latent space composition** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/15T89qRkZuOFfcHYEa24mlZUuFeni1QqI#scrollTo=izxG2oGAriLK&uniqifier=1)

**inference using partial pointcloud** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1ZY5LVsKR8qN99C0EeyyqVnsWWg4v6vPN#scrollTo=f53ea8fc)

## Step 0: Set up the environment
```angular2html
conda create -n virdo python=3.8
conda activate virdo
conda install pytorch==1.7.0 torchvision==0.8.0 torchaudio==0.7.0 cudatoolkit=11.0 -c pytorch
conda install -c fvcore -c iopath -c conda-forge fvcore iopath
conda install -c bottler nvidiacub
conda install pytorch3d=0.5.0 -c pytorch3d
pip install open3d==0.14.1
pip install plyfile==0.7.4
pip install scikit-image
```

## Step 1: Download pretrained model and dataset
Make sure to install wget ```$ apt-get install wget``` and unzip ```$ apt-get install unzip```

```angular2html
source download.sh
download_dataset
download_pretrained
```
### (Optionally) Manual
Alternatively, you can manually download the datasets and pretrained models from [here](https://www.dropbox.com/sh/4gnme6f0srhnk23/AAABlA6n8cfyo-GsaiDEqLoba?dl=0). Then put the files as below:
```
── VIRDO
│   ├── data
│   │   │── virdo_simul_dataset.pickle
│   ├── pretrained_model
│   │   │── force_final.pth
│   │   │── object_final.pth
│   │   │── deform_final.pth

```

## Step 2: Pretrain nominal shapes
```
python pretrain.py --name <log name> --gpu_id 0
```
If you want to check the result of your pretrained model, 
```
python pretrain.py --checkpoints_dir <dir> --gpu_id 0
```

then you will see the nominal reconstructions in /output/ directory.


## Step 3: Train entire dataset
```angular2html
python train.py --pretrain_path <path>  --checkpoints_dir <dir>
```
