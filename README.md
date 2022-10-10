# VIRDO

<<<<<<< HEAD
## Quick Start
**Reconstruction & latent space composition** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/15T89qRkZuOFfcHYEa24mlZUuFeni1QqI#scrollTo=izxG2oGAriLK&uniqifier=1)

**inference using partial pointcloud** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1ZY5LVsKR8qN99C0EeyyqVnsWWg4v6vPN#scrollTo=f53ea8fc)

=======
<<<<<<< HEAD
>>>>>>> no lfs
## Step 0: Set up the environment
```angular2html
conda create -n virdo python=3.8
conda activate virdo
<<<<<<< HEAD
=======
```
## Colab Examples
### Reconstruction & Latent Space Composition

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/15T89qRkZuOFfcHYEa24mlZUuFeni1QqI#scrollTo=izxG2oGAriLK&uniqifier=1)

### Inference Using Partial Pointcloud
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1ZY5LVsKR8qN99C0EeyyqVnsWWg4v6vPN#scrollTo=f53ea8fc)


## Preparation
Datasets and pretrained models can be downloaded from [here](https://www.dropbox.com/sh/4gnme6f0srhnk23/AAABlA6n8cfyo-GsaiDEqLoba?dl=0). Then put the  files as below:
```
>>>>>>> no lfs
conda install pytorch==1.7.0 torchvision==0.8.0 torchaudio==0.7.0 cudatoolkit=11.0 -c pytorch
conda install pytorch3d=0.5.0 -c pytorch3d
pip install open3d==0.14.1
pip install plyfile==0.7.4
pip install scikit-image
```

<<<<<<< HEAD
## Step 1: Download pretrained model and dataset
Make sure to install wget ```$ apt-get install wget``` and unzip ```$ apt-get install unzip```
=======

## Step 1: Download pretrained model and dataset
Makesure to install curl ```$ apt-get install curl``` and unzip ```$ apt-get install unzip```
>>>>>>> no lfs
```angular2html
source download.sh
download_dataset
download_pretrained
```
<<<<<<< HEAD
=======
### (Optionally) Manual
>>>>>>> no lfs
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
<<<<<<< HEAD
```
If you want to check the result of your pretrained model, 
```
python pretrain.py --checkpoints_dir <dir> --gpu_id 0
```
then you will see the nominal reconstructions in /output/ directory.


## Step 3: Train entire dataset
```angular2html
python train.py --pretrain_path <path>  --checkpoints_dir <dir>
=======
>>>>>>> no lfs
```