# VIRDO: Visio-tactile Implicit Representations of Deformable Objects
This is a github repository of a [project](https://arxiv.org/abs/2202.00868) (ICRA 2022).
Codes are based on [siren](https://github.com/vsitzmann/siren) and [pointnet](https://github.com/charlesq34/pointnet) repositories.


## 1. Quick Start
### Reconstruction & Latent Space Composition: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/15T89qRkZuOFfcHYEa24mlZUuFeni1QqI#scrollTo=izxG2oGAriLK&uniqifier=1)

### Inference Using Partial Pointcloud: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1ZY5LVsKR8qN99C0EeyyqVnsWWg4v6vPN#scrollTo=f53ea8fc)

## 2. Requirements
* open3d >=0.14.1
* pytorch >=1.10.0+cu111
* plyfile >=0.7.4


## 3. Preparation
Datasets and pretrained models can be downloaded from [here](https://www.dropbox.com/sh/4gnme6f0srhnk23/AAABlA6n8cfyo-GsaiDEqLoba?dl=0). Then put the  files as below:

```
── VIRDO
│   ├── data
│   │   │── virdo_simul_dataset.pickle
│   ├── pretrained_model
│   │   │── force_final.pth
│   │   │── object_final.pth
│   │   │── deform_final.pth

```
