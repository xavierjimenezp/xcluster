# Galaxy cluster detection through semantic segmentation
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://GitHub.com/Naereen/StrapDown.js/graphs/commit-activity) [![PyPI license](https://img.shields.io/pypi/l/ansicolortags.svg)](https://pypi.python.org/pypi/ansicolortags/) [![PythonVersion](https://camo.githubusercontent.com/fcb8bcdc6921dd3533a1ed259cebefdacbc27f2148eab6af024f6d6458d5ec1f/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f707974686f6e2d332e36253230253743253230332e37253230253743253230332e38253230253743253230332e392d626c7565)](https://img.shields.io/badge/python-3.6%20%7C%203.7%20%7C%203.8%20%7C%203.9-blue)


|Version         |Date                          |
|----------------|-------------------------------|
|alpha|04/05/2021            |



# Disclaimer
xCluster is currentlty under early development and may contain bugs or instabilities. 

# Installation 
Clone or download the xCluster repository:

`git clone https://github.com/xavierjimenezp/xcluster/`

Create a new conda environment (not available yet):

`conda env create -f environment_xcluster.yml`

(Additional information relative to conda environments: [click here](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#)) 

Activate the environment:

`source activate xcluster`
or 
`conda activate xcluster`

> **Note**: Eventually, planck HFI full sky maps and MILCA full sky map will be added as downloadable files (if public).

Code is ready to run !

# Quickstart

Modify the `params.py` file to specify paths to *planck* HFI full sky maps and MILCA full sky map. Files should be named as following

`HFI_SkyMap_100-field-IQU_2048_R3.00_full.fits`

`HFI_SkyMap_143-field-IQU_2048_R3.00_full.fits`

`HFI_SkyMap_143-field-IQU_2048_R3.00_full.fits`

`HFI_SkyMap_353-psb-field-IQU_2048_R3.00_full.fits`

`HFI_SkyMap_545-field-Int_2048_R3.00_full.fits`

`HFI_SkyMap_857-field-Int_2048_R3.00_full.fits`

`milca_ymaps.fits`


The following commands take *planck* 6 frenquency maps and preprocess them. 
Sky maps are splited with HEALPIX nside=2 into 48 tiles of equal-sized area of 860 square degrees each. Tiles 10, 39, 42 and 26 are used for validation and tile 7 is kept for test. Remaining patches are left for training.

**Training**: Individual patches with size 1.83º x 1.83º  (64 x 64 pixels), and resolution 1.7º, are extracted from the sky maps. These patches contain at least one cluster from the cluster catalog selected in the params file under *dataset*. Multiple patches of the same cluster are extracted with random translations to get and input training set with shape (~ 90,000 x 64 x 64 x 6). For each patch, a segmentation image is created where, for each cluster, a circular mask with radius *disk_radius* is created. The output training set input is (~ 90,000 x 64 x 64 x 1).

**Validation**: Same as training. input shape(~ 10,000 x 64 x 64 x 6).

**Test**: Tile number 7 is entirely splitted into patches to cover the whole HEALPIX. Some patches do not contain any cluster. Segmentation images are created the same way as training and validation.

`python xcluster.py --make_directories True`

`python xcluster.py --nodes 4 --dataset True`

`python xcluster.py --dataset True`

`python xcluster.py --train True`

`python xcluster.py --predict True`

# Usage

The following section presents the different parameters that need to be filled in `params.py` as well as the different arguments that can be used in order to use xCluster.

## Parameters
