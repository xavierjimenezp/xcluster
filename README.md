# Galaxy cluster detection through semantic segmentation
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://GitHub.com/Naereen/StrapDown.js/graphs/commit-activity) [![PyPI license](https://img.shields.io/pypi/l/ansicolortags.svg)](https://pypi.python.org/pypi/ansicolortags/) [![PythonVersion](https://camo.githubusercontent.com/fcb8bcdc6921dd3533a1ed259cebefdacbc27f2148eab6af024f6d6458d5ec1f/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f707974686f6e2d332e36253230253743253230332e37253230253743253230332e38253230253743253230332e392d626c7565)](https://img.shields.io/badge/python-3.6%20%7C%203.7%20%7C%203.8%20%7C%203.9-blue)

|Version         |Date                          |
|----------------|-------------------------------|
|alpha|06/05/2021            |

![enter image description here](https://github.com/xavierjimenezp/xcluster/blob/main/Figures/mask.png?raw=true)

# Disclaimer
xCluster is currentlty under early development and may contain bugs or instabilities. 

# Overview

xCluster is an algorithm that uses full sky-maps and galaxy cluster catalogs to train a U-Net-like convolutional neural network (CNN) for semantic segmentation in order to detect unkown clusters. It creates and preprocess the input/output datasets for the CNN, trains it and detects potential galaxy clusters in the test set. Finally, detections are cross-matched with known clusters and reliability is assessed by cross-matching with known IR sources.

## Dataset creation & preprocessing

Sky maps are splited with HEALPIX nside=2 into 48 tiles of equal-sized area of 860 square degrees each. Tiles 10, 26, 39, and 42 are used for validation while tile 7 is kept for test. Remaining tiles are left for training.

### Input & output dataset

**Training**: Individual patches with size 1.83º x 1.83º  (64 x 64 pixels), and resolution 1.7º, are extracted from the sky maps. These patches contain at least one cluster from the cluster catalog selected in the params file under *dataset*. Multiple patches of the same cluster are extracted with random translations to get and input training set with shape (~ 90,000 x 64 x 64 x ns), where ns is the number of sky-maps used. For each patch, a segmentation image is created where, for each cluster, a circular mask with radius *disk_radius* is created. The output training set input is (~ 90,000 x 64 x 64 x 1).

**Validation**: Same as training. input shape(~ 10,000 x 64 x 64 x ns).

**Test**: Tile number 7 is entirely splitted into patches to cover the whole HEALPIX. Some patches do not contain any cluster. Segmentation images are created the same way as training and validation.

### Preprocessing

Input data is preprocessed individually for training, validation and test individually. The shapes of the pixel distributions of the Planck HFI frequency maps are **very non Gaussian**, preventing a simple normalisation of the maps to their means and their standard
deviations. Therefore, data is standardized using the **median absolute deviation** (**MAD**), which is a robust measure of the variability of a univariate sample of quantitative data. Then, we apply **range compression** (an arcsinh function) to suppress the high amplitude values.

### CNN Segmentation

#### Architecture

U-net is a convolutional neural network with an encoder-decoder architecture which enables end-to-end feature extraction and pixel classification. It was originally proposed for the semantic segmentation of medical images in 2015. Since then, many variants based on the U-Net architecture have been proposed. xCluster can use any of the following architectures, which need to be specified in the parameter file.

|**Model Name**|**Reference**|
|--------------|-------------|
|[![Generic badge](https://img.shields.io/badge/U_Net-Up-green.svg)](https://shields.io/)| [Ronneberger et al. (2015)](https://link.springer.com/chapter/10.1007/978-3-319-24574-4_28) |
|[![Generic badge](https://img.shields.io/badge/V_Net-Down-red.svg)](https://shields.io/)| [Milletari et al. (2016)](https://arxiv.org/abs/1606.04797)|
| [![Generic badge](https://img.shields.io/badge/U_Net++-Up-green.svg)](https://shields.io/)| [Zhou et al. (2018)](https://link.springer.com/chapter/10.1007/978-3-030-00889-5_1) |
|[![Generic badge](https://img.shields.io/badge/Attention_U_Net-Up-green.svg)](https://shields.io/)| [Oktay et al. (2018)](https://arxiv.org/abs/1804.03999) |
|[![Generic badge](https://img.shields.io/badge/ResUNet_a-Up-green.svg)](https://shields.io/)| [Diakogiannis et al. (2020)](https://doi.org/10.1016/j.isprsjprs.2020.01.013) |
|[![Generic badge](https://img.shields.io/badge/U^2_Net-Up-green.svg)](https://shields.io/)| [Qin et al. (2020)](https://arxiv.org/abs/2005.09007) |
|[![Generic badge](https://img.shields.io/badge/U_Net_3+-Up-green.svg)](https://shields.io/)| [Huang et al. (2020)](https://arxiv.org/abs/2004.08790) |

Tensorflow implementations provided by python package [**`keras_unet_collection`**](https://github.com/yingkaisha/keras-unet-collection).

#### Loss function

Once a model architecture is selected, optimisation of model parameters is based on minimisation of the loss function during training. The most widely used for classification is binary cross-entropy. However, it is not suited for segmentation of highly imbalanced datasets. Since then, other loss functions have been deployed in order to adress this problem. xCluster can use any of the following loss functions, which need to be specified in the parameter file.

|**Loss Function**|**Reference**|
|--------------------|-------------|
| [![Generic badge](https://img.shields.io/badge/Binary_crossentropy-Up-green.svg)](https://shields.io/) |  |
|[![Generic badge](https://img.shields.io/badge/Focal_loss-Up-green.svg)](https://shields.io/)|  |
|[![Generic badge](https://img.shields.io/badge/Dice_loss-Up-green.svg)](https://shields.io/)| [Sudre et al. (2017)](https://link.springer.com/chapter/10.1007/978-3-319-67558-9_28) |
|[![Generic badge](https://img.shields.io/badge/Focal_Dice_loss-Up-green.svg)](https://shields.io/)| [Sudre et al. (2017)](https://link.springer.com/chapter/10.1007/978-3-319-67558-9_28)  |
|[![Generic badge](https://img.shields.io/badge/Tversky_loss-Up-green.svg)](https://shields.io/)| [Hashemi et al. (2018)](https://ieeexplore.ieee.org/abstract/document/8573779) |
|[![Generic badge](https://img.shields.io/badge/Focal_Tversky_loss-Up-green.svg)](https://shields.io/)| [Abraham et al. (2019)](https://ieeexplore.ieee.org/abstract/document/8759329) |
|[![Generic badge](https://img.shields.io/badge/Cosine_Tversky_loss-Up-green.svg)](https://shields.io/)| [Michael Yeung et al. (2021)](https://arxiv.org/abs/2102.04525) |
|[![Generic badge](https://img.shields.io/badge/Combo_loss-Up-green.svg)](https://shields.io/)| [Saeid Asgari Taghanaki et al. (2018)](https://arxiv.org/abs/1805.02798) |
|[![Generic badge](https://img.shields.io/badge/Mixed_focal_loss-Up-green.svg)](https://shields.io/)| [Michael Yeung et al. (2021)](https://arxiv.org/abs/2102.04525) |

#### Training

Model is trained using the following parameters, which are specified in the parameter file: epochs, batch size, patience, optimization algorithm and learning rate.

#### Prediction

Once the model is trained, the following detection criterium are used on prediction outputs.
- A pixel with value above 0.9 with at least 3 adjacent pixels is considered a detection.
- Center is computed as the barycenter of all adjacent pixels above 0.9.
- Each detection has a maximum radius of 15 arcmin.

Cluster catalogs are cross matched with detection centers to establish known/unkown detections, as well as the amount of possible detections for each patch. Since patches can overlap one another, duplicate detections are accounted for. Finally, unkown detection centers are cross matched with known IR sources to assess reliability.


# Installation 
Clone or download the xCluster repository:

`git clone https://github.com/xavierjimenezp/xcluster/`

Create a new conda environment:
> **Note**: This feature is not available yet, environment will be implemented later on.

`conda env create -f environment_xcluster.yml`

(Additional information relative to conda environments: [click here](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#)) 

Activate the environment:

`source activate xcluster`
or 
`conda activate xcluster`

> **Note**: Eventually, planck full sky maps will be added as downloadable files (if public).

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

`python xcluster.py --make_directories True`

`python xcluster.py --nodes 4 --dataset True`

`python xcluster.py --dataset True`

`python xcluster.py --train True`

`python xcluster.py --predict True`

> **Note**: Eventually, a colab or jupyter notebook file will be added for a simple prediction example.


# Usage

The following section presents the different parameters that need to be filled in `params.py` as well as the different arguments that can be used in order to use xCluster.

## Arguments

The `xcluster.py` python file takes the following arguments. Multiple arguments can be used (at least one bool type argument needs to be specified), in which case they will be executed in the following order.

 - `--nodes or -n`: `(optinal, int)` number of cores to be used. Defaults to 1.
> **Note**: only `--dataset` can use more than one core.
 - `--input or -i`: `(optinal, str)` input file name for parameters file. Defaults to params.
 - `--make_directories or -m`: `(optional, bool)`: if True, will create/modify/clean all needed directories. Check parameters file for more options. Defaults to False.
 - `--dataset or -d`: `(optional, bool)`: if True, will preprocess sky-maps into Input and Output '.npz' files  (-> datasets/*dataset*/)  for CNN segmentation.  Check parameters file for more options.  Defaults to False.
 - `--train or -t`: `(optional, bool)`: if True, 
 - `--predict or -p`: `(optional, bool)`: if True, 

## Parameters

List of parameters that need to be specified in the parameters input file. By default, xcluster will look for `params.py` if no file is specified under `--input`.

### PATHS
Paths that need to be specified for all xcluster steps. 

- `planck_path` `(mandatory, str)`:  path to planck HFI full sky maps. Check **Quickstart** section for file names.
- `milca_path` `(mandatory, str)`: path to MILCA full sky map. Check **Quickstart** section for file name.

> **Note**: Eventually, more sky maps will be added and more parameters will be added.

### GLOBAL
Parameters that need to be specified for all xcluster steps. 

- `dataset` `(mandatory, str)`: cluster catalog that will be used to extract patches from full sky-maps.  Options are: 'planck_z', 'planck_no-z', 'MCXC', 'RM30', 'RM50'. Check V. Bonjean 2018 for catalog definitions.
- `bands` `(mandatory, list of str)`: list of sky maps that will be used (e.g ['100GHz','143GHz','217GHz','353GHz','545GHz','857GHz', 'y-map'] contains all possible sky maps). 
- `merge_daily_output_directory` `(mandatory, bool)`: if True, will merge daily output directory with the most recent directory. if False, will create a new directory at output/*dataset*/yyyy-mm-dd.
- `disk_radius`  `(mandatory, float)` = disk radius (in arcmin) used circular segmentation output files. If None, a distribution between 0 and 15 arcmin will be used instead.

### DATASET

Parameters that need to be specified when using  `--dataset True`.

- `loops` `(mandatory, int)`: number of times the dataset containing patches with at least one cluster within will be added again to training set with random variations (translations).
> **Note**: Eventually, random rotations will be added as well.

- `fit_up_to_mode` `(mandatory, bool)`: if False, MAD is not used and the FWHM of a gaussian fit up to mode is used instead (see V. Bonjean 2018 for more details).
- `range_compression` `(mandatory, bool)`: if True, applies range compression for preprocess.
- `plot_dataset` `(mandatory, bool)`: if True, saves dataset plots in /output/yyyy-mm-dd/figures/.

### TRAIN

Parameters that need to be specified when using  `--train True`.

- `epochs` `(mandatory, int)`:  epochs number.
- `batch` `(mandatory, int)`: batch size.
- `lr` `(mandatory, float)`: learning rate.
- `patience` `(mandatory, int)`: mimimum epochs number.
- `model` `(mandatory, str)`: architecture used.  Options are (from the **overview** list): 'unet', 'vnet', 'unet_plus', 'r2u_net', 'attn_net', 'resunet_a', 'u2net', 'unet_3plus'.
- `loss` `(mandatory, str)`: loss function used.  Options are (from the **overview** list): 'binary_crossentropy', 'focal_loss', 'dice_loss', 'focal_dice_loss', 'tversky_loss', 'focal_tversky_loss', 'cosine_tversky_loss', 'combo_loss', 'mixed_focal_loss'.
- `optimizer` `(mandatory, str)`: optimization algorithm. Either 'adam' or 'sgd'.

## PREDICT

Parameters that need to be specified when using  `--predict True`.

- `plot_prediction` `(mandatory, bool)`:  if True, save prediction plots in /output/yyyy-mm-dd/figures/.
- `plot_individual_patchs` `(mandatory, bool)`:  if True, save prediction plots in /output/yyyy-mm-dd/figures/.

# ACKNOWLEDGMENTS

This work has made use of the CANDIDE Cluster at the Institut d'Astrophysique de Paris and made possible by grants from the PNCG and the DIM-ACAV.

# CONTACT

Xavier Jiménez < xavier.jimenez@ens-lyon.fr > < xavierjimenezp@gmail.com >

# LICENSE

[MIT License](https://github.com/xavierjimenezp/xcluster/blob/main/LICENSE)
