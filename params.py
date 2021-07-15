#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Xavier Jimenez
"""

import os

#------------------------------------------------------------------#
# # # # # Parameters # # # # #
#------------------------------------------------------------------#

## PATHS
path = os.getcwd() + '/' 
planck_path = '/n17data/jimenez/PLANCK/' #path to planck HFI full sky maps
milca_path = '/n17data/jimenez/PLANCK/' #path to milca full sky map
healpix_path = path + 'healpix/figures/' #will be created

## GLOBAL
dataset = 'planck_z'
bands = ['100GHz','143GHz','217GHz','353GHz']#['100GHz','143GHz','217GHz','353GHz','545GHz','857GHz', 'y-map']
merge_daily_output_directory = True
disk_radius = 2.5 #in arcmin, if None distribution between 0 and 15 arcmin
cold_cores = True
npix = 32

## CATALOGS
plot_catalogs = False

## DATASET
loops = 100 #times dataset will be added again to training set with random variations
label_only = False
fit_up_to_mode = False #if False, MAD will be used instead
range_compression = True
plot_dataset = True

## TRAIN
regions = [5]
n_labels = 1
epochs = 30 #60
batch = 20
lr = 1e-2
patience = 12 #30
model = 'unet'
loss = 'tversky_loss'
delta = 0.3
gamma =0.75
optimizer = 'sgd'

## PREDICT
plot_prediction = True
plot_individual_patchs = True
