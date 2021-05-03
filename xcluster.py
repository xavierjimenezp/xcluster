#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Xavier Jimenez
"""

import argparse
from functions import *

#------------------------------------------------------------------#
# # # # # Script # # # # #
#------------------------------------------------------------------#

parser = argparse.ArgumentParser()
parser.add_argument("-n", "--nodes", required=False, type=int, nargs="?", const=1)
parser.add_argument("-m", "--make_directories", required=False, type=bool, nargs="?", const=False)
parser.add_argument("-c", "--cluster_catalogs", required=False, type=bool, nargs="?", const=False)
parser.add_argument("-d", "--dataset", required=False, type=bool, nargs="?", const=False)
parser.add_argument("-t", "--train", required=False, type=bool, nargs="?", const=False)
parser.add_argument("-p", "--predict", required=False, type=bool, nargs="?", const=False)

args = parser.parse_args()

if args.nodes is None:
        args.nodes = 1

GenFiles = GenerateFiles(dataset = dataset, bands=bands, output_path = path)
MData = MakeData(dataset = dataset, bands=bands, planck_path=planck_path, milca_path=milca_path, disk_radius= disk_radius, output_path = path)
unet = CNNSegmentation(model = model, dataset = dataset, bands=bands, planck_path=planck_path, milca_path=milca_path, epochs=epochs, batch=batch, 
                       lr=lr, patience=patience, loss=loss, optimizer=optimizer, disk_radius=disk_radius, output_path = path)

if args.make_directories == True:

    GenFiles.clean_temp_directories()
    GenFiles.make_directories()
    GenFiles.make_directories(output = True, replace=merge_daily_output_directory)

if args.cluster_catalogs == True:
    planck_z, planck_no_z, MCXC_no_planck, RedMaPPer_no_planck = MData.create_catalogs(plot=plot_catalogs)
    # MData.plot_psz2_clusters(planck_path, milca_path, healpix_path)
    # GenFiles.remove_files_from_directory(healpix_path + 'PSZ2/')

if args.dataset == True:
    MData.train_data_generator(loops=loops, n_jobs=args.nodes, plot=False)
    MData.preprocess(leastsq=fit_up_to_mode, range_comp=range_compression, plot=plot_dataset)

if args.train == True:
    unet.train_model()

if args.predict == True:
    unet.evaluate_prediction(plot=plot_prediction, plot_patch = plot_individual_patchs)