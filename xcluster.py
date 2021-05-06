#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Xavier Jimenez
Main script
"""

import argparse
from functions import GenerateFiles, MakeData, CNNSegmentation
import importlib
import warnings
import logging
cs_logger = logging.getLogger('cutsky')
cs_logger.setLevel(logging.WARNING)
cs_logger.propagate = False
hpproj_logger = logging.getLogger('hpproj')
hpproj_logger.setLevel(logging.WARNING)
mpl_logger = logging.getLogger('matplotlib')
mpl_logger.setLevel(logging.WARNING)

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
parser.add_argument("-i", "--input", required=False, type=str)


args = parser.parse_args()

if args.input is None:
    import params as p
    warnings.simplefilter("always")
    warnings.warn("No parameter file was given, 'params.py' will be used")
else:
    p = importlib.import_module(args.input)

if args.nodes is None:
        args.nodes = 1

GenFiles = GenerateFiles(dataset = p.dataset, bands=p.bands, output_path = p.path)
MData = MakeData(dataset = p.dataset, bands=p.bands, planck_path=p.planck_path, milca_path=p.milca_path, disk_radius= p.disk_radius, output_path = p.path)
CNN = CNNSegmentation(model = p.model, dataset = p.dataset, bands=p.bands, planck_path=p.planck_path, milca_path=p.milca_path, epochs=p.epochs, batch=p.batch, 
                       lr=p.lr, patience=p.patience, loss=p.loss, optimizer=p.optimizer, disk_radius=p.disk_radius, output_path = p.path)

if args.make_directories == True:

    GenFiles.clean_temp_directories()
    GenFiles.make_directories()
    GenFiles.make_directories(output = True, replace=p.merge_daily_output_directory)

if args.cluster_catalogs == True:
    planck_z, planck_no_z, MCXC_no_planck, RedMaPPer_no_planck = MData.create_catalogs(plot=p.plot_catalogs)
    # MData.plot_psz2_clusters(planck_path, milca_path, healpix_path)
    # GenFiles.remove_files_from_directory(healpix_path + 'PSZ2/')

if args.dataset == True:
    MData.train_data_generator(loops=p.loops, n_jobs=args.nodes, plot=False)
    MData.preprocess(leastsq=p.fit_up_to_mode, range_comp=p.range_compression, plot=p.plot_dataset)

if args.train == True:
    CNN.train_model()

if args.predict == True:
    CNN.evaluate_prediction(plot=p.plot_prediction, plot_patch = p.plot_individual_patchs)