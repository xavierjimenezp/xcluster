#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Xavier Jimenez
Main script
"""

import numpy as np
import argparse
from joblib import Parallel, delayed
import tqdm
from generate_files import GenerateFiles
from make_data import MakeData
from segmentation import CNNSegmentation

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
# # # # # Arguments # # # # #
#------------------------------------------------------------------#

parser = argparse.ArgumentParser()
parser.add_argument("-n", "--nodes", required=False, type=int, nargs="?", const=1)
parser.add_argument("-m", "--make_directories", required=False, type=bool, nargs="?", const=False)
parser.add_argument("-c", "--cluster_catalogs", required=False, type=bool, nargs="?", const=False)
parser.add_argument("-d", "--dataset", required=False, type=bool, nargs="?", const=False)
parser.add_argument("-t", "--train", required=False, type=bool, nargs="?", const=False)
parser.add_argument("-p", "--predict", required=False, type=bool, nargs="?", const=False)
parser.add_argument("-g", "--grid", required=False, type=str, nargs="?", const='train')
parser.add_argument("-v", "--plot_tversky", required=False, type=bool, nargs="?", const=False)
parser.add_argument("-a", "--architecture", required=False, type=str, nargs="?", const='unet')
parser.add_argument("-l", "--loss", required=False, type=str, nargs="?", const='tversky_loss')
parser.add_argument("-i", "--input", required=False, type=str)
parser.add_argument("-x", "--test", required=False, type=bool, nargs="?", const=False)

args = parser.parse_args()

if args.input is None:
    import params as p
    warnings.simplefilter("always")
    warnings.warn("No parameter file was given, 'params.py' will be used")
else:
    p = importlib.import_module(args.input)

if args.nodes is None:
        args.nodes = 1

#------------------------------------------------------------------#
# # # # # Functions # # # # #
#------------------------------------------------------------------#

def evaluate_individual_prediction(region, delta, npix, cold_cores, epochs, batch, lr, patience, disk_radius):
    print(region)
    print(delta)
    if cold_cores:
        n_labels = 2
    else:
        n_labels = 1
    CNN = CNNSegmentation(model = args.architecture, range_comp=p.range_compression, dataset = p.dataset, bands=p.bands, npix=npix, n_labels=n_labels, cold_cores=cold_cores, planck_path=p.planck_path, milca_path=p.milca_path, 
                        epochs=epochs, batch=batch, lr=lr, patience=patience, loss=args.loss, optimizer=p.optimizer, loops=p.loops, disk_radius=disk_radius, delta=delta, gamma=p.gamma, output_path = p.path)
    
    GenFiles = GenerateFiles(dataset = p.dataset, output_path = p.path)
    GenFiles.make_directories(output = True, replace=True)
    try:
        CNN.evaluate_prediction(regions = [region], plot=False, plot_patch = False)
    except:
        CNN = CNNSegmentation(model = args.architecture, range_comp=p.range_compression, dataset = p.dataset, bands=p.bands, npix=npix, n_labels=n_labels, cold_cores=cold_cores, planck_path=p.planck_path, milca_path=p.milca_path, 
                        epochs=epochs, batch=batch, lr=lr, patience=patience, loss=args.loss, optimizer=p.optimizer, loops=p.loops, disk_radius=disk_radius, delta=delta, gamma=p.gamma, output_path = p.path)
        GenFiles = GenerateFiles(dataset = p.dataset, output_path = p.path)
        GenFiles.make_directories(output = True, replace=True)
        CNN.evaluate_prediction(regions = [region], plot=False, plot_patch = False)

#------------------------------------------------------------------#
# # # # # Script # # # # #
#------------------------------------------------------------------#

if args.make_directories == True:
    GenFiles = GenerateFiles(dataset = p.dataset, output_path = p.path)

    GenFiles.clean_temp_directories()
    GenFiles.make_directories()
    GenFiles.make_directories(output = True, replace=p.merge_daily_output_directory)

if args.cluster_catalogs == True:
    MData = MakeData(dataset = p.dataset, npix=p.npix, loops=p.loops, planck_path=p.planck_path, milca_path=p.milca_path, disk_radius= None, output_path = p.path)

    # MData.create_catalogs(plot=False)
    # MData.create_catalogs(plot=p.plot_catalogs)
    MData.create_fake_source_catalog()

if args.test == True:
    MData = MakeData(dataset = p.dataset, npix=p.npix, loops=p.loops, planck_path=p.planck_path, milca_path=p.milca_path, disk_radius= p.disk_radius, output_path = p.path)

    warnings.simplefilter("always")
    warnings.warn("'-x' or '--test' is used for testing functions")
    MData.make_input(p=0, cold_cores=p.cold_cores, save_files=False, plot=True, verbose=False)
    # MData.test_data_generator(cold_cores=p.cold_cores, label_only=p.label_only, n_jobs=args.nodes, plot=True, verbose=False)
    # MData.preprocess(cold_cores=p.cold_cores, leastsq=p.fit_up_to_mode, range_comp=p.range_compression, plot=p.plot_dataset)

if args.dataset == True:
    if np.isscalar(p.disk_radius):
        MData = MakeData(dataset = p.dataset, npix=p.npix, loops=p.loops, planck_path=p.planck_path, milca_path=p.milca_path, disk_radius= p.disk_radius, output_path = p.path)

        MData.train_data_generator(loops=p.loops, cold_cores=p.cold_cores, label_only=p.label_only, n_jobs=args.nodes)
        MData.test_data_generator(cold_cores=p.cold_cores, label_only=p.label_only, n_jobs=args.nodes, plot=False, verbose=False)
        MData.preprocess(cold_cores=p.cold_cores, label_only=p.label_only, leastsq=p.fit_up_to_mode, range_comp=p.range_compression, n_jobs=args.nodes, plot=p.plot_dataset)
    else:
        for disk_radius in p.disk_radius:
            MData = MakeData(dataset = p.dataset, npix=p.npix, loops=p.loops, planck_path=p.planck_path, milca_path=p.milca_path, disk_radius= disk_radius, output_path = p.path)

            MData.train_data_generator(loops=p.loops, cold_cores=p.cold_cores, label_only=p.label_only, n_jobs=args.nodes)
            MData.test_data_generator(cold_cores=p.cold_cores, label_only=p.label_only, n_jobs=args.nodes, plot=False, verbose=False)
            MData.preprocess(cold_cores=p.cold_cores, label_only=p.label_only, leastsq=p.fit_up_to_mode, range_comp=p.range_compression, n_jobs=args.nodes, plot=p.plot_dataset)

if args.train == True:
    if np.isscalar(p.disk_radius):
        CNN = CNNSegmentation(model = args.architecture, range_comp=p.range_compression, dataset = p.dataset, bands=p.bands, npix=p.npix, n_labels=p.n_labels, cold_cores=p.cold_cores, planck_path=p.planck_path, milca_path=p.milca_path, 
                            epochs=p.epochs, batch=p.batch, lr=p.lr, patience=p.patience, loss=args.loss, optimizer=p.optimizer, loops=p.loops, disk_radius=p.disk_radius, delta=p.delta, gamma=p.gamma, output_path = p.path)

        CNN.train_model(regions = p.regions)
    else:
        for disk_radius in p.disk_radius:
            CNN = CNNSegmentation(model = args.architecture, range_comp=p.range_compression, dataset = p.dataset, bands=p.bands, npix=p.npix, n_labels=p.n_labels, cold_cores=p.cold_cores, planck_path=p.planck_path, milca_path=p.milca_path, 
                            epochs=p.epochs, batch=p.batch, lr=p.lr, patience=p.patience, loss=args.loss, optimizer=p.optimizer, loops=p.loops, disk_radius=disk_radius, delta=p.delta, gamma=p.gamma, output_path = p.path)

            CNN.train_model(regions = p.regions)

if args.predict == True:
    if np.isscalar(p.disk_radius):
        CNN = CNNSegmentation(model = args.architecture, range_comp=p.range_compression, dataset = p.dataset, bands=p.bands, npix=p.npix, n_labels=p.n_labels, cold_cores=p.cold_cores, planck_path=p.planck_path, milca_path=p.milca_path, 
                            epochs=p.epochs, batch=p.batch, lr=p.lr, patience=p.patience, loss=args.loss, optimizer=p.optimizer, loops=p.loops, disk_radius=p.disk_radius, delta=p.delta, gamma=p.gamma, output_path = p.path)

        CNN.evaluate_prediction(regions = p.regions, plot=p.plot_prediction, plot_patch = p.plot_individual_patchs)
    else:
        CNN = CNNSegmentation(model = args.architecture, range_comp=p.range_compression, dataset = p.dataset, bands=p.bands, npix=p.npix, n_labels=p.n_labels, cold_cores=p.cold_cores, planck_path=p.planck_path, milca_path=p.milca_path, 
                            epochs=p.epochs, batch=p.batch, lr=p.lr, patience=p.patience, loss=args.loss, optimizer=p.optimizer, loops=p.loops, disk_radius=disk_radius, delta=p.delta, gamma=p.gamma, output_path = p.path)

        CNN.evaluate_prediction(regions = p.regions, plot=p.plot_prediction, plot_patch = p.plot_individual_patchs)


if args.grid == 'train' or args.grid == 'evaluate':
    deltas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]#[0.025, 0.05, 0.075, 0.1, 0.125, 0.15, 0.2, 0.3, 0.4, 0.5, 0.7, 0.9]
    cold_cores = True #False
    if cold_cores:
        regions = [4,5,6,7,8,9]
        n_labels = 2
    else:
        regions = [0,1,2,3,4,5,6,7,8,9,10,11,12,13]
        n_labels = 1
    npix = 32 #64

    epochs = 20 #20 #30
    batch = 100
    lr = 5e-3 #1e-2
    patience = 8 #8 #20
    disk_radius = 2.5

    if args.grid == 'train':
        for region in regions:
            for delta in deltas:
                CNN = CNNSegmentation(model = args.architecture, range_comp=p.range_compression, dataset = p.dataset, bands=p.bands, npix=npix, n_labels=n_labels, cold_cores=cold_cores, planck_path=p.planck_path, milca_path=p.milca_path, 
                        epochs=epochs, batch=batch, lr=lr, patience=patience, loss=args.loss, optimizer=p.optimizer, loops=p.loops, disk_radius=disk_radius, delta=delta, gamma=p.gamma, output_path = p.path)
    
                CNN.train_model(regions = [region])

    if args.grid == 'evaluate':
        Parallel(n_jobs=args.nodes)(delayed(evaluate_individual_prediction)(region, delta, npix, cold_cores, epochs, batch, lr, patience, disk_radius) for delta in deltas for region in regions)
        CNN = CNNSegmentation(model = args.architecture, range_comp=p.range_compression, dataset = p.dataset, bands=p.bands, npix=npix, n_labels=p.n_labels, cold_cores=cold_cores, planck_path=p.planck_path, milca_path=p.milca_path, 
                        epochs=p.epochs, batch=p.batch, lr=p.lr, patience=p.patience, loss=args.loss, optimizer=p.optimizer, loops=p.loops, disk_radius=p.disk_radius, delta=p.delta, gamma=p.gamma, output_path = p.path)
        CNN.plot_tversky(regions)




if args.plot_tversky == True:
    cold_cores = False
    npix = 64
    epochs = 20 #30
    batch = 100
    lr = 1e-2 #5e-3 #1e-2 
    patience = 8 #20
    disk_radius = 2.5
    if cold_cores:
        regions = [4,5,6,7,8,9]
        n_labels = 2
    else:
        regions = [0,1,2,3,4,5,6,7,8,9,10,11,12,13]
        n_labels = 1
    # try:
    CNN = CNNSegmentation(model = args.architecture, range_comp=p.range_compression, dataset = p.dataset, bands=p.bands, npix=npix, n_labels=n_labels, cold_cores=cold_cores, planck_path=p.planck_path, milca_path=p.milca_path, 
                                    epochs=epochs, batch=batch, lr=lr, patience=patience, loss=args.loss, optimizer=p.optimizer, loops=p.loops, disk_radius=disk_radius, delta=p.delta, gamma=p.gamma, output_path = p.path)
    
    CNN.plot_tversky(regions)     
    # CNN.best_result(regions)
    # except:  
    #     CNN = CNNSegmentation(model = args.architecture, range_comp=p.range_compression, dataset = p.dataset, bands=p.bands, npix=npix, n_labels=n_labels, cold_cores=cold_cores, planck_path=p.planck_path, milca_path=p.milca_path, 
    #                                     epochs=epochs, batch=batch, lr=5e-3, patience=patience, loss=args.loss, optimizer=p.optimizer, loops=p.loops, disk_radius=disk_radius, delta=p.delta, gamma=p.gamma, output_path = p.path)
    #     # CNN.best_result(regions)
    #     CNN.plot_tversky(regions)     