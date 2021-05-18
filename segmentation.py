#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 01 10:00:58 2021

@author: Xavier Jimenez
"""


#------------------------------------------------------------------#
# # # # # Imports # # # # #
#------------------------------------------------------------------#
import numpy as np
import pandas as pd
import os
import time
from scipy import ndimage
import losses
import models
from generate_files import GenerateFiles
from make_data import MakeData

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import LogNorm
import seaborn as sns
import matplotlib.style as style
style.use('seaborn-poster') #sets the size of the charts
style.use('ggplot')

from astropy.io import fits
from astropy.wcs import WCS
from astropy.utils.data import get_pkg_data_filename
from astropy.coordinates import SkyCoord, match_coordinates_sky
import astropy.units as u
from astropy.stats import mad_std
import astrotools.healpytools as hpt
import astropy_healpix as ahp
from astropy.coordinates import ICRS

from tqdm import tqdm
from collections import Counter

import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger, TensorBoard
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
import tensorflow.keras.backend as K
from keras_unet_collection import models as tf_models

import healpy as hp
from hpproj import CutSky, to_coord

import logging
cs_logger = logging.getLogger('cutsky')
cs_logger.setLevel(logging.WARNING)
cs_logger.propagate = False
hpproj_logger = logging.getLogger('hpproj')
hpproj_logger.setLevel(logging.WARNING)
mpl_logger = logging.getLogger('matplotlib')
mpl_logger.setLevel(logging.WARNING)


class CNNSegmentation(MakeData):

    def __init__(self, dataset, bands, planck_path, milca_path, model, range_comp, epochs, batch, lr, patience, loss, optimizer, loops, size=64, disk_radius = None, drop_out=False, output_path = None):
        super().__init__(dataset, bands, planck_path, milca_path, disk_radius=disk_radius, output_path=output_path)

        self.range_comp = range_comp
        self.loops = loops
        self.size = size
        self.drop_out = drop_out
        self.epochs = epochs
        self.batch = batch
        self.lr = lr 
        self.patience = patience
        self.pmax=0.9
        self.dmin=3
        self.dmax=15

        self.output_name = 'e%s_b%s_lr%s_p%s_d%s'%(epochs, batch, lr, patience, disk_radius)

        optimizers_dict = {'sgd': SGD(lr=self.lr, momentum=0.9), 'adam': Adam(learning_rate=self.lr)}
        self.optimizer =  optimizers_dict[optimizer]
        self.optimizer_name =  optimizer
        losses_dict = {'binary_crossentropy': 'binary_crossentropy', 'weighted_binary_crossentropy': 'binary_crossentropy', 'tversky_loss': losses.tversky_loss, 'focal_tversky_loss': losses.focal_tversky_loss(gamma=0.75), 'dice_loss': losses.dice_loss, 
                       'combo_loss': losses.combo_loss(alpha=0.5,beta=0.5), 'cosine_tversky_loss': losses.cosine_tversky_loss(gamma=1), 'focal_dice_loss': losses.focal_dice_loss(delta=0.7, gamma_fd=0.75), 
                       'focal_loss': losses.focal_loss(alpha=None, beta=None, gamma_f=2.), 'mixed_focal_loss': losses.mixed_focal_loss(weight=None, alpha=None, beta=None, delta=0.7, gamma_f=2.,gamma_fd=0.75)}
        self.loss = losses_dict[loss]
        self.loss_name = loss
        input_size = (self.npix, self.npix, len(self.bands))
        filter_num = [64, 128, 256, 512]#, 1024]
        n_labels = 1
        dilation_num = [1, 3, 15, 31]
        filter_num_down = [64, 128, 256, 512]#, 1024]

        #               'vnet': tf_models.vnet_2d(input_size, filter_num, n_labels, res_num_ini=1, res_num_max=3, 
        #                     activation='ReLU', output_activation='Softmax', batch_norm=False, pool=True, unpool=True, name='vnet'),


        model_dict={'unet': models.unet, 
                    'attn_unet': tf_models.att_unet_2d(input_size, filter_num, n_labels, stack_num_down=2, stack_num_up=2, 
                            activation='ReLU', atten_activation='ReLU', attention='add', output_activation='Sigmoid', 
                            batch_norm=True, weights=None, pool=False, unpool=False, freeze_batch_norm=True, name='attunet'),
                    'r2u_net': tf_models.r2_unet_2d(input_size, filter_num, n_labels, 
                            stack_num_down=2, stack_num_up=2, recur_num=2,
                            activation='ReLU', output_activation='Softmax', 
                            batch_norm=False, pool=True, unpool=True, name='r2_unet'),
                    'unet_plus': tf_models.unet_plus_2d(input_size, filter_num, n_labels, stack_num_down=2, stack_num_up=2,
                            activation='ReLU', output_activation='Softmax', batch_norm=False, pool=True, unpool=True, deep_supervision=False, 
                            backbone=None, weights=None, freeze_backbone=True, freeze_batch_norm=True, name='xnet'),
                    'resunet_a': tf_models.resunet_a_2d(input_size, filter_num, dilation_num, n_labels,
                            aspp_num_down=256, aspp_num_up=128, activation='ReLU', output_activation='Softmax', 
                            batch_norm=True, pool=True, unpool=True, name='resunet'),
                    'u2net': tf_models.u2net_2d(input_size, n_labels, filter_num_down, filter_num_up='auto', filter_mid_num_down='auto', filter_mid_num_up='auto', 
                            filter_4f_num='auto', filter_4f_mid_num='auto', activation='ReLU', output_activation='Sigmoid', 
                            batch_norm=False, pool=True, unpool=True, deep_supervision=False, name='u2net'),
                    'unet_3plus': tf_models.unet_3plus_2d(input_size, n_labels, filter_num_down, filter_num_skip='auto', filter_num_aggregate='auto', 
                            stack_num_down=2, stack_num_up=1, activation='ReLU', output_activation='Sigmoid',
                            batch_norm=False, pool=True, unpool=True, deep_supervision=False, 
                            backbone=None, weights=None, freeze_backbone=True, freeze_batch_norm=True, name='unet3plus')}

        self.model = model_dict[model]
        self.model_name = model

    # def __init__(self, model, range_comp, dataset, bands, planck_path, milca_path, epochs, batch, lr, patience, loss, optimizer, loops, size=64, disk_radius = None, drop_out=False, output_path = None):
        self.path = os.getcwd() + '/'
        self.dataset = dataset # 'planck_z', 'planck_z_no-z', 'MCXC', 'RM30', 'RM50'
        self.bands = bands # '100GHz','143GHz','217GHz','353GHz','545GHz','857GHz', 'y-map'
        self.range_comp = range_comp
        maps = []
        self.freq = 0
        self.loops = loops
        if '100GHz' in  bands:
            maps.append((planck_path + "HFI_SkyMap_100-field-IQU_2048_R3.00_full.fits", {'legend': 'HFI 100', 'docontour': True}))
            self.freq += 2
        if '143GHz' in bands:
            maps.append((planck_path + "HFI_SkyMap_143-field-IQU_2048_R3.00_full.fits", {'legend': 'HFI 143', 'docontour': True}))
            self.freq += 4
        if '217GHz' in bands:
            maps.append((planck_path + "HFI_SkyMap_217-field-IQU_2048_R3.00_full.fits", {'legend': 'HFI 217', 'docontour': True}))
            self.freq += 8
        if '353GHz' in bands:
            maps.append((planck_path + "HFI_SkyMap_353-psb-field-IQU_2048_R3.00_full.fits", {'legend': 'HFI 353', 'docontour': True}))
            self.freq += 16
        if '545GHz' in bands:
            maps.append((planck_path + "HFI_SkyMap_545-field-Int_2048_R3.00_full.fits", {'legend': 'HFI 545', 'docontour': True}))
            self.freq += 32
        if '857GHz' in bands:
            maps.append((planck_path + "HFI_SkyMap_857-field-Int_2048_R3.00_full.fits", {'legend': 'HFI 857', 'docontour': True}))
            self.freq += 64
        if 'y-map' in bands:
            maps.append((milca_path + "milca_ymaps.fits", {'legend': 'MILCA y-map', 'docontour': True}))
            self.freq += 128
        maps.append((milca_path + "milca_ymaps.fits", {'legend': 'MILCA y-map', 'docontour': True}))
        
        self.maps = maps

        self.temp_path = self.path + 'to_clean/'
        if output_path is None:
            self.output_path = self.path + 'output/' + self.dataset + time.strftime("/%Y-%m-%d/")
        else:
            self.output_path = output_path + 'output/' + self.dataset + time.strftime("/%Y-%m-%d/")
        self.dataset_path = self.path + 'datasets/' + self.dataset + '/'
        self.planck_path = planck_path
        self.milca_path = milca_path
        self.npix = size
        self.pixsize = 1.7
        self.nside = 2
        if disk_radius is not None:
            self.disk_radius = disk_radius
        else:
            self.disk_radius = 'istri'

        self.size = size
        self.drop_out = drop_out
        self.epochs = epochs
        self.batch = batch
        self.lr = lr 
        self.patience = patience
        self.pmax=0.9
        self.dmin=3
        self.dmax=15

        self.output_name = 'e%s_b%s_lr%s_p%s_d%s'%(epochs, batch, lr, patience, disk_radius)

        optimizers_dict = {'sgd': SGD(lr=self.lr, momentum=0.9), 'adam': Adam(learning_rate=self.lr)}
        self.optimizer =  optimizers_dict[optimizer]
        self.optimizer_name =  optimizer
        losses_dict = {'binary_crossentropy': 'binary_crossentropy', 'weighted_binary_crossentropy': 'binary_crossentropy', 'tversky_loss': losses.tversky_loss, 'focal_tversky_loss': losses.focal_tversky_loss(gamma=0.75), 'dice_loss': losses.dice_loss, 
                       'combo_loss': losses.combo_loss(alpha=0.5,beta=0.5), 'cosine_tversky_loss': losses.cosine_tversky_loss(gamma=1), 'focal_dice_loss': losses.focal_dice_loss(delta=0.7, gamma_fd=0.75), 
                       'focal_loss': losses.focal_loss(alpha=None, beta=None, gamma_f=2.), 'mixed_focal_loss': losses.mixed_focal_loss(weight=None, alpha=None, beta=None, delta=0.7, gamma_f=2.,gamma_fd=0.75)}
        self.loss = losses_dict[loss]
        self.loss_name = loss
        input_size = (self.npix, self.npix, len(self.bands))
        filter_num = [64, 128, 256, 512]#, 1024]
        n_labels = 1
        dilation_num = [1, 3, 15, 31]
        filter_num_down = [64, 128, 256, 512]#, 1024]

        #               'vnet': tf_models.vnet_2d(input_size, filter_num, n_labels, res_num_ini=1, res_num_max=3, 
        #                     activation='ReLU', output_activation='Softmax', batch_norm=False, pool=True, unpool=True, name='vnet'),


        model_dict={'unet': models.unet, 
                    'attn_unet': tf_models.att_unet_2d(input_size, filter_num, n_labels, stack_num_down=2, stack_num_up=2, 
                            activation='ReLU', atten_activation='ReLU', attention='add', output_activation='Sigmoid', 
                            batch_norm=True, weights=None, pool=False, unpool=False, freeze_batch_norm=True, name='attunet'),
                    'r2u_net': tf_models.r2_unet_2d(input_size, filter_num, n_labels, 
                            stack_num_down=2, stack_num_up=2, recur_num=2,
                            activation='ReLU', output_activation='Softmax', 
                            batch_norm=False, pool=True, unpool=True, name='r2_unet'),
                    'unet_plus': tf_models.unet_plus_2d(input_size, filter_num, n_labels, stack_num_down=2, stack_num_up=2,
                            activation='ReLU', output_activation='Softmax', batch_norm=False, pool=True, unpool=True, deep_supervision=False, 
                            backbone=None, weights=None, freeze_backbone=True, freeze_batch_norm=True, name='xnet'),
                    'resunet_a': tf_models.resunet_a_2d(input_size, filter_num, dilation_num, n_labels,
                            aspp_num_down=256, aspp_num_up=128, activation='ReLU', output_activation='Softmax', 
                            batch_norm=True, pool=True, unpool=True, name='resunet'),
                    'u2net': tf_models.u2net_2d(input_size, n_labels, filter_num_down, filter_num_up='auto', filter_mid_num_down='auto', filter_mid_num_up='auto', 
                            filter_4f_num='auto', filter_4f_mid_num='auto', activation='ReLU', output_activation='Sigmoid', 
                            batch_norm=False, pool=True, unpool=True, deep_supervision=False, name='u2net'),
                    'unet_3plus': tf_models.unet_3plus_2d(input_size, n_labels, filter_num_down, filter_num_skip='auto', filter_num_aggregate='auto', 
                            stack_num_down=2, stack_num_up=1, activation='ReLU', output_activation='Sigmoid',
                            batch_norm=False, pool=True, unpool=True, deep_supervision=False, 
                            backbone=None, weights=None, freeze_backbone=True, freeze_batch_norm=True, name='unet3plus')}

        self.model = model_dict[model]
        self.model_name = model

    def npy_to_tfdata(self, batch_size=20, buffer_size=1000, input_train=None, input_val=None, input_test=None, output_train=None, output_val=None, output_test=None):
        if input_train is None:
            if self.range_comp:
                input_train = np.load(self.dataset_path + 'input_train_pre_f%s_'%self.freq + self.dataset + '.npz')['arr_0']
            else:
                input_train = np.load(self.dataset_path + 'input_train_pre_f%s_r0_'%self.freq + self.dataset + '.npz')['arr_0']
        if input_val is None:
            if self.range_comp:
                input_val = np.load(self.dataset_path + 'input_val_pre_f%s_'%self.freq + self.dataset + '.npz')['arr_0']
            else:
                input_val = np.load(self.dataset_path + 'input_val_pre_f%s_r0_'%self.freq + self.dataset + '.npz')['arr_0']
        if input_test is None:
            if self.range_comp:
                input_test = np.load(self.dataset_path + 'input_test_pre_f%s_'%self.freq + self.dataset + '.npz')['arr_0']
            else:
                input_test = np.load(self.dataset_path + 'input_test_pre_f%s_r0_'%self.freq + self.dataset + '.npz')['arr_0']
        if output_train is None:
            output_train = np.load(self.dataset_path + 'label_train_pre_f%s_d%s_'%(self.freq, self.disk_radius) + self.dataset + '.npz')['arr_0']
        if output_val is None:
            output_val = np.load(self.dataset_path + 'label_val_pre_f%s_d%s_'%(self.freq, self.disk_radius) + self.dataset + '.npz')['arr_0']
        if output_test is None:
            output_test = np.load(self.dataset_path + 'label_test_pre_f%s_d%s_'%(self.freq, self.disk_radius) + self.dataset + '.npz')['arr_0']


        train_dataset = tf.data.Dataset.from_tensor_slices((input_train, output_train))
        val_dataset = tf.data.Dataset.from_tensor_slices((input_val, output_val))
        test_dataset = tf.data.Dataset.from_tensor_slices((input_test, output_test))

        train_dataset = train_dataset.shuffle(buffer_size).batch(batch_size).repeat()
        train_dataset = train_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
        val_dataset = val_dataset.shuffle(buffer_size).batch(batch_size).repeat()
        val_dataset = val_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
        test_dataset = test_dataset.batch(batch_size)


        return train_dataset, val_dataset, test_dataset

    def prepare(self, ds, shuffle=False, augment=False):
        AUTOTUNE = tf.data.AUTOTUNE
        data_augmentation = tf.keras.Sequential([
            tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
            tf.keras.layers.experimental.preprocessing.RandomRotation(0.2)
            ])

        if shuffle:
            ds = ds.shuffle(1000)

        # Batch all datasets
        ds = ds.batch(self.batch)

        # Use data augmentation only on the training set
        if augment:
            ds = ds.map(lambda x, y: (data_augmentation(x, training=True), y), 
                        num_parallel_calls=AUTOTUNE)

        # Use buffered prefecting on all datasets
        return ds.prefetch(buffer_size=AUTOTUNE)

    def train_model(self):
        input_train = np.load(self.dataset_path + 'input_train_pre_f%s_'%self.freq + self.dataset + '.npz')['arr_0']
        input_val = np.load(self.dataset_path + 'input_val_pre_f%s_'%self.freq + self.dataset + '.npz')['arr_0']
        train_dataset, valid_dataset, _ = self.npy_to_tfdata(batch_size=self.batch, buffer_size=1000)

        callbacks = [
            ModelCheckpoint(monitor='val_loss', filepath=self.path + "tf_saves/" + self.dataset + "/model_%s_%s_%s_f%s_"%(self.model_name, self.loss_name, self.optimizer_name, self.freq) + self.output_name + ".h5", save_best_only=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=self.patience),
            CSVLogger(self.path + "tf_saves/" + self.dataset + "/data_%s_%s_%s_f%s_"%(self.model_name, self.loss_name, self.optimizer_name, self.freq) + self.output_name + ".csv"),
            TensorBoard(),
            EarlyStopping(monitor='val_loss', patience=self.patience, restore_best_weights=True)
        ]


        if self.model_name == 'unet':
            input_size = (self.npix, self.npix, len(self.bands))
            model = self.model(self.optimizer, input_size, self.loss)
        else:
            model  =self.model
            metrics = [losses.dsc, tf.keras.metrics.Recall(), tf.keras.metrics.Precision()]#, losses.iou]
            model.compile(loss=self.loss, optimizer=self.optimizer, metrics=metrics) #tf.keras.losses.categorical_crossentropy

        
        if self.loss_name == 'weighted_binary_crossentropy':
            model_history = model.fit(train_dataset.map(self.add_sample_weights),
                validation_data=valid_dataset,
                epochs=self.epochs,
                steps_per_epoch = int(len(input_train)/(self.batch)),
                validation_steps= int(len(input_val)//(self.batch)),
                callbacks=callbacks)
        else:
            model_history = model.fit(train_dataset,
                validation_data=valid_dataset,
                epochs=self.epochs,
                steps_per_epoch = int(len(input_train)/(self.batch)),
                validation_steps= int(len(input_val)//(self.batch)),
                callbacks=callbacks)

    def pixel_neighbours(self, im, center, p, pmax, dmax):

        rows, cols = np.shape(im)

        i, j = p[0], p[1]

        rmin = i - 1 if i - 1 >= 0 else 0
        rmax = i + 1 if i + 1 < rows else i

        cmin = j - 1 if j - 1 >= 0 else 0
        cmax = j + 1 if j + 1 < cols else j

        neighbours = []

        for x in range(rmin, rmax + 1):
            for y in range(cmin, cmax + 1):
                if np.sqrt((x-center[0])**2 + (y-center[1])**2) < dmax:
                    neighbours.append([x, y])
        neighbours.remove([p[0], p[1]])

        bellow_pmax = []
        for (c1, c2) in neighbours:
            if im[c1,c2] < pmax:
                bellow_pmax.append([c1,c2])

        for (c1, c2) in bellow_pmax:
            neighbours.remove([c1,c2])

        return neighbours

    def detect_clusters(self, im, pmax, dmin, dmax):
        
        im_copy = im.copy()
        mask = np.ones_like(im)
        center = [np.argmax(im_copy)//np.shape(im_copy)[1], np.argmax(im_copy)%np.shape(im_copy)[1]]
                
        x_peak_list, y_peak_list = [], []
        mask_list = []

        while im[center[0], center[1]] > pmax:
            center = [np.argmax(im_copy)//np.shape(im_copy)[1], np.argmax(im_copy)%np.shape(im_copy)[1]]
            individual_mask = np.ones_like(im)
            individual_mask[center[0], center[1]] = 0

            mask[center[0], center[1]] = 0
            all_neighbours = []
            neighbours = self.pixel_neighbours(im_copy, center, center, pmax=pmax, dmax=dmax)
            new_neighbours = neighbours.copy()
            all_neighbours = neighbours.copy()

            for (c1,c2) in neighbours:
                mask[c1,c2] = 0
                individual_mask[c1,c2] = 0
            im_copy = im_copy * mask
            empty_counter = []
            while len(empty_counter) < len(new_neighbours):
                empty_counter = []
                for p in new_neighbours:
                    neighbours = self.pixel_neighbours(im_copy, center, p, pmax=pmax, dmax=dmax)
                    if not neighbours:
                        empty_counter.append(True)
                    else:
                        all_neighbours = np.concatenate((np.array(all_neighbours), np.array(neighbours)))
                        for (c1,c2) in neighbours:
                            mask[c1,c2] = 0
                            individual_mask[c1,c2] = 0
                        im_copy = im_copy * mask
                new_neighbours = all_neighbours[len(new_neighbours):]
            if len(all_neighbours) > dmin:
                mask_list.append(individual_mask)
                x_peak_list.append(center[1])
                y_peak_list.append(center[0])

        return mask_list, x_peak_list, y_peak_list

    def match_detections_against_catalogs(self, pixel_coords, plot=True):

        maps = self.maps

        planck_z = pd.read_csv(self.path + 'catalogs/planck_z' + '.csv')
        planck_no_z = pd.read_csv(self.path + 'catalogs/planck_no-z' + '.csv')
        MCXC = pd.read_csv(self.path + 'catalogs/MCXC_no_planck' + '.csv')
        RM50 = pd.read_csv(self.path + 'catalogs/RM50_no_planck' + '.csv')
        RM30 = pd.read_csv(self.path + 'catalogs/RM30_no_planck' + '.csv')
        coord_catalog = pd.concat([planck_z[['RA', 'DEC']].copy(), planck_no_z[['RA', 'DEC']].copy(), MCXC[['RA', 'DEC']].copy(),
                RM50[['RA', 'DEC']].copy(), RM30[['RA', 'DEC']].copy()], ignore_index=True)

        cutsky = CutSky(maps, npix=self.npix, pixsize=self.pixsize, low_mem=False)

        test_coords = np.load(self.dataset_path + 'test_coordinates_f%s_'%self.freq + self.dataset + '.npy')
        dataset_type_test = np.load(self.dataset_path + 'type_test_f%s_'%self.freq + self.dataset + '.npz')['arr_0']
        test_position = np.ndarray((len(pixel_coords), 2))
        index = 0

        type_count_test = Counter(dataset_type_test)
        print("[Inputs] training: {:.0f}, validation: {:.0f}, test: {:.0f}".format(type_count_test['train'], type_count_test['val'], type_count_test['test']))

        for i in range(len(dataset_type_test)):
            if dataset_type_test[i] == 'test':
                test_position[index,0], test_position[index,1]  = test_coords[0,i], test_coords[1,i]
                index += 1

        test_coords = SkyCoord(ra=test_position[:,0],
                               dec=test_position[:,1], unit='deg')

        ra_list, dec_list = [], []
        for i, coord in enumerate(test_coords):
            patch = cutsky.cut_fits(coord)
            HDU = patch[-1]['fits']
            wcs = WCS(HDU.header)
            # x, y = wcs.world_to_pixel_values(test_position[i,0], test_position[i,1])
            # print(x, y)
            for j in range(len(pixel_coords[i])):
                ra, dec = wcs.pixel_to_world_values(pixel_coords[i][j][1], pixel_coords[i][j][0])
                ra_list.append(ra)
                dec_list.append(dec)

        df_detections = pd.DataFrame(data={'RA': ra_list,'DEC': dec_list})

        df_duplicates = self.return_dup_coords(df_detections, plot=True)

        # if len(df_duplicates) != 0 and len(df_duplicates) != 1:
        #     df_patch_dup = self.check_coord_on_patch(df_duplicates, cutsky, test_coords, plot=True)
            # print(df_patch_dup.head(5))


        df_planck_z_detections, planck_z_detections_number = self.match_with_catalog(df_detections, planck_z, output_name='planck_z', plot=plot)
        df_planck_no_z_detections, planck_no_z_detections_number = self.match_with_catalog(df_detections, planck_no_z, output_name='planck_no-z', plot=plot)
        df_MCXC_detections, MCXC_detections_number = self.match_with_catalog(df_detections, MCXC, output_name='MCXC', plot=plot)
        df_RM50_detections, RM50_detections_number = self.match_with_catalog(df_detections, RM50, output_name='RM50', plot=plot)
        df_RM30_detections, RM30_detections_number = self.match_with_catalog(df_detections, RM30, output_name='RM30', plot=plot)
        
        npz = self.objects_in_patch(p=0, catalog='planck_z', plot=False)
        npnz = self.objects_in_patch(p=0, catalog='planck_no-z')
        nmcxc = self.objects_in_patch(p=0, catalog='MCXC')
        nrm50 = self.objects_in_patch(p=0, catalog='RM30')
        nrm30 = self.objects_in_patch(p=0, catalog='RM50')

        print('planck_z: %s(%s)/%s(%s)'%(len(df_planck_z_detections), planck_z_detections_number, npz[0], npz[1]), 
              'planck_no-z: %s(%s)/%s(%s)'%(len(df_planck_no_z_detections), planck_no_z_detections_number, npnz[0], npnz[1]), 
              'MCXC: %s(%s)/%s(%s)'%(len(df_MCXC_detections), MCXC_detections_number, nmcxc[0], nmcxc[1]), 
              'RM50: %s(%s)/%s(%s)'%(len(df_RM50_detections), RM50_detections_number, nrm50[0], nrm50[1]), 
              'RM30: %s(%s)/%s(%s)'%(len(df_RM30_detections), RM30_detections_number, nrm30[0], nrm30[1]), 
              'Unknown: %s(%s)'%(len(df_detections) - len(df_duplicates) + (planck_z_detections_number-len(df_planck_z_detections) - (planck_no_z_detections_number-len(df_planck_no_z_detections))
                                 - (MCXC_detections_number-len(df_MCXC_detections)) - (RM50_detections_number-len(df_RM50_detections)) - (RM30_detections_number-len(df_RM30_detections)))
                                 - len(df_planck_z_detections)-len(df_planck_no_z_detections)-len(df_MCXC_detections)-len(df_RM50_detections)-len(df_RM30_detections), 
                                 len(df_detections)-len(df_planck_z_detections)-len(df_planck_no_z_detections)-len(df_MCXC_detections)-len(df_RM50_detections)-len(df_RM30_detections)))

        try:
            file = open(self.output_path + 'figures/' + 'prediction_%s_%s_%s_f%s_'%(self.model_name, self.loss_name, self.optimizer_name, self.freq) + self.output_name + "/results.txt","w")
        except:
            file = open(self.output_path + 'figures/' + 'prediction_%s_l%s_o%s_f%s_'%(self.model_name, self.loss_name, self.optimizer_name, self.freq) + self.output_name + "/results.txt","w")
        
        L = ['planck_z: %s(%s)/%s(%s)'%(len(df_planck_z_detections), planck_z_detections_number, npz[0], npz[1])+ " \n", 
             'planck_no-z: %s(%s)/%s(%s)'%(len(df_planck_no_z_detections), planck_no_z_detections_number, npnz[0], npnz[1])+ " \n",
             'MCXC: %s(%s)/%s(%s)'%(len(df_MCXC_detections), MCXC_detections_number, nmcxc[0], nmcxc[1])+ " \n",
             'RM50: %s(%s)/%s(%s)'%(len(df_RM50_detections), RM50_detections_number, nrm50[0], nrm50[1])+ " \n",
             'RM30: %s(%s)/%s(%s)'%(len(df_RM30_detections), RM30_detections_number, nrm30[0], nrm30[1])+ " \n",
             'Unknown: %s(%s)'%(len(df_detections) - len(df_duplicates) + (planck_z_detections_number-len(df_planck_z_detections) - (planck_no_z_detections_number-len(df_planck_no_z_detections))
                                 - (MCXC_detections_number-len(df_MCXC_detections)) - (RM50_detections_number-len(df_RM50_detections)) - (RM30_detections_number-len(df_RM30_detections)))
                                 - len(df_planck_z_detections)-len(df_planck_no_z_detections)-len(df_MCXC_detections)-len(df_RM50_detections)-len(df_RM30_detections), 
                                 len(df_detections)-len(df_planck_z_detections)-len(df_planck_no_z_detections)-len(df_MCXC_detections)-len(df_RM50_detections)-len(df_RM30_detections))] 
        file.writelines(L)
        file.close()
  

        # print('duplicate detections (total): %s'%(len(df_duplicates)),
        #       'planck_z: %s(%s)/%s(%s)'%(len(df_planck_z_detections), planck_z_detections_number, 69, 87), 
        #       'planck_no-z: %s(%s)/%s(%s)'%(len(df_planck_no_z_detections), planck_no_z_detections_number, 3, 7), 
        #       'MCXC: %s(%s)/%s(%s)'%(len(df_MCXC_detections), MCXC_detections_number, 14, 14), 
        #       'RM50: %s(%s)/%s(%s)'%(len(df_RM50_detections), RM50_detections_number, 139, 167), 
        #       'RM30: %s(%s)/%s(%s)'%(len(df_RM30_detections), RM30_detections_number, 40, 54), 
        #       'Unknown: %s(%s)'%(len(df_detections) - len(df_duplicates) + (planck_z_detections_number-len(df_planck_z_detections) - (planck_no_z_detections_number-len(df_planck_no_z_detections))
        #                          - (MCXC_detections_number-len(df_MCXC_detections)) - (RM50_detections_number-len(df_RM50_detections)) - (RM30_detections_number-len(df_RM30_detections)))
        #                          - len(df_planck_z_detections)-len(df_planck_no_z_detections)-len(df_MCXC_detections)-len(df_RM50_detections)-len(df_RM30_detections), 
        #                          len(df_detections)-len(df_planck_z_detections)-len(df_planck_no_z_detections)-len(df_MCXC_detections)-len(df_RM50_detections)-len(df_RM30_detections)))

        return df_planck_z_detections, df_planck_no_z_detections, df_MCXC_detections, df_RM50_detections, df_RM30_detections

    def cluster_number(self, df_catalog, cutsky, test_coords):
        # if type(test_position) == np.ndarray:
        #     df_test_positions = pd.DataFrame(data={'RA': test_position[:,0], 'DEC': test_position[:,1]})
        # else:
        #     df_test_positions = test_position
        # df_test_positions = self.match_with_catalog(df_test_positions, df, output_name=output_name, plot=True)
        coords = SkyCoord(ra=df_catalog['RA'].values, dec=df_catalog['DEC'].values, unit='deg', frame='icrs')

        ra_list, dec_list = [], []
        # for j, catalog_coord in enumerate(coords):
        #     if hp.ang2pix(self.nside, catalog_coord.ra, catalog_coord.dec, lonlat=True) == 6 or hp.ang2pix(self.nside, catalog_coord.ra, catalog_coord.dec, lonlat=True) == 7:
        #         df_catalog = df_catalog.drop(j)
        # print(len(coords))
        # coords = SkyCoord(ra=df_catalog['RA'].values, dec=df_catalog['DEC'].values, unit='deg', frame='icrs')
        # print(len(coords))

        for i, coord in enumerate(test_coords):
            patch = cutsky.cut_fits(coord)
            HDU = patch[-1]['fits']
            wcs = WCS(HDU.header)
            for j, catalog_coord in enumerate(coords):
                x, y = wcs.world_to_pixel_values(catalog_coord.galactic.l, catalog_coord.galactic.b)
                if x>0 and x<self.npix and y>0 and y<self.npix:
                    print(x,y)
                    ra_list.append(df_catalog['RA'].values[j])   
                    dec_list.append(df_catalog['DEC'].values[j])
        
        df_in_patch = pd.DataFrame(data = {'RA': ra_list, 'DEC': dec_list})
        print(df_in_patch.head())

        if len(df_in_patch) != 0 and len(df_in_patch) != 1:
            df_in_patch = self.remove_duplicates(df_in_patch)

        return len(df_in_patch)

    def remove_duplicates(self, df, tol=4):
        coords = SkyCoord(ra=df['RA'].values, dec=df['DEC'].values, unit='deg')
        _, d2d, _ = match_coordinates_sky(coords, coords, nthneighbor=2)
        isdup = d2d < tol*u.arcmin
        df['isdup'] = isdup
        df.query("isdup == False", inplace=True)
        df.drop(columns=['isdup'], inplace=True)

        return df

    def return_dup_coords(self, df_detections, tol=4, plot=True):

        ID = np.arange(0, len(df_detections))
        df_detections_copy = df_detections[['RA', 'DEC']].copy()
        df_detections = df_detections.rename(columns={"RA": "RA_1", "DEC": "DEC_1"})
        df_detections_copy = df_detections_copy.rename(columns={"RA": "RA_2", "DEC": "DEC_2"})
        df_detections_copy.insert(loc=0, value=ID, column='ID')

        coords_1 = SkyCoord(ra=df_detections['RA_1'].values, dec=df_detections['DEC_1'].values, unit='deg')
        coords_2 = SkyCoord(ra=df_detections_copy['RA_2'].values, dec=df_detections_copy['DEC_2'].values, unit='deg')

        idx, d2d, _ = match_coordinates_sky(coords_1, coords_2, nthneighbor=2)
        isdup = d2d < tol*u.arcmin

        df_d2d = pd.DataFrame(data={'isdup': isdup, 'idx': idx, 'd2d': d2d})
        df_d2d.query("isdup == True", inplace=True)
        df_d2d.drop(columns=['isdup'], inplace=True)

        if plot == True:
            fig = plt.figure(figsize=(8,8), tight_layout=False)
            ax = fig.add_subplot(111)
            ax.set_facecolor('white')
            ax.grid(True, color='grey', lw=0.5)
            ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
            ax.set_xlabel(r'$\mathrm{angular\;distance\;\left(arcmin\right)}$', fontsize=20)
            ax.set_ylabel('Counts', fontsize=20)
            ax.hist(np.array(df_d2d['d2d'].values)*60)
            ax.axvline(tol, color='k', linestyle='--')
            ax.set_xlim(0, 2*tol)
            plt.savefig(self.output_path + 'figures/prediction_%s_%s_%s_f%s_%s/'%(self.model_name, self.loss_name, self.optimizer_name, self.freq, self.output_name) + 'd2d_detections_duplicates_' + self.dataset + '.png', bbox_inches='tight', transparent=False)
            plt.show()
            plt.close()

        df_detections['isdup'], df_detections['ID'], df_detections['d2d'] = isdup, idx, d2d

        df_common = pd.merge(df_detections, df_detections_copy, indicator=True, on='ID', how='outer').query('_merge=="both"').drop('_merge', axis=1)

        df_common.query("isdup == True", inplace=True)
        df_common.drop(columns=['isdup', 'ID'], inplace=True)

        return df_common

    def check_coord_on_patch(self, df_duplicates, cutsky, test_coords, plot=True):
        df_patch_dup = pd.DataFrame(columns = ['patch_id', 'RA_1', 'DEC_1', 'RA_2', 'DEC_2'])
        for i, coord in enumerate(test_coords):
            patch = cutsky.cut_fits(coord)
            HDU = patch[-1]['fits']
            wcs = WCS(HDU.header)
            x1, y1 = wcs.world_to_pixel_values(df_duplicates['RA_1'].values[i], df_duplicates['DEC_1'].values[i])
            x2, y2 = wcs.world_to_pixel_values(df_duplicates['RA_2'].values[i], df_duplicates['DEC_2'].values[i])
            if x1>0 and x1<self.npix and y1>0 and y1<self.npix and x2>0 and x2<self.npix and y2>0 and y2<self.npix:
                df_patch_dup['patch_id'] = i
                df_patch_dup['RA_1'] = df_duplicates['RA_1'].values[i]
                df_patch_dup['DEC_1'] = df_duplicates['DEC_1'].values[i]
                df_patch_dup['RA_2'] = df_duplicates['RA_2'].values[i]
                df_patch_dup['DEC_2'] = df_duplicates['DEC_2'].values[i]


        if plot == True:
            GenerateFiles.make_directory(self, path_to_file = self.output_path + 'figures/' + 'duplicate_detections_' + self.output_name)
            for i in range(len(df_patch_dup)):
                fig = plt.figure(figsize=(8,8), tight_layout=False)

                ax = fig.add_subplot(111)
                patch = cutsky.cut_fits(test_coords[i])
                HDU = patch[-1]['fits']
                wcs = WCS(HDU.header)
                x1, y1 = wcs.world_to_pixel_values(df_duplicates['RA_1'].values[i], df_duplicates['DEC_1'].values[i])
                x2, y2 = wcs.world_to_pixel_values(df_duplicates['RA_2'].values[i], df_duplicates['DEC_2'].values[i])
                ax.imshow(HDU.data, origin='lower')
                ax.scatter(x1,y1)
                ax.scatter(x2,y2)

                plt.savefig(self.output_path + 'figures/' + 'duplicate_detections_' + self.output_name + '/patch_%s'%i + '.png', bbox_inches='tight', transparent=False)
                plt.show()
                plt.close()

        return df_patch_dup

    def match_with_catalog(self, df_main, df_catalog, output_name=None, plot=False):  

        ID = np.arange(0, len(df_catalog))
        df_catalog = df_catalog[['RA', 'DEC']].copy()
        df_catalog.insert(loc=0, value=ID, column='ID')

        scatalog_sub = SkyCoord(ra=df_main['RA'].values, dec=df_main['DEC'].values, unit='deg')
        pcatalog_sub = SkyCoord(ra=df_catalog['RA'].values, dec=df_catalog['DEC'].values, unit='deg')
        idx, d2d, _ = match_coordinates_sky(scatalog_sub, pcatalog_sub, nthneighbor=1)

        tol = 7
        ismatched = d2d < tol*u.arcminute #threshold to consider whether or not two galaxies are the same

        df_d2d = pd.DataFrame(data={'ismatched': ismatched, 'idx': idx, 'd2d': d2d})

        df_d2d.query("ismatched == True", inplace=True)
        df_d2d.drop(columns=['ismatched'], inplace=True)

        if plot == True:
            fig = plt.figure(figsize=(8,8), tight_layout=False)
            ax = fig.add_subplot(111)
            ax.set_facecolor('white')
            ax.grid(True, color='grey', lw=0.5)
            ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
            ax.set_xlabel(r'$\mathrm{angular\;distance\;\left(arcmin\right)}$', fontsize=20)
            ax.set_ylabel('Counts', fontsize=20)
            ax.hist(np.array(df_d2d['d2d'].values)*60)
            ax.axvline(tol, color='k', linestyle='--')
            ax.set_xlim(0, 2*tol)
            plt.savefig(self.output_path + 'figures/prediction_%s_%s_%s_f%s_%s/'%(self.model_name, self.loss_name, self.optimizer_name, self.freq, self.output_name) + 'd2d_detections_%s'%output_name + '.png', bbox_inches='tight', transparent=False)
            plt.show()
            plt.close()

        df_main['ismatched'], df_main['ID'] = ismatched, idx

        df_catalog.drop(columns=['RA', 'DEC'], inplace=True)

        df_common = pd.merge(df_main, df_catalog, indicator=True, on='ID', how='outer').query('_merge=="both"').drop('_merge', axis=1)

        df_common.query("ismatched == True", inplace=True)

        size = len(df_common)
        if len(df_common) != 0 and len(df_common) != 1:
            df_common = self.remove_duplicates(df_common)
        df_common = df_common.drop_duplicates(subset='ID', keep="first")
        df_common.drop(columns=['ismatched', 'ID'], inplace=True)

        return df_common, size
        


    def evaluate_prediction(self, plot=True, plot_patch=True):
        _, _, test_dataset = self.npy_to_tfdata(batch_size=self.batch, buffer_size=1000)

        from tensorflow.keras.utils import CustomObjectScope

        with CustomObjectScope({'iou': losses.iou, 'f1': losses.f1, 'dsc': losses.dsc, self.loss_name: self.loss, 'loss_function': self.loss}):
            try:
                model = tf.keras.models.load_model(self.path + "tf_saves/" + self.dataset + "/model_%s_%s_%s_f%s_"%(self.model_name, self.loss_name, self.optimizer_name, self.freq) + self.output_name + ".h5")
            except:
                model = tf.keras.models.load_model(self.path + "tf_saves/" + self.dataset + "/model_%s_l%s_o%s_f%s_"%(self.model_name, self.loss_name, self.optimizer_name, self.freq) + self.output_name + ".h5")

        if plot == True:
            GenerateFiles.make_directory(self, path_to_file = self.output_path + 'figures/' + 'prediction_%s_%s_%s_f%s_'%(self.model_name, self.loss_name, self.optimizer_name, self.freq) + self.output_name)
            for metric in ['dsc', 'precision', 'recall', 'loss', 'lr']:#['f1', 'acc', 'iou', 'precision', 'recall', 'loss', 'lr']:
                self.plot_metric(metric)
            pixel_coords = self.show_predictions(model, dataset = test_dataset, num=30, plot=plot_patch)
        else:
            self.make_predictions(model, dataset = test_dataset, num=30)
            pixel_coords = self.load_predictions()


        self.match_detections_against_catalogs(pixel_coords, plot=plot)

    

    def load_predictions(self):
        pixel_coords = []
        try:
            pred_mask = np.load(self.dataset_path + 'prediction_mask_%s_%s_%s_f%s_%s'%(self.model_name, self.loss_name, self.optimizer_name, self.freq, self.output_name) + '.npy')
        except:
            pred_mask = np.load(self.dataset_path + 'prediction_mask_%s_l%s_o%s_f%s_%s'%(self.model_name, self.loss_name, self.optimizer_name, self.freq, self.output_name) + '.npy')

        for k in range(len(pred_mask)):
            coords_in_patch = []
            mask_list, _, _ = self.detect_clusters(im = pred_mask[k], pmax=self.pmax, dmin=self.dmin, dmax=self.dmax)
            if mask_list:
                for i in range(len(mask_list)):
                    com = ndimage.measurements.center_of_mass(pred_mask[k]*(np.ones_like(pred_mask[k]) - mask_list[i]))
                    coords_in_patch.append(com)

            pixel_coords.append(coords_in_patch)
        
        return pixel_coords

    def make_predictions(self, model, dataset, num=1):
        n = 0
        milca_test = np.load(self.dataset_path + 'milca_test_pre_f%s_'%self.freq + self.dataset + '.npz')['arr_0']
        predicted_masks = np.ndarray((len(milca_test),self.npix,self.npix,1))

        for image, mask in dataset.take(num):
            pred_mask = model.predict(image)
            for k in range(len(pred_mask)):
                predicted_masks[self.batch*n+k,:,:,0] = pred_mask[k,:,:,0]
            n += 1

        np.save(self.dataset_path + 'prediction_mask_%s_%s_%s_f%s_%s'%(self.model_name, self.loss_name, self.optimizer_name, self.freq, self.output_name) , predicted_masks)

    def show_predictions(self, model, dataset, num=1, plot=True):
        milca_test = np.load(self.dataset_path + 'milca_test_pre_f%s_'%self.freq + self.dataset + '.npz')['arr_0']

        n = 0
        pixel_coords = []
        predicted_masks = np.ndarray((len(milca_test),self.npix,self.npix,1))
        final_masks = np.ndarray((len(milca_test),self.npix,self.npix,1))

        for image, mask in dataset.take(num):
            print(n)
            pred_mask = model.predict(image)
            for k in range(len(pred_mask)):
                predicted_masks[self.batch*n+k,:,:,0] = pred_mask[k,:,:,0]
                if plot == True:
                    plt.figure(figsize=(36, 7), tight_layout=True)

                    title = ['y-map', 'True Mask', 'Predicted Mask', 'Detected clusters']

                    plt.subplot(1, 4, 1)
                    plt.title(title[0])
                    plt.imshow(milca_test[self.batch*n+k,:,:,0], origin='lower')
                    plt.axis('off')

                    plt.subplot(1, 4, 2)
                    plt.title(title[1])
                    plt.imshow(mask[k], origin='lower')
                    plt.axis('off')
                                
                    plt.subplot(1, 4, 3)
                    plt.title(title[2])
                    plt.imshow(pred_mask[k,:,:,0], origin='lower', vmin=0, vmax=1) #, norm=LogNorm(vmin=0.001, vmax=1)
                    plt.axis('off')
                    plt.colorbar()

                    plt.subplot(1, 4, 4)
                    plt.title(title[3])
                    mask_list, x_peak_list, y_peak_list = self.detect_clusters(im = pred_mask[k,:,:,0], pmax=self.pmax, dmin=self.dmin, dmax=self.dmax)
                    if mask_list:
                        new_mask = np.zeros_like(pred_mask[k,:,:,0])
                        coords_in_patch = []
                        for i in range(len(mask_list)):
                            new_mask = new_mask + (np.ones_like(pred_mask[k,:,:,0]) - mask_list[i][:,:])
                            
                            plt.imshow(new_mask, origin='lower')
                            com = ndimage.measurements.center_of_mass(pred_mask[k,:,:,0]*(np.ones_like(pred_mask[k,:,:,0]) - mask_list[i]))
                            # plt.scatter(x_peak_list[i], y_peak_list[i], color='red')
                            plt.scatter(com[1], com[0], color='blue')
                            coords_in_patch.append(com)

                        final_masks[self.batch*n+k,:,:,0] = np.where(new_mask < 0, 0, new_mask)
                    else:
                        plt.imshow(np.zeros_like(pred_mask[k,:,:,0]), origin='lower')
                        final_masks[self.batch*n+k,:,:,0] = np.zeros_like(pred_mask[k][:,:,0])
                        coords_in_patch = []
                    pixel_coords.append(coords_in_patch)
                    plt.axis('off')

                    plt.savefig(self.output_path + 'figures/prediction_%s_%s_%s_f%s_%s/'%(self.model_name, self.loss_name, self.optimizer_name, self.freq, self.output_name) + 'prediction_%s_%s'%(n, k)  + '.png', bbox_inches='tight', transparent=False)
                    plt.show()
                    plt.close()
                else:
                    mask_list, x_peak_list, y_peak_list = self.detect_clusters(im = pred_mask[k,:,:,0], pmax=self.pmax, dmin=self.dmin, dmax=self.dmax)
                    if mask_list:
                        new_mask = np.zeros_like(pred_mask[k,:,:,0])
                        coords_in_patch = []
                        for i in range(len(mask_list)):
                            new_mask = new_mask + (np.ones_like(pred_mask[k,:,:,0]) - mask_list[i][:,:])
                            com = ndimage.measurements.center_of_mass(pred_mask[k,:,:,0]*(np.ones_like(pred_mask[k,:,:,0]) - mask_list[i]))
                            coords_in_patch.append(com)
                        final_masks[self.batch*n+k,:,:,0] = np.where(new_mask < 0, 0, new_mask)
                    else:
                        final_masks[self.batch*n+k,:,:,0] =  np.zeros_like(pred_mask[k][:,:,0])
                        coords_in_patch = []

                    pixel_coords.append(coords_in_patch)
            n += 1
            np.save(self.dataset_path + 'prediction_mask_%s_%s_%s_f%s_%s'%(self.model_name, self.loss_name, self.optimizer_name, self.freq, self.output_name) , predicted_masks)
            np.save(self.dataset_path + 'final_mask_%s_%s_%s_f%s_%s'%(self.model_name, self.loss_name, self.optimizer_name, self.freq, self.output_name) , final_masks)

        return pixel_coords

    def plot_metric(self, metric):

        try:
            data = pd.read_csv(self.path + "tf_saves/" + self.dataset + "/data_%s_%s_%s_f%s_"%(self.model_name, self.loss_name, self.optimizer_name, self.freq) + "%s.csv"%(self.output_name))
        except:
            data = pd.read_csv(self.path + "tf_saves/" + self.dataset + "/data_%s_l%s_o%s_f%s_"%(self.model_name, self.loss_name, self.optimizer_name, self.freq) + "%s.csv"%(self.output_name))
        
        train = data[metric]
        if metric != 'lr':
            val = data['val_' + metric]

        epochs = range(len(train))

        plt.figure()
        plt.plot(epochs, train, 'r', label='Training ' + metric)
        if metric != 'lr':
            plt.plot(epochs, val, 'bo', label='Validation ' + metric)
        plt.title('Training and Validation ' + metric)
        plt.xlabel('Epoch')
        plt.ylabel(metric + ' Value')
        if metric in ['f1', 'acc', 'iou', 'precision', 'recall', 'precision_1', 'recall_1', 'tp', 'tn', 'dsc']:
            plt.ylim([0, 1])
        plt.legend()
        plt.savefig(self.output_path + 'figures/prediction_%s_%s_%s_f%s_%s/'%(self.model_name, self.loss_name, self.optimizer_name, self.freq, self.output_name) + metric  + '.png', bbox_inches='tight', transparent=False)
        plt.show()
        plt.close()

    def add_sample_weights(self, image, label):
        # The weights for each class, with the constraint that:
        #     sum(class_weights) == 1.0
        class_weights = tf.constant([1.0, 10.0])#self.compute_class_weight()
        class_weights = class_weights/tf.reduce_sum(class_weights)

        # Create an image of `sample_weights` by using the label at each pixel as an 
        # index into the `class weights` .
        sample_weights = tf.gather(class_weights, indices=tf.cast(label, tf.int32))

        return image, label, sample_weights

    def compute_class_weight(self):
        output_train = np.load(self.dataset_path + 'label_train_pre_f%s_d%s_'%(self.freq, self.disk_radius) + self.dataset + '.npz')['arr_0']
        counter_1 = 0
        class_1 = 0
        class_0 = 0
        for i in range(len(output_train)):
            ones = np.sum(output_train[i,:,:,0])
            zeros = self.npix**2 - ones
            class_0 += zeros
            if ones == 0:
                pass
            else:
                counter_1 += 1
                class_1 += ones

        freq_1 = class_1/(counter_1 * self.npix**2)
        freq_0 = class_0/(len(output_train) * self.npix**2)
        median_freq = np.median([freq_0, freq_1])

        weight_0 = median_freq/freq_0
        weight_1 = median_freq/freq_1


        return tf.constant([weight_0, weight_1])

    def compute_pos_weight(self):
        output_train = np.load(self.dataset_path + 'label_train_pre_f%s_d%s_'%(self.freq, self.disk_radius) + self.dataset + '.npz')['arr_0']
        counter_1 = 0
        class_1 = 0
        class_0 = 0
        for i in range(len(output_train)):
            ones = np.sum(output_train[i,:,:,0])
            zeros = self.npix**2 - ones
            class_0 += zeros
            if ones == 0:
                pass
            else:
                counter_1 += 1
                class_1 += ones

        freq_1 = class_1/(counter_1 * self.npix**2)
        freq_0 = class_0/(len(output_train) * self.npix**2)
        median_freq = np.median([freq_0, freq_1])

        pos_weight = median_freq/freq_1

        return tf.constant(pos_weight)

    def weighted_binary_cross_entropy_loss(self, labels, logits):
        pos_weight = self.compute_pos_weight()
        loss = tf.nn.weighted_cross_entropy_with_logits(labels, logits, pos_weight = pos_weight)
        # loss = tf.reduce_mean(weighted_losses)
        return loss

    def objects_in_patch(self, p, catalog, plot=False):

        #------------------------------------------------------------------#
        # # # # # Create common catalog # # # # #
        #------------------------------------------------------------------#
        output_test = np.load(self.dataset_path + 'label_test_pre_f%s_d%s_'%(self.freq, self.disk_radius) + self.dataset + '.npz')['arr_0']

        if self.dataset == catalog:
            nthneighbor = 2
        else:
            nthneighbor = 1

        planck_z = pd.read_csv(self.path + 'catalogs/planck_z' + '.csv')
        planck_no_z = pd.read_csv(self.path + 'catalogs/planck_no-z' + '.csv')
        MCXC = pd.read_csv(self.path + 'catalogs/MCXC_no_planck' + '.csv')
        RM30 = pd.read_csv(self.path + 'catalogs/RM30_no_planck' + '.csv')
        RM50 = pd.read_csv(self.path + 'catalogs/RM50_no_planck' + '.csv')

        if catalog == 'planck_z':
            cluster_catalog = planck_z[['RA', 'DEC']].copy()
        elif catalog == 'planck_no-z':
            cluster_catalog = planck_no_z[['RA', 'DEC']].copy()
        elif catalog == 'MCXC':
            cluster_catalog = MCXC[['RA', 'DEC']].copy()
        elif catalog == 'RM30':
            cluster_catalog = RM30[['RA', 'DEC']].copy()
        elif catalog == 'RM50':
            cluster_catalog = RM50[['RA', 'DEC']].copy()

        if self.dataset == 'planck_z':
            coord_catalog = planck_z[['RA', 'DEC']].copy()
        elif self.dataset == 'planck_no-z':
            coord_catalog = planck_no_z[['RA', 'DEC']].copy()
        elif self.dataset == 'MCXC':
            coord_catalog = MCXC[['RA', 'DEC']].copy()
        elif self.dataset == 'RM30':
            coord_catalog = RM30[['RA', 'DEC']].copy()
        elif self.dataset == 'RM50':
            coord_catalog = RM50[['RA', 'DEC']].copy()

        #------------------------------------------------------------------#
        # # # # # Create ramdon coordinate translations # # # # #
        #------------------------------------------------------------------#

        input_size = len(coord_catalog['RA'].values)
        coords_ns = SkyCoord(ra=coord_catalog['RA'].values, dec=coord_catalog['DEC'].values, unit='deg')
        np.random.seed(p)
        random_coord_x = np.random.rand(1, input_size).flatten()
        np.random.seed(p)
        random_coord_y = np.random.rand(1, input_size).flatten()
        coords = SkyCoord(ra=coord_catalog['RA'].values -30*1.7/60 + (60*1.7/60)*random_coord_x,
                          dec=coord_catalog['DEC'].values -30*1.7/60 + (60*1.7/60)*random_coord_y, unit='deg')

        #------------------------------------------------------------------#
        # # # # # Check for potential neighbours # # # # #
        #------------------------------------------------------------------#

        scatalog = SkyCoord(ra=coord_catalog['RA'].values, dec=coord_catalog['DEC'].values, unit='deg')
        pcatalog = SkyCoord(ra=cluster_catalog['RA'].values, dec=cluster_catalog['DEC'].values, unit='deg')
        cluster_density = []
        coord_neighbours = []
        for i in range(input_size):
            k = nthneighbor
            idx, d2d, _ = match_coordinates_sky(scatalog, pcatalog, nthneighbor = k)
            ra_diff = np.abs(coord_catalog['RA'].values[i] - cluster_catalog['RA'].values[idx[i]])
            dec_diff = np.abs(coord_catalog['DEC'].values[i] - cluster_catalog['DEC'].values[idx[i]])
            if ra_diff < 1.76 and dec_diff < 1.76:
                neighb = [[cluster_catalog['RA'].values[idx[i]], cluster_catalog['DEC'].values[idx[i]]]]
                k += 1
                while ra_diff < 1.76 and dec_diff < 1.76:
                    idx, d2d, _ = match_coordinates_sky(scatalog, pcatalog, nthneighbor = k)
                    ra_diff = np.abs(coord_catalog['RA'].values[i] - cluster_catalog['RA'].values[idx[i]])
                    dec_diff = np.abs(coord_catalog['DEC'].values[i] - cluster_catalog['DEC'].values[idx[i]])
                    if ra_diff < 1.76 and dec_diff < 1.76:
                        neighb.append([cluster_catalog['RA'].values[idx[i]], cluster_catalog['DEC'].values[idx[i]]])
                        k += 1
            else:
                neighb = [[]]
            coord_neighbours.append(neighb)
            cluster_density.append(k-nthneighbor)

        #------------------------------------------------------------------#
        # # # # # Create patch & masks # # # # #
        #------------------------------------------------------------------#

        maps = self.maps

        cutsky = CutSky(maps, npix=self.npix, pixsize=self.pixsize, low_mem=False)

        total_objects = 0
        index = 0
        dataset_type = []
        df_coords = pd.DataFrame(columns=['RA', 'DEC'])

        hpi = ahp.HEALPix(nside=self.nside, order='ring', frame=ICRS())
        test_coords = [hpi.healpix_to_skycoord(healpix_index = 6), hpi.healpix_to_skycoord(healpix_index = 7)]

        for i, coord in enumerate(coords):
            count = 0


            if hp.ang2pix(self.nside, coord.galactic.l.degree, coord.galactic.b.degree, lonlat=True) == 6 or hp.ang2pix(self.nside, coord.galactic.l.degree, coord.galactic.b.degree, lonlat=True) == 7:
            # for h in range(len(test_coords)):
                # if np.abs(coord.ra.degree - test_coords[h].ra.degree) < 14.5 and np.abs(coord.dec.degree - test_coords[h].dec.degree) < 14.5:    
                # dataset_type.append('test')

                patch = cutsky.cut_fits(coord)
                HDU = patch[-1]['fits']
                wcs = WCS(HDU.header)
                h, w = self.npix, self.npix
                
                    
                if cluster_density[i] > 0:
                    if self.dataset == catalog:
                        x, y = wcs.world_to_pixel_values(coord_catalog['RA'].values[i], coord_catalog['DEC'].values[i])
                        center = [(x,y)]
                        ang_center = [(coord_catalog['RA'].values[i], coord_catalog['DEC'].values[i])]
                    else:
                        # x, y = wcs.world_to_pixel_values(cluster_catalog['RA'].values[i], cluster_catalog['DEC'].values[i])
                        # center = [(x,y)]
                        # ang_center = [(cluster_catalog['RA'].values[i], cluster_catalog['DEC'].values[i])]
                        center = []
                        ang_center = []
                    for j in range(cluster_density[i]):
                        x, y = wcs.world_to_pixel_values(coord_neighbours[i][j][0], coord_neighbours[i][j][1])
                        center.append((x, y))
                        ang_center.append((coord_neighbours[i][j][0], coord_neighbours[i][j][1]))
                    if center:
                        mask, count, ra, dec = MakeData.create_circular_mask(self, h, w, center=center, ang_center=ang_center, radius=self.disk_radius)
                        total_objects += count
                    else:
                        count = 0
                else:
                    if self.dataset == catalog:
                        x, y = wcs.world_to_pixel_values(coord_catalog['RA'].values[i], coord_catalog['DEC'].values[i])
                        center = [(x,y)]
                        ang_center = [(coord_catalog['RA'].values[i], coord_catalog['DEC'].values[i])]
                        mask, count, ra, dec = MakeData.create_circular_mask(self, h, w, center=center, ang_center=ang_center, radius=self.disk_radius)
                        total_objects += count
                    else:
                        mask = np.zeros((h,w))
                        count = 0          
                        ra, dec = [], []

                df_coords = pd.concat((df_coords, pd.DataFrame(data={'RA': ra, 'DEC': dec})))

                if plot:
                    plt.figure(figsize=(21, 7), tight_layout=True)
                    plt.subplot(1, 3, 1)
                    plt.title('y-map')
                    plt.imshow(HDU.data, origin='lower')
                    plt.axis('off')

                    plt.subplot(1, 3, 2)
                    plt.title('True mask')
                    plt.imshow(output_test[index,:,:,0], origin='lower')
                    plt.axis('off')

                    plt.subplot(1, 3, 3)
                    plt.title('%s'%count)
                    plt.imshow(mask, origin='lower')
                    # plt.scatter(center)
                    plt.axis('off')

                    GenerateFiles.make_directory(self, path_to_file = self.output_path + 'figures/' + 'total_mask_' + catalog)
                    plt.savefig(self.output_path + 'figures/total_mask_%s/'%(catalog) + 'mock_mask_%s'%(index)  + '.png', bbox_inches='tight', transparent=False)
                    plt.show()
                    plt.close()

                index += 1

            # elif hp.ang2pix(self.nside, coord.galactic.l.degree, coord.galactic.b.degree, lonlat=True) == 9 or hp.ang2pix(self.nside, coord.galactic.l.degree, coord.galactic.b.degree, lonlat=True) == 38 or hp.ang2pix(self.nside, coord.galactic.l.degree, coord.galactic.b.degree, lonlat=True) == 41:
            #     dataset_type.append('val')
            # else:
            #     dataset_type.append('train')

        df_coords = self.remove_duplicates(df_coords, tol=2)

        print('\n')
        print(total_objects)
        print('\n')

        return len(df_coords), total_objects
