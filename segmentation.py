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

# import logging
# cs_logger = logging.getLogger('cutsky')
# cs_logger.setLevel(logging.WARNING)
# cs_logger.propagate = False
# hpproj_logger = logging.getLogger('hpproj')
# hpproj_logger.setLevel(logging.WARNING)
# mpl_logger = logging.getLogger('matplotlib')
# mpl_logger.setLevel(logging.WARNING)


class CNNSegmentation(MakeData):

    def __init__(self, dataset, bands, npix, n_labels, cold_cores, planck_path, milca_path, model, range_comp, epochs, batch, lr, patience, loss, optimizer, loops, size=64, disk_radius = None, delta=0.4, gamma=0.75, drop_out=False, output_path = None):
        super().__init__(dataset, npix, loops, planck_path, milca_path, disk_radius=disk_radius, output_path=output_path)
        self.range_comp = range_comp
        self.loops = loops
        self.size = size
        self.drop_out = drop_out
        self.epochs = epochs
        self.batch = batch
        self.lr = lr 
        self.patience = patience
        self.pmax=0.6
        self.dmin=2
        self.dmax=60
        self.bands = bands
        self.delta = delta
        self.gamma = gamma
        self.cold_cores = cold_cores

        self.freq = 0
        self.planck_freq = 0
        if '100GHz' in  bands:
            self.freq += 2
            self.planck_freq += 2
        if '143GHz' in bands:
            self.freq += 4
            self.planck_freq += 4
        if '217GHz' in bands:
            self.freq += 8
            self.planck_freq += 8
        if '353GHz' in bands:
            self.freq += 16
            self.planck_freq += 16
        if '545GHz' in bands:
            self.freq += 32
            self.planck_freq += 32
        if '857GHz' in bands:
            self.freq += 64
            self.planck_freq += 64
        if 'y-map' in bands:
            self.freq += 128
        if 'CO' in bands:
            self.freq += 256
        if 'p-noise' in bands:
            self.freq += 512

        self.output_name = 'l%s_e%s_b%s_lr%s_p%s_d%s'%(n_labels, epochs, batch, lr, patience, disk_radius)

        optimizers_dict = {'sgd': SGD(lr=self.lr, momentum=0.9), 'adam': Adam(learning_rate=self.lr), 'rmsprop': tf.keras.optimizers.RMSprop(learning_rate=self.lr, rho=0.9, momentum=0.5)}
        self.optimizer =  optimizers_dict[optimizer]
        self.optimizer_name =  optimizer
        losses_dict = {'categorical_crossentropy': tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), 'binary_crossentropy': 'binary_crossentropy', 'weighted_binary_crossentropy': 'binary_crossentropy', 
                       'tversky_loss': losses.tversky_loss(delta=self.delta), 'focal_tversky_loss': losses.focal_tversky_loss(delta=self.delta, gamma=self.gamma), 'dice_loss': losses.dice_loss, 
                       'modified_tversky_loss': losses.modified_tversky_loss,
                       'combo_loss': losses.combo_loss(alpha=0.5,beta=0.5), 'focal_dice_loss': losses.focal_dice_loss(delta=0.5, gamma_fd=self.gamma), 
                       'focal_loss': losses.focal_loss(alpha=None, beta=None, gamma_f=2.), 'mixed_focal_loss': losses.mixed_focal_loss(weight=None, alpha=None, beta=None, delta=0.7, gamma_f=2.,gamma_fd=0.75)}
        self.loss = losses_dict[loss]
        self.loss_name = loss
        input_size = (self.npix, self.npix, len(self.bands))
        # filter_num = [8, 16, 32, 64, 128]
        filter_num = [64, 128, 256, 512, 1024]
        self.n_labels = n_labels
        dilation_num = [1, 3, 15, 31]
        filter_num_down = [64, 128, 256, 512]#, 1024]

        #               'vnet': tf_models.vnet_2d(input_size, filter_num, n_labels, res_num_ini=1, res_num_max=3, 
        #                     activation='ReLU', output_activation='Softmax', batch_norm=False, pool=True, unpool=True, name='vnet'),

        if model == 'unet':
            self.model = tf_models.unet_2d(input_size, filter_num, n_labels, stack_num_down=2, stack_num_up=2,
                            activation='ReLU', output_activation='Sigmoid', batch_norm=False, pool=True, unpool=True, 
                            backbone=None, weights=None, freeze_backbone=True, freeze_batch_norm=True, name='unet')
        elif model == 'attn_unet':
            self.model = tf_models.att_unet_2d(input_size, filter_num, self.n_labels, stack_num_down=2, stack_num_up=2, 
                            activation='ReLU', atten_activation='ReLU', attention='add', output_activation='Sigmoid', 
                            batch_norm=True, weights=None, pool=False, unpool=False, freeze_batch_norm=True, name='attunet')
        elif model == 'r2u_net':
            self.model = tf_models.r2_unet_2d(input_size, filter_num, self.n_labels, 
                            stack_num_down=2, stack_num_up=2, recur_num=2,
                            activation='ReLU', output_activation='Sigmoid', 
                            batch_norm=False, pool=True, unpool=True, name='r2_unet')
        elif model == 'unet_plus':
            self.model = tf_models.unet_plus_2d(input_size, filter_num, self.n_labels, stack_num_down=2, stack_num_up=2,
                            activation='ReLU', output_activation='Sigmoid', batch_norm=False, pool=True, unpool=True, deep_supervision=False, 
                            backbone=None, weights=None, freeze_backbone=True, freeze_batch_norm=True, name='xnet')
        elif model == 'resunet_a':
            self.model = tf_models.resunet_a_2d(input_size, filter_num, dilation_num, self.n_labels,
                            aspp_num_down=256, aspp_num_up=128, activation='ReLU', output_activation='Sigmoid', 
                            batch_norm=True, pool=True, unpool=True, name='resunet')
        elif model == 'u2net':
            self.model = tf_models.u2net_2d(input_size, self.n_labels, filter_num_down, filter_num_up='auto', filter_mid_num_down='auto', filter_mid_num_up='auto', 
                            filter_4f_num='auto', filter_4f_mid_num='auto', activation='ReLU', output_activation='Sigmoid', 
                            batch_norm=False, pool=True, unpool=True, deep_supervision=False, name='u2net')
        elif model == 'unet_3plus':
            self.model = tf_models.unet_3plus_2d(input_size, self.n_labels, filter_num_down, filter_num_skip='auto', filter_num_aggregate='auto', 
                            stack_num_down=2, stack_num_up=1, activation='ReLU', output_activation='Sigmoid',
                            batch_norm=False, pool=True, unpool=True, deep_supervision=False, 
                            backbone=None, weights=None, freeze_backbone=True, freeze_batch_norm=True, name='unet3plus')
        self.model_name = model

        if loss == 'tversky_loss':
            self.pre_output_name = 'f%s_s%s_c%s_%s_%s_%s_%s'%(self.freq, self.npix, int(self.cold_cores), self.model_name, self.loss_name, self.delta, self.optimizer_name)
        elif loss == 'focal_tversky_loss':
            self.pre_output_name = 'f%s_s%s_c%s_%s_%s_%s_%s_%s'%(self.freq, self.npix, int(self.cold_cores), self.model_name, self.loss_name, self.delta, self.gamma, self.optimizer_name)
        else:
            self.pre_output_name = 'f%s_s%s_c%s_%s_%s_%s'%(self.freq, self.npix, int(self.cold_cores), self.model_name, self.loss_name, self.optimizer_name)


    def dec_to_bin(self, n):
        return str(bin(n)[2:])

    def bin_to_dec(self, n):
        return int(n,2)

    def remove_intersection(self, lst1, lst2):
        lst3 = [value for value in lst1 if value not in lst2]
        return lst3

    def missing_frequencies(self):

        freq_list = [2,4,8,16,32,64,128,256,512]

        bin_freq = self.dec_to_bin(self.freq)
        
        n = len(bin_freq)
        freq_num = []
        for i,string in enumerate(bin_freq):
            if self.bin_to_dec(string+'0'*(n-1-i)) != 0:
                freq_num.append(self.bin_to_dec(string+'0'*(n-1-i)))

        freq_list_minus_freq_num = self.remove_intersection(freq_list, freq_num)
        index_to_remove = []
        for freq in freq_list_minus_freq_num:
            if freq == 2:
                index_to_remove.append(0)
            if freq == 4:
                index_to_remove.append(1)
            if freq == 8:
                index_to_remove.append(2)
            if freq == 16:
                index_to_remove.append(3)
            if freq == 32:
                index_to_remove.append(4)
            if freq == 64:
                index_to_remove.append(5)
            if freq == 128:
                index_to_remove.append(6)
            if freq == 256:
                index_to_remove.append(7)
            if freq == 512:
                index_to_remove.append(8)

        return index_to_remove

    def remove_index(self, input):
        index_to_remove = self.missing_frequencies()
        return np.delete(input, index_to_remove, axis=3)

    def remove_input_index(self, input, cluster):
        #not gona work, needs to account for validation
        if cluster:
            index_to_remove = np.arange(0,1072,1)
            for loop in range(self.loops - 1):
                index_to_remove = np.concatenate((index_to_remove, np.arange( (1072 + 1049)*loop,(1072 + 1049)*loop + 1072 ,1)) )
            


    def npy_to_tfdata(self, region, batch_size=10, buffer_size=1000, only_train=True, only_test=False):
        if only_train:
            if self.range_comp:
                try:
                    input_train = np.load(self.dataset_path + 'input_train_pre_r%s_f%s_s%s_c%s_'%(region, 1022, self.npix, int(self.cold_cores)) + self.dataset + '.npz')['arr_0']
                except:
                    input_train = np.load(self.dataset_path + 'input_train_pre_r%s_f%s_'%(region, 1022) + self.dataset + '.npz')['arr_0']
            else:
                input_train = np.load(self.dataset_path + 'input_train_pre_r%s_f%s_r0_s%s_c%s_'%(region, 1022, self.npix, int(self.cold_cores)) + self.dataset + '.npz')['arr_0']
            input_train = self.remove_index(input_train)

            if self.range_comp:
                try:
                    input_val = np.load(self.dataset_path + 'input_val_pre_r%s_f%s_s%s_c%s_'%(region, 1022, self.npix, int(self.cold_cores)) + self.dataset + '.npz')['arr_0']
                except:
                    input_val = np.load(self.dataset_path + 'input_val_pre_r%s_f%s_'%(region, 1022) + self.dataset + '.npz')['arr_0']
            else:
                input_val = np.load(self.dataset_path + 'input_val_pre_r%s_f%s_r0_s%s_c%s_'%(region, 1022, self.npix, int(self.cold_cores)) + self.dataset + '.npz')['arr_0']
            input_val = self.remove_index(input_val)

            try:
                output_train = np.load(self.dataset_path + 'label_train_pre_r%s_f%s_d%s_s%s_c%s_'%(region, 1022, self.disk_radius, self.npix, int(self.cold_cores)) + self.dataset + '.npz')['arr_0']
            except:
                output_train = np.load(self.dataset_path + 'label_train_pre_r%s_f%s_d%s_'%(region, 1022, self.disk_radius) + self.dataset + '.npz')['arr_0']
            label_train = np.ndarray((np.shape(output_train)[0], self.npix, self.npix, self.n_labels))
            label_train[:,:,:,0] = output_train[:,:,:,0].astype(int)
            try:
                label_train[:,:,:,1] = output_train[:,:,:,1].astype(int)
            except:
                pass

            try:
                output_val = np.load(self.dataset_path + 'label_val_pre_r%s_f%s_d%s_s%s_c%s_'%(region, 1022, self.disk_radius, self.npix, int(self.cold_cores)) + self.dataset + '.npz')['arr_0']
            except:
                output_val = np.load(self.dataset_path + 'label_val_pre_r%s_f%s_d%s_'%(region, 1022, self.disk_radius) + self.dataset + '.npz')['arr_0']
            label_val = np.ndarray((np.shape(output_val)[0], self.npix, self.npix, self.n_labels))
            label_val[:,:,:,0] = output_val[:,:,:,0].astype(int)
            try:
                label_val[:,:,:,1] = output_val[:,:,:,1].astype(int)
            except:
                pass

            train_dataset = tf.data.Dataset.from_tensor_slices((input_train, label_train))
            val_dataset = tf.data.Dataset.from_tensor_slices((input_val, label_val))

            train_dataset = train_dataset.shuffle(buffer_size).batch(batch_size).repeat()
            train_dataset = train_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
            val_dataset = val_dataset.shuffle(buffer_size).batch(batch_size).repeat()
            val_dataset = val_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

            return train_dataset, val_dataset

        if only_test:
            if self.range_comp:
                try:
                    input_test = np.load(self.dataset_path + 'input_test_pre_r%s_f%s_s%s_c%s_'%(region, 1022, self.npix, int(self.cold_cores)) + self.dataset + '.npz')['arr_0']
                except:
                    input_test = np.load(self.dataset_path + 'input_test_pre_r%s_f%s_'%(region, 1022) + self.dataset + '.npz')['arr_0']
            else:
                input_test = np.load(self.dataset_path + 'input_test_pre_r%s_f%s_r0_'%(region, 1022) + self.dataset + '.npz')['arr_0']
            input_test = self.remove_index(input_test)

            try:
                output_test = np.load(self.dataset_path + 'label_test_pre_r%s_f%s_d%s_s%s_c%s_'%(region, 1022, self.disk_radius, self.npix, int(self.cold_cores)) + self.dataset + '.npz')['arr_0']
            except:
                output_test = np.load(self.dataset_path + 'label_test_pre_r%s_f%s_d%s_'%(region, 1022, self.disk_radius) + self.dataset + '.npz')['arr_0']
            label_test = np.ndarray((np.shape(output_test)[0], self.npix, self.npix, self.n_labels))
            label_test[:,:,:,0] = output_test[:,:,:,0].astype(int)
            try:
                label_test[:,:,:,1] = output_test[:,:,:,1].astype(int)
            except:
                pass

            test_dataset = tf.data.Dataset.from_tensor_slices((input_test, label_test))            
            test_dataset = test_dataset.batch(batch_size)


            return test_dataset


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

    def train_model(self, regions):
        for region in regions:
            tf.keras.backend.clear_session()
            # train_size = 9300
            # val_size = 800
            # if self.cold_cores:
            #     train_size = train_size*2
            #     val_size = val_size*2
            ############ COULD BE DONE BETTER ############
            try:
                train_size = int(len(np.load(self.dataset_path + 'label_train_pre_r%s_f%s_d%s_s%s_c%s_'%(region, 1022, self.disk_radius, self.npix, int(self.cold_cores)) + self.dataset + '.npz')['arr_0']))
                val_size = int(len(np.load(self.dataset_path + 'label_val_pre_r%s_f%s_d%s_s%s_c%s_'%(region, 1022, self.disk_radius, self.npix, int(self.cold_cores)) + self.dataset + '.npz')['arr_0']))
            except:
                train_size = int(len(np.load(self.dataset_path + 'label_train_pre_r%s_f%s_d%s_'%(region, 1022, self.disk_radius) + self.dataset + '.npz')['arr_0']))
                val_size = int(len(np.load(self.dataset_path + 'label_val_pre_r%s_f%s_d%s_'%(region, 1022, self.disk_radius) + self.dataset + '.npz')['arr_0']))
            ##############################################
            train_dataset, valid_dataset = self.npy_to_tfdata(region, batch_size=self.batch, buffer_size=1000, only_train=True, only_test=False)

            print('\n')
            print('---------------------------------------')
            print('[REGION] %s'%region)
            print('[TRAINING SIZE] %s'%train_size)
            print('[VALIDATION SIZE] %s'%val_size)
            print('---------------------------------------')
            print('\n')

            callbacks = [
                    ModelCheckpoint(monitor='val_loss', filepath=self.path + "tf_saves/" + self.dataset + "/model_r%s_%s_%s"%(region, self.pre_output_name, self.output_name) + ".h5", save_best_only=True),
                    ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=self.patience),
                    CSVLogger(self.path + "tf_saves/" + self.dataset + "/data_r%s_%s_%s"%(region, self.pre_output_name, self.output_name) + ".csv"),
                    TensorBoard(),
                    EarlyStopping(monitor='val_loss', patience=self.patience, restore_best_weights=True)
                ]
           

            model  = self.model
            metrics = [losses.dice_coefficient, tf.keras.metrics.Recall(), tf.keras.metrics.Precision()] #losses.dsc, 
            model.compile(loss=self.loss, optimizer=self.optimizer, metrics=metrics, run_eagerly=True) 

            if self.loss_name == 'weighted_binary_crossentropy':
                model_history = model.fit(train_dataset.map(self.add_sample_weights),
                    validation_data=valid_dataset,
                    epochs=self.epochs,
                    steps_per_epoch = train_size//(self.batch),
                    validation_steps= val_size//(self.batch),
                    callbacks=callbacks)
            else:
                model_history = model.fit(train_dataset,
                    validation_data=valid_dataset,
                    epochs=self.epochs,
                    steps_per_epoch = train_size//(self.batch),
                    validation_steps= val_size//(self.batch),
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
            if len(all_neighbours) >= dmin:
                mask_list.append(individual_mask)
                x_peak_list.append(center[1])
                y_peak_list.append(center[0])

        return mask_list, x_peak_list, y_peak_list

    def match_detections_against_catalogs(self, region, pixel_coords, precision, recall, dsc, plot=True):

        maps = self.maps

        planck_z = pd.read_csv(self.path + 'catalogs/planck_z' + '.csv')
        planck_no_z = pd.read_csv(self.path + 'catalogs/planck_no-z' + '.csv')
        MCXC = pd.read_csv(self.path + 'catalogs/MCXC_no_planck' + '.csv')
        RM50 = pd.read_csv(self.path + 'catalogs/RM50_no_planck' + '.csv')
        RM30 = pd.read_csv(self.path + 'catalogs/RM30_no_planck' + '.csv')
        ACT = pd.read_csv(self.path + 'catalogs/ACT_no_planck' + '.csv')
        SPT = pd.read_csv(self.path + 'catalogs/SPT_no_planck' + '.csv')


        x_left, x_right, y_up, y_down = self.test_regions[region]

        planck_z.query("GLAT > %s"%y_down, inplace=True)
        planck_z.query("GLAT < %s"%y_up, inplace=True)
        planck_z.query("GLON > %s"%x_left, inplace=True)
        planck_z.query("GLON < %s"%x_right, inplace=True)

        planck_no_z.query("GLAT > %s"%y_down, inplace=True)
        planck_no_z.query("GLAT < %s"%y_up, inplace=True)
        planck_no_z.query("GLON > %s"%x_left, inplace=True)
        planck_no_z.query("GLON < %s"%x_right, inplace=True)

        MCXC.query("GLAT > %s"%y_down, inplace=True)
        MCXC.query("GLAT < %s"%y_up, inplace=True)
        MCXC.query("GLON > %s"%x_left, inplace=True)
        MCXC.query("GLON < %s"%x_right, inplace=True)

        try:
            RM50.query("GLAT > %s"%y_down, inplace=True)
            RM50.query("GLAT < %s"%y_up, inplace=True)
            RM50.query("GLON > %s"%x_left, inplace=True)
            RM50.query("GLON < %s"%x_right, inplace=True)
        except:
            pass

        try:
            RM30.query("GLAT > %s"%y_down, inplace=True)
            RM30.query("GLAT < %s"%y_up, inplace=True)
            RM30.query("GLON > %s"%x_left, inplace=True)
            RM30.query("GLON < %s"%x_right, inplace=True)
        except:
            pass

        try:
            ACT.query("GLAT > %s"%y_down, inplace=True)
            ACT.query("GLAT < %s"%y_up, inplace=True)
            ACT.query("GLON > %s"%x_left, inplace=True)
            ACT.query("GLON < %s"%x_right, inplace=True)
        except:
            pass
        
        try:
            SPT.query("GLAT > %s"%y_down, inplace=True)
            SPT.query("GLAT < %s"%y_up, inplace=True)
            SPT.query("GLON > %s"%x_left, inplace=True)
            SPT.query("GLON < %s"%x_right, inplace=True)
        except:
            pass

 
        try:
            ACT = MakeData.remove_duplicates_on_lonlat(self, ACT, df_with_dup=MCXC, tol=7)
        except:
            ACT = pd.DataFrame()
       
        try:
            SPT = MakeData.remove_duplicates_on_lonlat(self, SPT, df_with_dup=MCXC, tol=7)
        except:
            SPT = pd.DataFrame()
        if not SPT.empty and not ACT.empty:
            SPT = MakeData.remove_duplicates_on_lonlat(self, SPT, df_with_dup=ACT, tol=7)

        try:
            RM50 = MakeData.remove_duplicates_on_lonlat(self, RM50, df_with_dup=MCXC, tol=7)
        except:
            RM50 = pd.DataFrame()
        if not RM50.empty and not ACT.empty:
            RM50 = MakeData.remove_duplicates_on_lonlat(self, RM50, df_with_dup=ACT, tol=5)
        if not RM50.empty and not SPT.empty:
            RM50 = MakeData.remove_duplicates_on_lonlat(self, RM50, df_with_dup=SPT, tol=5)

        try:
            RM30 = MakeData.remove_duplicates_on_lonlat(self, RM30, df_with_dup=MCXC, tol=7)
        except:
            RM30 = pd.DataFrame()
        if not RM30.empty and not ACT.empty:
            RM30 = MakeData.remove_duplicates_on_lonlat(self, RM30, df_with_dup=ACT, tol=5)
        if not RM30.empty and not SPT.empty:
            RM30 = MakeData.remove_duplicates_on_lonlat(self, RM30, df_with_dup=SPT, tol=5)
        if not RM30.empty and not RM50.empty:
            RM30 = MakeData.remove_duplicates_on_lonlat(self, RM30, df_with_dup=RM50, tol=2)    
        

        false_catalog = pd.read_csv(self.path + 'catalogs/False_SZ_catalog_f%s.csv'%self.planck_freq)
        false_catalog.query("GLAT > %s"%y_down, inplace=True)
        false_catalog.query("GLAT < %s"%y_up, inplace=True)
        false_catalog.query("GLON > %s"%x_left, inplace=True)
        false_catalog.query("GLON < %s"%x_right, inplace=True)

        CS = pd.read_csv(self.path + 'catalogs/CS_f%s.csv'%self.planck_freq)
        CS.query("GLAT > %s"%y_down, inplace=True)
        CS.query("GLAT < %s"%y_up, inplace=True)
        CS.query("GLON > %s"%x_left, inplace=True)
        CS.query("GLON < %s"%x_right, inplace=True)
        
        CC = pd.read_csv(self.path + 'catalogs/PGCC.csv')
        CC.query("GLAT > %s"%y_down, inplace=True)
        CC.query("GLAT < %s"%y_up, inplace=True)
        CC.query("GLON > %s"%x_left, inplace=True)
        CC.query("GLON < %s"%x_right, inplace=True)

        if not CC.empty and not CS.empty:
            CS = MakeData.remove_duplicates_on_lonlat(self, CS, df_with_dup=CC, tol=5)   
            CS = MakeData.remove_duplicates_on_lonlat(self, CS, tol=2, with_itself=True)  

        cutsky = CutSky(maps, npix=self.npix, pixsize=self.pixsize, low_mem=False)

        test_coords, test_catalog = MakeData.test_coords(self, x_left, x_right, y_up, y_down)

        # dataset_type_test = np.load(self.dataset_path + 'type_test_r%s_f%s_'%(region, 1022) + self.dataset + '.npz')['arr_0']
        # type_count_test = Counter(dataset_type_test)
        # print("[Inputs] training: {:.0f}, validation: {:.0f}, test: {:.0f}".format(type_count_test['train'], type_count_test['val'], type_count_test['test']))



        assert len(pixel_coords) == len(test_coords)

        l_list, b_list = [], []
        for i, coord in enumerate(test_coords):
            patch = cutsky.cut_fits(coord)
            HDU = patch[-1]['fits']
            wcs = WCS(HDU.header)
            for j in range(len(pixel_coords[i])):
                l, b = wcs.pixel_to_world_values(pixel_coords[i][j][1], pixel_coords[i][j][0])
                l_list.append(l)
                b_list.append(b)

        df_detections = pd.DataFrame(data={'GLON': l_list, 'GLAT': b_list})

        # df_duplicates = self.return_dup_coords(region, df_detections, plot=True)
        if not planck_z.empty:
            df_planck_z_detections, planck_z_duplicate_detections = self.match_with_catalog(region, df_detections, planck_z, tol=7, output_name='planck_z', plot=plot)
        else:
            df_planck_z_detections = pd.DataFrame(columns = ['GLON', 'GLAT'])
        if not planck_no_z.empty:
            df_planck_no_z_detections, planck_no_z_duplicate_detections = self.match_with_catalog(region, df_detections, planck_no_z, tol=7, output_name='planck_no-z', plot=plot)
        else:
            df_planck_no_z_detections = pd.DataFrame(columns = ['GLON', 'GLAT'])
        if not MCXC.empty:
            df_MCXC_detections, MCXC_duplicate_detections = self.match_with_catalog(region, df_detections, MCXC, tol=7, output_name='MCXC', plot=plot)
        else:
            df_MCXC_detections = pd.DataFrame(columns = ['GLON', 'GLAT'])
        if not RM50.empty:
            df_RM50_detections, RM50_duplicate_detections = self.match_with_catalog(region, df_detections, RM50, tol=5, output_name='RM50', plot=plot)
        else:
            df_RM50_detections = pd.DataFrame(columns = ['GLON', 'GLAT'])
        if not RM30.empty:
            df_RM30_detections, RM30_duplicate_detections = self.match_with_catalog(region, df_detections, RM30, tol=5, output_name='RM30', plot=plot)
        else:
            df_RM30_detections = pd.DataFrame(columns = ['GLON', 'GLAT'])
        if not ACT.empty:
            df_ACT_detections, ACT_duplicate_detections = self.match_with_catalog(region, df_detections, ACT, tol=5, output_name='ACT', plot=plot)
        else:
            df_ACT_detections = pd.DataFrame(columns = ['GLON', 'GLAT'])
        if not SPT.empty:
            df_SPT_detections, SPT_duplicate_detections = self.match_with_catalog(region, df_detections, SPT, tol=5, output_name='SPT', plot=plot)
        else:
            df_SPT_detections = pd.DataFrame(columns = ['GLON', 'GLAT'])

        df_false_clusters_detections, false_clusters_duplicate_detections = self.match_with_catalog(region, df_detections, false_catalog, tol=5, output_name='false_clusters', plot=plot)
        df_cs, _ = self.match_with_catalog(region, df_detections, CS, tol=5, output_name='compact_sources', plot=plot)
        df_cc, _ = self.match_with_catalog(region, df_detections, CC, tol=5, output_name='cold_cores', plot=plot)

        if not planck_z.empty:
            df_unknown = MakeData.remove_duplicates_on_lonlat(self, df_detections, df_with_dup=planck_z, tol=7)
        if not planck_no_z.empty:
            df_unknown = MakeData.remove_duplicates_on_lonlat(self, df_unknown, df_with_dup=planck_no_z, tol=7)
        if not MCXC.empty:
            df_unknown = MakeData.remove_duplicates_on_lonlat(self, df_unknown, df_with_dup=MCXC, tol=7)
        if not RM50.empty:
            df_unknown = MakeData.remove_duplicates_on_lonlat(self, df_unknown, df_with_dup=RM50, tol=5)
        if not RM30.empty:
            df_unknown = MakeData.remove_duplicates_on_lonlat(self, df_unknown, df_with_dup=RM30, tol=5)
        if not ACT.empty:
            df_unknown = MakeData.remove_duplicates_on_lonlat(self, df_unknown, df_with_dup=ACT, tol=5)
        if not SPT.empty:
            df_unknown = MakeData.remove_duplicates_on_lonlat(self, df_unknown, df_with_dup=SPT, tol=5)
        unknown_detections = MakeData.remove_duplicates_on_lonlat(self, df_unknown, df_with_dup=false_catalog, tol=5)

        unknown_detections.to_csv(self.output_path + 'files/' + 'r%s/'%region + 'unknown_detections_r%s_%s_%s'%(region, self.pre_output_name, self.output_name), index=False)

        try:
            total = pd.read_csv(self.output_path + 'files/' + 'r%s/'%region + 'prediction_r%s_'%(region) + 'f%s_s%s_c%s_%s_%s_%s_'%(self.freq, self.npix, int(self.cold_cores), self.model_name, self.loss_name, self.optimizer_name) + '%s'%(self.output_name) + ".csv")
            npz = total['planck z'].values[0]
            npnz = total['planck no z'].values[0]
            nmcxc = total['MCXC'].values[0]
            nrm50 = total['RM50'].values[0]
            nrm30 = total['RM30'].values[0]
            nact = total['ACT'].values[0]
            nspt = total['SPT'].values[0]
            nfc = total['False detections'].values[0]
            ncc = total['Cold cores'].values[0]
            ncs = total['Compact sources'].values[0]
        except:
            npz = self.clusters_in_patch(region, catalog=planck_z)
            npnz = self.clusters_in_patch(region, catalog=planck_no_z)
            if npnz == 0:
                npnz = 0.1
            nmcxc = self.clusters_in_patch(region, catalog=MCXC)
            if nmcxc == 0:
                nmcxc = 0.1
            if not RM50.empty:
                if len(RM50) < 5:
                    nrm50 = len(RM50)
                else:
                    nrm50 = self.clusters_in_patch(region, catalog=RM50)
                if nrm50 == 0:
                    nrm50 = 0.1
            else:
                nrm50 = 0.1
            if not RM30.empty:
                if len(RM30) < 5:
                    nrm30 = len(RM30)
                else:
                    nrm30 = self.clusters_in_patch(region, catalog=RM30)
                if nrm30 == 0:
                    nrm30 = 0.1
            else:
                nrm30 = 0.1
            if not ACT.empty:
                if len(ACT) < 5:
                    nact = len(ACT)
                else:
                    nact = self.clusters_in_patch(region, catalog=ACT)
                if nact == 0:
                    nact = 0.1
            else:
                nact = 0.1
            if not SPT.empty:
                if len(SPT) < 5:
                    nspt = len(SPT)
                else:
                    nspt = self.clusters_in_patch(region, catalog=SPT)
                if nspt == 0:
                    nspt = 0.1
            else:
                nspt = 0.1
            nfc = self.false_clusters_in_patch(region, false_catalog=false_catalog)
            ncs = self.false_clusters_in_patch(region, false_catalog=CS)
            if not CC.empty:
                ncc = self.false_clusters_in_patch(region, false_catalog=CC)
            else:
                ncc = 0.1
        
        file = open(self.output_path + 'figures/' + 'r%s/'%region + 'prediction_r%s_%s_%s'%(region, self.pre_output_name, self.output_name) + "/results.txt","a")
        
        L = ['planck_z: %s/%s'%(len(df_planck_z_detections), npz)+ ' {:.2f}%'.format(100*len(df_planck_z_detections)/npz) + " \n", 
              'planck_no-z: %s/%s'%(len(df_planck_no_z_detections), npnz)+ ' {:.2f}%'.format(100*len(df_planck_no_z_detections)/npnz) +  " \n", 
              'MCXC: %s/%s'%(len(df_MCXC_detections), nmcxc)+ ' {:.2f}%'.format(100*len(df_MCXC_detections)/nmcxc) + " \n", 
              'ACT: %s/%s'%(len(df_ACT_detections), nact)+ ' {:.2f}%'.format(100*len(df_ACT_detections)/nact) + " \n", 
              'SPT: %s/%s'%(len(df_SPT_detections), nspt)+ ' {:.2f}%'.format(100*len(df_SPT_detections)/nspt) + " \n", 
              'RM50: %s/%s'%(len(df_RM50_detections), nrm50)+ ' {:.2f}%'.format(100*len(df_RM50_detections)/nrm50) + " \n", 
              'RM30: %s/%s'%(len(df_RM30_detections), nrm30)+ ' {:.2f}%'.format(100*len(df_RM30_detections)/nrm30) + " \n", 
              'Cold cores: %s/%s'%(len(df_cc), ncc)+ ' {:.2f}%'.format(100*len(df_cc)/ncc) + " \n",
              'Compact sources: %s/%s'%(len(df_cs), ncs)+ ' {:.2f}%'.format(100*len(df_cs)/ncs) + " \n",
              'False detections: %s/%s'%(len(df_false_clusters_detections), nfc)+ ' {:.2f}%'.format(100*len(df_false_clusters_detections)/nfc) + " \n",
              'Unknown: %s'%(len(unknown_detections))] 

        
        try:
            if self.loss_name == 'focal_tversky_loss':
                _ = pd.read_csv(self.output_path + 'files/' + 'r%s/'%region + 'prediction_r%s_'%(region) + 'f%s_s%s_c%s_%s_%s_%s_%s_'%(self.freq, self.npix, int(self.cold_cores), self.model_name, self.loss_name, self.gamma, self.optimizer_name) + '%s'%(self.output_name) + ".csv")
            else:
                _ = pd.read_csv(self.output_path + 'files/' + 'r%s/'%region + 'prediction_r%s_'%(region) + 'f%s_s%s_c%s_%s_%s_%s_'%(self.freq, self.npix, int(self.cold_cores), self.model_name, self.loss_name, self.optimizer_name) + '%s'%(self.output_name) + ".csv")
            ## if the above file does not exist, try will fail and the except will create the file with a header
            results = pd.DataFrame(data = {'model': [self.model_name], 'loss': [self.loss_name], 'delta': [self.delta], 'gamma': [self.gamma], 'precision': precision[0], 'recall': recall[0],
                'dsc': dsc[0], 'planck z': [len(df_planck_z_detections)/npz], 'planck no z':[len(df_planck_no_z_detections)/npnz], 'MCXC': [len(df_MCXC_detections)/nmcxc], 
                'RM30': [len(df_RM30_detections)/nrm30], 'RM50': [len(df_RM50_detections)/nrm50], 'ACT': [len(df_ACT_detections)/nact], 'SPT': [len(df_SPT_detections)/nspt], 
                'False detections':[len(df_false_clusters_detections)/nfc], 'Cold cores': [len(df_cc)/ncc], 'Compact sources': [len(df_cs)/ncs], 'Unknown': [len(unknown_detections)],
                'total density': [(len(df_detections))], 
                'planck z density': [(len(df_planck_z_detections))], 
                'density without planck z': [(len(df_detections)-len(df_planck_z_detections))], 
                'detection ratio without planck z': [(len(df_detections)-len(df_planck_z_detections))]})

            if self.loss_name == 'focal_tversky_loss':
                results.to_csv(self.output_path + 'files/' + 'r%s/'%region + 'prediction_r%s_'%(region) + 'f%s_s%s_c%s_%s_%s_%s_%s_'%(self.freq, self.npix, int(self.cold_cores), self.model_name, self.loss_name, self.gamma, self.optimizer_name) + '%s'%(self.output_name) + ".csv", mode='a', header=False, index=False)
                results = pd.read_csv(self.output_path + 'files/' + 'r%s/'%region + 'prediction_r%s_'%(region) + 'f%s_s%s_c%s_%s_%s_%s_%s_'%(self.freq, self.npix, int(self.cold_cores), self.model_name, self.loss_name, self.gamma, self.optimizer_name) + '%s'%(self.output_name) + ".csv")
                results.sort_values(by=['delta'], na_position='first', inplace=True)
                results.to_csv(self.output_path + 'files/' + 'r%s/'%region + 'prediction_r%s_'%(region) + 'f%s_s%s_c%s_%s_%s_%s_%s_'%(self.freq, self.npix, int(self.cold_cores), self.model_name, self.loss_name, self.gamma, self.optimizer_name) + '%s'%(self.output_name) + ".csv", mode='w', header=True, index=False)
            else:
                results.to_csv(self.output_path + 'files/' + 'r%s/'%region + 'prediction_r%s_'%(region) + 'f%s_s%s_c%s_%s_%s_%s_'%(self.freq, self.npix, int(self.cold_cores), self.model_name, self.loss_name, self.optimizer_name) + '%s'%(self.output_name) + ".csv", mode='a', header=False, index=False)
                results = pd.read_csv(self.output_path + 'files/' + 'r%s/'%region + 'prediction_r%s_'%(region) + 'f%s_s%s_c%s_%s_%s_%s_'%(self.freq, self.npix, int(self.cold_cores), self.model_name, self.loss_name, self.optimizer_name) + '%s'%(self.output_name) + ".csv")
                results.sort_values(by=['delta'], na_position='first', inplace=True)
                results.to_csv(self.output_path + 'files/' + 'r%s/'%region + 'prediction_r%s_'%(region) + 'f%s_s%s_c%s_%s_%s_%s_'%(self.freq, self.npix, int(self.cold_cores), self.model_name, self.loss_name, self.optimizer_name) + '%s'%(self.output_name) + ".csv", mode='w', header=True, index=False)

        except:
            results = pd.DataFrame(data = {'model': [np.nan, self.model_name], 'loss': [np.nan, self.loss_name], 'delta': [np.nan, self.delta], 'gamma': [np.nan, self.gamma], 'precision': [np.nan, precision[0]], 'recall': [np.nan, recall[0]],
                'dsc': [np.nan, dsc[0]], 'planck z': [npz, len(df_planck_z_detections)/npz], 'planck no z':[npnz, len(df_planck_no_z_detections)/npnz], 'MCXC': [nmcxc, len(df_MCXC_detections)/nmcxc], 
                'RM30': [nrm30, len(df_RM30_detections)/nrm30], 'RM50': [nrm50, len(df_RM50_detections)/nrm50], 'ACT': [nact, len(df_ACT_detections)/nact], 'SPT': [nspt, len(df_SPT_detections)/nspt], 
                'False detections':[nfc, len(df_false_clusters_detections)/nfc], 'Cold cores': [ncc, len(df_cc)/ncc], 'Compact sources': [ncs, len(df_cs)/ncs], 'Unknown': [np.nan, len(unknown_detections)],
                'total density': [np.nan, (len(df_detections))], 
                'planck z density': [np.nan, (len(df_planck_z_detections))], 
                'density without planck z': [np.nan, (len(df_detections)-len(df_planck_z_detections))], 
                'detection ratio without planck z': [np.nan, (len(df_detections)-len(df_planck_z_detections))]})
            
            if self.loss_name == 'focal_tversky_loss':
                results.to_csv(self.output_path + 'files/' + 'r%s/'%region + 'prediction_r%s_'%(region) + 'f%s_s%s_c%s_%s_%s_%s_%s_'%(self.freq, self.npix, int(self.cold_cores), self.model_name, self.loss_name, self.gamma, self.optimizer_name) + '%s'%(self.output_name) + ".csv", mode='w', header=True, index=False)
            else:
                results.to_csv(self.output_path + 'files/' + 'r%s/'%region + 'prediction_r%s_'%(region) + 'f%s_s%s_c%s_%s_%s_%s_'%(self.freq, self.npix, int(self.cold_cores), self.model_name, self.loss_name, self.optimizer_name) + '%s'%(self.output_name) + ".csv", mode='w', header=True, index=False)

        file.writelines(L)
        file.close()
  
        return unknown_detections, df_planck_z_detections, df_planck_no_z_detections, df_MCXC_detections, df_RM50_detections, df_RM30_detections, df_false_clusters_detections


    def best_result(self, regions):
        tot_planck_z, tot_planck_no_z, tot_mcxc, tot_rm30, tot_rm50, tot_act, tot_spt, tot_fd, tot_cc, tot_cs, tot_patch = 0,0,0,0,0,0,0,0,0,0,0
        for i,region in enumerate(regions):
            x_left, x_right, y_up, y_down = self.test_regions[region]
            _, test_catalog = MakeData.test_coords(self, x_left, x_right, y_up, y_down)
            
            if self.loss_name == 'focal_tversky_loss':
                results = pd.read_csv(self.output_path + 'files/' + 'r%s/'%region + 'prediction_r%s_'%(region) + 'f%s_s%s_c%s_%s_%s_%s_%s_'%(self.freq, self.npix, int(self.cold_cores), self.model_name, self.loss_name, self.gamma, self.optimizer_name) + '%s'%(self.output_name) + ".csv")
            else:
                results = pd.read_csv(self.output_path + 'files/' + 'r%s/'%region + 'prediction_r%s_'%(region) + 'f%s_s%s_c%s_%s_%s_%s_'%(self.freq, self.npix, int(self.cold_cores), self.model_name, self.loss_name, self.optimizer_name) + '%s'%(self.output_name) + ".csv")

            tot_planck_z += results['planck z'].to_numpy()[0]
            tot_planck_no_z += results['planck no z'].to_numpy()[0]
            tot_mcxc += results['MCXC'].to_numpy()[0]
            tot_rm30 += results['RM30'].to_numpy()[0]
            tot_rm50 += results['RM50'].to_numpy()[0]
            tot_act += results['ACT'].to_numpy()[0]
            tot_spt += results['SPT'].to_numpy()[0]
            tot_fd += results['False detections'].to_numpy()[0]
            tot_patch += len(test_catalog)
            tot_cc += results['Cold cores'].to_numpy()[0]
            tot_cs += results['Compact sources'].to_numpy()[0]

        best_results = pd.DataFrame(data = {'region': [np.nan], 'model': [self.model_name], 'loss': [self.loss_name],'delta': [np.nan], 'gamma': [np.nan], 'precision': [np.nan], 'dsc': [np.nan], 'recall': [np.nan],
            'planck z': [tot_planck_z], 'planck no z': [tot_planck_no_z], 'MCXC': [tot_mcxc], 'RM30': [tot_rm30], 'RM50': [tot_rm50], 
            'ACT': [tot_act], 'SPT': [tot_spt], 'False detections': [tot_fd], 'Cold cores': [tot_cc], 'Compact sources': [tot_cs], 'Unknown': [np.nan], 'score': [np.nan],
            'total density': [np.nan], 'planck z density': [np.nan], 'density without planck z': [np.nan], 'detection ratio without planck z': [np.nan]}) 

        for i,region in enumerate(regions):
            if i == 0:
                unknown_detections = pd.read_csv(self.output_path + 'files/' + 'r%s/'%region + 'unknown_detections_r%s_%s_%s'%(region, self.pre_output_name, self.output_name))
            else:
                unknown_detections = pd.concat((unknown_detections, pd.read_csv(self.output_path + 'files/' + 'r%s/'%region + 'unknown_detections_r%s_%s_%s'%(region, self.pre_output_name, self.output_name))))

            if self.loss_name == 'focal_tversky_loss':
                results = pd.read_csv(self.output_path + 'files/' + 'r%s/'%region + 'prediction_r%s_'%(region) + 'f%s_s%s_c%s_%s_%s_%s_%s_'%(self.freq, self.npix, int(self.cold_cores), self.model_name, self.loss_name, self.gamma, self.optimizer_name) + '%s'%(self.output_name) + ".csv")
            else:
                results = pd.read_csv(self.output_path + 'files/' + 'r%s/'%region + 'prediction_r%s_'%(region) + 'f%s_s%s_c%s_%s_%s_%s_'%(self.freq, self.npix, int(self.cold_cores), self.model_name, self.loss_name, self.optimizer_name) + '%s'%(self.output_name) + ".csv")

            results.insert(0, "region", [region]*len(results))
            score = [np.nan]
            index2drop = []
            for j in range(1, len(results)):
                if results['Unknown'].to_numpy()[j] < 5000:
                    score.append(results['precision'].to_numpy()[j] + results['planck z'].to_numpy()[j] + results['MCXC'].to_numpy()[j] + results['RM50'].to_numpy()[j] + results['ACT'].to_numpy()[j] + results['SPT'].to_numpy()[j] - results['False detections'].to_numpy()[j])#- results['Cold cores'].to_numpy()[j] - results['Compact sources'].to_numpy()[j])
                else:
                    index2drop.append(j)
            results.drop(index=index2drop, inplace=True)
            results.reset_index(inplace=True)
            results["score"] = score
            results.sort_values(by=['score'], na_position='first', ascending=False, inplace=True)
            

            best_results = pd.concat((best_results, results.iloc[0:2]), axis=0)

        best_results.drop(columns=['index', 'score'], inplace=True)

        # total = pd.DataFrame(data = {'planck z': [1/tot_planck_z], 'planck no z': [1/tot_planck_no_z], 'MCXC': [1/tot_mcxc], 'RM30': [1/tot_rm30], 'RM50': [1/tot_rm50], 
        #     'ACT': [1/tot_act], 'SPT': [1/tot_spt], 'False detections': [1/tot_fd], 'Unknown': [1]})

        best_result = best_results.groupby(['region'], as_index=False).prod()
        best_result = best_result[['planck z', 'planck no z', 'MCXC', 'RM30', 'RM50', 'ACT', 'SPT', 'False detections', 'Cold cores', 'Compact sources', 'Unknown', 'total density', 'planck z density', 'density without planck z', 'detection ratio without planck z']].copy()
        final_result = best_result[['planck z', 'planck no z', 'MCXC', 'RM30', 'RM50', 'ACT', 'SPT', 'False detections', 'Cold cores', 'Compact sources', 'Unknown', 'total density', 'planck z density', 'density without planck z', 'detection ratio without planck z']].copy()
        final_result = final_result.sum()
        # result = pd.DataFrame(columns=['planck z', 'planck no z', 'MCXC', 'RM30', 'RM50', 'ACT', 'SPT', 'False detections', 'Unknown'])

        best_result.loc['total'] = [tot_planck_z, tot_planck_no_z, tot_mcxc, tot_rm30,
                                 tot_rm50, tot_act, tot_spt, tot_fd, tot_cc ,tot_cs, 
                                 np.nan, tot_patch*self.ndeg*self.ndeg, tot_patch*self.ndeg*self.ndeg, tot_patch*self.ndeg*self.ndeg, 
                                 tot_planck_no_z + tot_mcxc + tot_rm30 + tot_rm50 + tot_act + tot_spt]
        try:
            final_result.drop(columns=['index'], inplace=True)
        except:
            pass

        best_result.loc['detections'] = [final_result['planck z'], final_result['planck no z'], final_result['MCXC'], final_result['RM30'],
                                 final_result['RM50'], final_result['ACT'], final_result['SPT'], final_result['False detections'], 
                                 final_result['Cold cores'], final_result['Compact sources'], 
                                 final_result['Unknown'], final_result['total density'], final_result['planck z density'], 
                                 final_result['density without planck z'], final_result['detection ratio without planck z']]

        final_result['planck z'] = final_result['planck z']/tot_planck_z
        final_result['planck no z'] = final_result['planck no z']/tot_planck_no_z
        final_result['MCXC'] = final_result['MCXC']/tot_mcxc
        final_result['RM30'] = final_result['RM30']/tot_rm30
        final_result['RM50'] = final_result['RM50']/tot_rm50
        final_result['ACT'] = final_result['ACT']/tot_act
        final_result['SPT'] = final_result['SPT']/tot_spt
        final_result['Cold cores'] = final_result['Cold cores']/tot_fd
        final_result['Compact sources'] = final_result['Compact sources']/tot_fd
        final_result['False detections'] = final_result['False detections']/tot_fd
        final_result['total density'] = final_result['total density']/tot_patch*self.ndeg*self.ndeg
        final_result['planck z density'] = final_result['planck z density']/tot_patch*self.ndeg*self.ndeg
        final_result['density without planck z'] = final_result['density without planck z']/tot_patch*self.ndeg*self.ndeg
        final_result['detection ratio without planck z'] = final_result['detection ratio without planck z']/(tot_planck_no_z + tot_mcxc + tot_rm30 + tot_rm50 + tot_act + tot_spt)

        best_result.loc['ratio'] = [final_result['planck z'], final_result['planck no z'], final_result['MCXC'], final_result['RM30'],
                                 final_result['RM50'], final_result['ACT'], final_result['SPT'], final_result['False detections'], 
                                 final_result['Cold cores'], final_result['Compact sources'], 
                                 final_result['Unknown'], final_result['total density'], final_result['planck z density'], 
                                 final_result['density without planck z'], final_result['detection ratio without planck z']]

        
        if self.loss_name == 'focal_tversky_loss':
            best_result.to_csv(self.output_path + 'files/' + 'prediction_' + 'f%s_s%s_c%s_%s_%s_%s_%s_'%(self.freq, self.npix, int(self.cold_cores), self.model_name, self.loss_name, self.gamma, self.optimizer_name) + '%s'%(self.output_name) + ".csv", mode='w', header=True, index=False)
        else:
             best_result.to_csv(self.output_path + 'files/' + 'prediction_' + 'f%s_s%s_c%s_%s_%s_%s_'%(self.freq, self.npix, int(self.cold_cores), self.model_name, self.loss_name, self.optimizer_name) + '%s'%(self.output_name) + ".csv", mode='w', header=True, index=False)
           
        print(best_result.head(40))

        self.stack_detections(unknown_detections, output_name='unknown')
        # planck_z = pd.read_csv(self.path + 'catalogs/planck_z' + '.csv')
        # self.stack_detections(planck_z, output_name='planck_z')

        print(best_result.head(40))

        return unknown_detections


    def plot_tversky(self, regions):

        if not self.cold_cores:
            fig = plt.figure(figsize=(30,50))
        if self.cold_cores:
            fig = plt.figure(figsize=(30, 40))

        fig.tight_layout()
        plt.subplots_adjust(wspace=0.7, hspace=0.4)

        for i,region in enumerate(regions):
            # try:
            if self.loss_name == 'focal_tversky_loss':
                results = pd.read_csv(self.output_path + 'files/' + 'r%s/'%region + 'prediction_r%s_'%(region) + 'f%s_s%s_c%s_%s_%s_%s_%s_'%(self.freq, self.npix, int(self.cold_cores), self.model_name, self.loss_name, self.gamma, self.optimizer_name) + '%s'%(self.output_name) + ".csv")
            else:
                results = pd.read_csv(self.output_path + 'files/' + 'r%s/'%region + 'prediction_r%s_'%(region) + 'f%s_s%s_c%s_%s_%s_%s_'%(self.freq, self.npix, int(self.cold_cores), self.model_name, self.loss_name, self.optimizer_name) + '%s'%(self.output_name) + ".csv")
            # except:
            #     results = pd.read_csv(self.output_path + 'files/' + 'r%s/'%region + 'prediction_r%s_'%(region) + 'f%s_%s_%s_%s_'%(self.freq, self.model_name, self.loss_name, self.optimizer_name) + '%s'%(self.output_name) + ".csv")

            ax1 = fig.add_subplot(5,3,1+i)
            ax2 = ax1.twinx()
            ax3 = ax2.twiny()
            ax1.get_shared_x_axes().join(ax2, ax3)
            ax2.grid(False)
            ax3.grid(False)
            if i%3 == 2:
                ax2.set_ylabel('Unknown detections', fontsize=20)

            ax1.set_facecolor('white')
            ax1.grid(True, color='grey', lw=0.5)
            if not self.cold_cores and i > 10:
                ax1.set_xlabel(r'$\delta$', fontsize=20)
            if self.cold_cores and i > 2:
                ax1.set_xlabel(r'$\delta$', fontsize=20)
            if i%3 == 0:
                ax1.set_ylabel('Ratio', fontsize=20)
            ax1.set_title('region %s'%int(region), fontsize=20, y=1.1)

            ax1.set_ylim(0,1.01)
            ax1.set_xlim(0,1)
            ax2.set_ylim(0,5000)
            # ax2.set_yscale('log')

            if i == len(regions)-1:
                # ax1.plot(results['delta'].to_numpy()[1:], results['precision'].to_numpy()[1:], linestyle='--', label='Precision')
                # ax1.plot(results['delta'].to_numpy()[1:], results['recall'].to_numpy()[1:], linestyle='--', label='Recall')
                # ax1.plot(results['delta'].to_numpy()[1:], results['dsc'].to_numpy()[1:], linestyle='--', label='Dice coefficient')
                ax1.plot(results['delta'].to_numpy()[1:], np.where(results['planck z'].to_numpy()[1:] < 1, results['planck z'].to_numpy()[1:], 1), linewidth=4, color='k', label='Planck z (tot: %s)'%int(results['planck z'].to_numpy()[0]))
                ax1.plot(results['delta'].to_numpy()[1:], results['planck no z'].to_numpy()[1:], label='Planck no z (tot: %s)'%int(results['planck no z'].to_numpy()[0]))
                ax1.plot(results['delta'].to_numpy()[1:], results['MCXC'].to_numpy()[1:], label='MCXC (tot: %s)'%int(results['MCXC'].to_numpy()[0]))
                ax1.plot(results['delta'].to_numpy()[1:], results['RM30'].to_numpy()[1:], label='RM30 (tot: %s)'%int(results['RM30'].to_numpy()[0]))
                ax1.plot(results['delta'].to_numpy()[1:], results['RM50'].to_numpy()[1:], label='RM50 (tot: %s)'%int(results['RM50'].to_numpy()[0]))
                ax1.plot(results['delta'].to_numpy()[1:], results['ACT'].to_numpy()[1:], label='ACT (tot: %s)'%int(results['ACT'].to_numpy()[0]))
                ax1.plot(results['delta'].to_numpy()[1:], results['SPT'].to_numpy()[1:], label='SPT (tot: %s)'%int(results['SPT'].to_numpy()[0]))
                ax1.plot(results['delta'].to_numpy()[1:], results['Cold cores'].to_numpy()[1:], linewidth=4, color='red', label='Cold cores (tot: %s)'%int(results['Cold cores'].to_numpy()[0]))
                ax1.plot(results['delta'].to_numpy()[1:], results['Compact sources'].to_numpy()[1:], color='orange', label='Compact sources (tot: %s)'%int(results['Compact sources'].to_numpy()[0]))

                ax2.plot(results['delta'].to_numpy()[1:], results['Unknown'].to_numpy()[1:], linewidth=4, linestyle='--', label='Unknown detections', color ='k')
            
                legend = ax2.legend(bbox_to_anchor=(1.2,0), loc="lower left", shadow=True, fontsize='x-large')
                legend = ax1.legend(bbox_to_anchor=(1.2,0.1), loc="lower left", shadow=True, fontsize='x-large')

                ax2.set_ylabel('Unknown detections', fontsize=20)
            else:
                # ax1.plot(results['delta'].to_numpy()[1:], results['precision'].to_numpy()[1:], linestyle='--')
                # ax1.plot(results['delta'].to_numpy()[1:], results['recall'].to_numpy()[1:], linestyle='--')
                # ax1.plot(results['delta'].to_numpy()[1:], results['dsc'].to_numpy()[1:], linestyle='--')
                ax1.plot(results['delta'].to_numpy()[1:], np.where(results['planck z'].to_numpy()[1:] < 1, results['planck z'].to_numpy()[1:], 1), linewidth=4, color='k', label='%s'%int(results['planck z'].to_numpy()[0]))
                ax1.plot(results['delta'].to_numpy()[1:], results['planck no z'].to_numpy()[1:], label='%s'%int(results['planck no z'].to_numpy()[0]))
                ax1.plot(results['delta'].to_numpy()[1:], results['MCXC'].to_numpy()[1:], label='%s'%int(results['MCXC'].to_numpy()[0]))
                ax1.plot(results['delta'].to_numpy()[1:], results['RM30'].to_numpy()[1:], label='%s'%int(results['RM30'].to_numpy()[0]))
                ax1.plot(results['delta'].to_numpy()[1:], results['RM50'].to_numpy()[1:], label='%s'%int(results['RM50'].to_numpy()[0]))
                ax1.plot(results['delta'].to_numpy()[1:], results['ACT'].to_numpy()[1:], label='%s'%int(results['ACT'].to_numpy()[0]))
                ax1.plot(results['delta'].to_numpy()[1:], results['SPT'].to_numpy()[1:], label='%s'%int(results['SPT'].to_numpy()[0]))
                ax1.plot(results['delta'].to_numpy()[1:], results['Cold cores'].to_numpy()[1:], linewidth=4, color='red', label='%s'%int(results['Cold cores'].to_numpy()[0]))
                ax1.plot(results['delta'].to_numpy()[1:], results['Compact sources'].to_numpy()[1:], color='orange', label='%s'%int(results['Compact sources'].to_numpy()[0]))

                ax2.plot(results['delta'].to_numpy()[1:], results['Unknown'].to_numpy()[1:], linewidth=4, linestyle='--', color ='k')
            
                legend = ax1.legend(bbox_to_anchor=(1.2,0), loc="lower left", shadow=True, fontsize='x-large')

            # order
            ax3.zorder = 1 # fills in back
            ax1.zorder = 2 # then the line
            ax2.zorder = 3 # then the points
            ax1.patch.set_visible(False)

        if self.loss_name == 'focal_tversky_loss':
            plt.savefig(self.output_path + 'figures/' + 'prediction_' + 'f%s_s%s_c%s_%s_%s_%s_%s_'%(self.freq, self.npix, int(self.cold_cores), self.model_name, self.loss_name, self.gamma, self.optimizer_name) + '%s'%(self.output_name) + '.png', bbox_inches='tight', transparent=False)
            plt.savefig(self.output_path + 'figures/' + 'prediction_' + 'f%s_s%s_c%s_%s_%s_%s_%s_'%(self.freq, self.npix, int(self.cold_cores), self.model_name, self.loss_name, self.gamma, self.optimizer_name) + '%s'%(self.output_name) + '.pdf', bbox_inches='tight', transparent=True)
        else:
            plt.savefig(self.output_path + 'figures/' + 'prediction_' + 'f%s_s%s_c%s_%s_%s_%s_'%(self.freq, self.npix, int(self.cold_cores), self.model_name, self.loss_name, self.optimizer_name) + '%s'%(self.output_name) + '.png', bbox_inches='tight', transparent=False)
            plt.savefig(self.output_path + 'figures/' + 'prediction_' + 'f%s_s%s_c%s_%s_%s_%s_'%(self.freq, self.npix, int(self.cold_cores), self.model_name, self.loss_name, self.optimizer_name) + '%s'%(self.output_name) + '.pdf', bbox_inches='tight', transparent=True)
        plt.show()


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

    def return_dup_coords(self, region, df_detections, tol=4, plot=True):
        #### DEPRECATED FOR NOW ####
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
            plt.savefig(self.output_path + 'figures/' + 'r%s/'%region + 'prediction_r%s_%s_%s'%(region, self.pre_output_name, self.output_name) + '/d2d_detections_duplicates_' + self.dataset + '.png', bbox_inches='tight', transparent=False)
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

    def match_with_catalog(self, region, df_main, df_catalog, tol=7, output_name=None, plot=False):  

        ID = np.arange(0, len(df_catalog))
        df_catalog = df_catalog[['GLON', 'GLAT']].copy()
        df_catalog.insert(loc=0, value=ID, column='ID')

        scatalog_sub = SkyCoord(df_main['GLON'].values, df_main['GLAT'].values, unit='deg', frame='galactic')
        pcatalog_sub = SkyCoord(df_catalog['GLON'].values, df_catalog['GLAT'].values, unit='deg', frame='galactic')
        idx, d2d, _ = match_coordinates_sky(scatalog_sub, pcatalog_sub, nthneighbor=1)

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
            plt.savefig(self.output_path + 'figures/' + 'r%s/'%region + 'prediction_r%s_%s_%s'%(region, self.pre_output_name, self.output_name) + '/d2d_detections_%s'%output_name + '.png', bbox_inches='tight', transparent=False)
            plt.show()
            plt.close()

        df_main['ismatched'], df_main['ID'], df_main['d2d'] = ismatched, idx, d2d

        df_catalog.drop(columns=['GLON', 'GLAT'], inplace=True)

        df_common = pd.merge(df_main, df_catalog, indicator=True, on='ID', how='outer').query('_merge=="both"').drop('_merge', axis=1)

        df_common.sort_values("d2d", inplace = True) 

        df_common.query("ismatched == True", inplace=True)

        df_duplicates = df_common.copy()

        df_duplicates = df_duplicates[df_duplicates.duplicated(subset=['ID'], keep=False)]

        df_duplicates.drop_duplicates(subset ="ID", keep = 'last', inplace = True)  

        df_common.drop_duplicates(subset ="ID", keep = 'first', inplace = True) 

        # size = len(df_common)
        if len(df_common) != 0 and len(df_common) != 1:
            df_common = MakeData.remove_duplicates_on_lonlat(self, df_common, tol=5, with_itself = True)
        else:
            df_common.drop(columns=['ismatched', 'ID'], inplace=True)

        return df_common, df_duplicates
        


    def evaluate_prediction(self, regions, plot=True, plot_patch=True):

        from tensorflow.keras.utils import CustomObjectScope

        for region in regions:
            test_dataset = self.npy_to_tfdata(region, batch_size=self.batch, buffer_size=1000, only_train=False, only_test=True)

            with CustomObjectScope({'iou': losses.iou, 'f1': losses.f1, 'dsc': losses.dsc, self.loss_name: self.loss, 'loss_function': self.loss, 'dice_coefficient': losses.dice_coefficient}):
                model = tf.keras.models.load_model(self.path + "tf_saves/" + self.dataset + "/model_r%s_%s_%s"%(region, self.pre_output_name, self.output_name) + ".h5")
            
            GenerateFiles.make_directory(self, path_to_file = self.output_path + 'figures/' + 'r%s/'%region)
            GenerateFiles.make_directory(self, path_to_file = self.output_path + 'figures/' + 'r%s/'%region + 'prediction_r%s_%s_%s'%(region, self.pre_output_name, self.output_name))
            GenerateFiles.make_directory(self, path_to_file = self.output_path + 'files/' + 'r%s/'%region)
     
            if plot == True:
                for metric in ['dsc', 'precision', 'recall', 'loss', 'lr', 'precision_1', 'recall_1', 'precision_2', 'recall_2', 'precision_3', 'recall_3', 'precision_4', 'recall_4', 'precision_5', 'recall_5', 
                                'precision_6', 'recall_6', 'precision_7', 'recall_7', 'precision_8', 'recall_8', 'precision_9', 'recall_9', 'precision_10', 'recall_10', 'precision_11', 'recall_11']:
                    self.plot_metric(region, metric)
                pixel_coords, precision, recall, dsc = self.show_predictions(region, model, dataset = test_dataset, num=1000, plot=plot_patch)
            else:
                pixel_coords, precision, recall, dsc = self.make_predictions(region, model, dataset = test_dataset, num=1000)
                # pixel_coords = self.load_predictions(region)

            unknown_detections, df_planck_z_detections, df_planck_no_z_detections, df_MCXC_detections, df_RM50_detections, df_RM30_detections, df_false_clusters_detections = self.match_detections_against_catalogs(region, pixel_coords, precision, recall, dsc, plot=plot)
            # if not unknown_detections.empty:
            #     self.stack_detections(region, unknown_detections, output_name='stack_unkown_detections')
            # if not df_planck_z_detections.empty:
            #     self.stack_detections(region, df_planck_z_detections, output_name='stack_planck_z_detections')
            # if not df_planck_no_z_detections.empty:
            #     self.stack_detections(region, df_planck_no_z_detections, output_name='stack_planck_no_z_detections')
            # if not df_MCXC_detections.empty:
            #     self.stack_detections(region, df_MCXC_detections, output_name='stack_MCXC_detections')
            # if not df_RM50_detections.empty:
            #     self.stack_detections(region, df_RM50_detections, output_name='stack_RM50_detections')
            # if not df_RM30_detections.empty:
            #     self.stack_detections(region, df_RM30_detections, output_name='stack_RM30_detections')
            # if not df_false_clusters_detections.empty:
            #     self.stack_detections(region, df_false_clusters_detections, output_name='stack_false_cluster_detections')

    
    def stack_detections(self, detections, output_name):

        maps = []
        title = ['100 GHz', '143 GHz', '217 GHz', '353 GHz', '545 GHz', '857 GHz', 'y-map', 'CO', 'noise']
        maps.append((self.planck_path + "HFI_SkyMap_100-field-IQU_2048_R3.00_full.fits", {'legend': 'HFI 100', 'docontour': True}))
        maps.append((self.planck_path + "HFI_SkyMap_143-field-IQU_2048_R3.00_full.fits", {'legend': 'HFI 143', 'docontour': True}))
        maps.append((self.planck_path + "HFI_SkyMap_217-field-IQU_2048_R3.00_full.fits", {'legend': 'HFI 217', 'docontour': True}))
        maps.append((self.planck_path + "HFI_SkyMap_353-psb-field-IQU_2048_R3.00_full.fits", {'legend': 'HFI 353', 'docontour': True}))
        maps.append((self.planck_path + "HFI_SkyMap_545-field-Int_2048_R3.00_full.fits", {'legend': 'HFI 545', 'docontour': True}))
        maps.append((self.planck_path + "HFI_SkyMap_857-field-Int_2048_R3.00_full.fits", {'legend': 'HFI 857', 'docontour': True}))
        maps.append((self.milca_path + "milca_ymaps.fits", {'legend': 'MILCA y-map', 'docontour': True}))
        maps.append((self.planck_path + "COM_CompMap_CO21-commander_2048_R2.00.fits", {'legend': 'CO', 'docontour': True}))
        maps.append((self.planck_path + 'COM_CompMap_Compton-SZMap-milca-stddev_2048_R2.00.fits', {'legend': 'noise', 'docontour': True}))

        cutsky = CutSky(maps, npix=self.npix, pixsize=self.pixsize, low_mem=False)
        unknown_coords = SkyCoord(detections['GLON'].values, detections['GLAT'].values, unit='deg', frame='galactic')
        stack = np.ndarray((len(unknown_coords),self.npix,self.npix, len(maps)))
        index2remove = []
        for i,coord in enumerate(unknown_coords):
            patch = cutsky.cut_fits(coord)
            HDU = patch[-1]['fits']
            wcs = WCS(HDU.header)
            for j in range(len(maps)):
                stack[i,:,:,j] = patch[j]['fits'].data #/np.amax(patch[j]['fits'].data)
                if np.mean(stack[i, self.npix//2 - 1:self.npix//2 + 2, self.npix//2 - 1:self.npix//2 +2, j]) < -10e5:
                    index2remove.append(i)
                    break
        
        print(index2remove)
        stack = np.delete(stack, index2remove, axis=0)

        stack_sum = np.sum(stack, axis=0)
        SEDs = np.mean(stack_sum[self.npix//2 - 1:self.npix//2 + 2, self.npix//2 - 1:self.npix//2 +2, :6], axis=(0,1))

        print(SEDs)
        print(np.arange(len(SEDs)))
        print(np.amin(SEDs), np.amax(SEDs))

        fig = plt.figure(figsize=(9,7), tight_layout=False)
        ax = fig.add_subplot(111)
        ax.set_facecolor('white')
        ax.grid(True, color='grey', lw=0.5)
        ax.set_yscale('symlog')
        # ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        ax.scatter(np.arange(len(SEDs)), SEDs, marker='o')
        # ax.set_ylim(np.amin(SEDs), np.amax(SEDs))
        ax.set_xlabel("frequency", fontsize=20)
        ax.set_ylabel("Stacked SED", fontsize=20)
        ax.xaxis.set_ticks(np.arange(len(SEDs)))
        ax.xaxis.set_ticklabels(['100 GHz','143 GHz','217 GHz','353 GHz','545 GHz','857 GHz'], rotation=45)
        if output_name == 'planck_z':
            plt.savefig(self.output_path + 'figures/' + 'SED_%s'%(output_name)   + '.png', bbox_inches='tight', transparent=False)
        else:
            plt.savefig(self.output_path + 'figures/' + 'SED_%s_%s_%s'%(output_name, self.pre_output_name, self.output_name)   + '.png', bbox_inches='tight', transparent=False)
        plt.show()
        plt.close()

        # fig = plt.figure(figsize=(9, 7), tight_layout=True)
        # plt.yscale('log')
        # plt.scatter(np.arange(len(SEDs)), SEDs, marker='o')
        # ax = plt.gca()
        # ax.xaxis.set_ticks(np.arange(len(SEDs)))
        # ax.xaxis.set_ticklabels(['100 GHz','143 GHz','217 GHz','353 GHz','545 GHz','857 GHz'], rotation=45)
        # plt.xlabel("frequency", fontsize=20)
        # plt.ylabel("Stacked SED", fontsize=20)

        # plt.savefig(self.output_path + 'figures/' + 'SED_%s_%s_%s'%(output_name, self.pre_output_name, self.output_name)   + '.png', bbox_inches='tight', transparent=False)
        # plt.show()
        # plt.close()

        fig = plt.figure(figsize=(20, 4), tight_layout=True)
        fig.suptitle(output_name, fontsize=14)

        for j in range(np.shape(stack_sum)[2]):
            plt.subplot(1, len(maps), j+1)
            plt.title(title[j], y=1.02)
            plt.imshow(stack_sum[self.npix//2 - 9:self.npix//2 + 10, self.npix//2 - 9:self.npix//2 +10,j], origin='lower')#, norm=LogNorm())
            # plt.colorbar()
            plt.axis('off')

        plt.savefig(self.output_path + 'figures/' + 'stack_%s_%s_%s'%(output_name, self.pre_output_name, self.output_name) + '.png', bbox_inches='tight', transparent=False)
        plt.show()
        plt.close()
        
        
        

        

    def make_predictions(self, region, model, dataset, num=1):
        n = 0
        pixel_coords = []
        try:
            milca_test = np.load(self.dataset_path + 'milca_test_pre_r%s_f%s_s%s_c%s_'%(region, 1022, self.npix, int(self.cold_cores)) + self.dataset + '.npz')['arr_0']
        except:
            milca_test = np.load(self.dataset_path + 'milca_test_pre_r%s_f%s_'%(region, 1022) + self.dataset + '.npz')['arr_0']
        predicted_masks = np.ndarray((len(milca_test),self.npix,self.npix,self.n_labels))

        precision = np.zeros((self.n_labels, len(milca_test)))
        recall = np.zeros((self.n_labels, len(milca_test)))
        dsc = np.zeros((self.n_labels, len(milca_test)))
        file = open(self.output_path + 'figures/' + 'r%s/'%region + 'prediction_r%s_%s_%s'%(region, self.pre_output_name, self.output_name) + "/results.txt","w")
     
        index_remove = []
        for image, mask in tqdm(dataset.take(num)):
            pred_mask = model.predict(image)
            for k in range(len(pred_mask)):
                for nl in range(self.n_labels):
                    predicted_masks[self.batch*n+k,:,:,nl] = pred_mask[k,:,:,nl]
                    is_all_zero = np.all((mask[k,:,:,nl] == 0))
                    if is_all_zero == False:
                        dsc[nl, self.batch*n+k] = losses.dsc(tf.convert_to_tensor(mask[k,:,:,nl], np.float64), tf.convert_to_tensor(pred_mask[k,:,:,nl], np.float64)).numpy()
                        precision[nl, self.batch*n+k] = losses.precision(tf.convert_to_tensor(mask[k,:,:,nl], np.float64), tf.convert_to_tensor(pred_mask[k,:,:,nl], np.float64)).numpy()
                        recall[nl, self.batch*n+k] = losses.recall(tf.convert_to_tensor(mask[k,:,:,nl], np.float64), tf.convert_to_tensor(pred_mask[k,:,:,nl], np.float64)).numpy()
                    else:
                        index_remove.append(self.batch*n+k)



                mask_list, _, _ = self.detect_clusters(im = pred_mask[k,:,:,0], pmax=self.pmax, dmin=self.dmin, dmax=self.dmax)
                if mask_list:
                    coords_in_patch = []
                    for i in range(len(mask_list)):
                        com = ndimage.measurements.center_of_mass(pred_mask[k,:,:,0]*(np.ones_like(pred_mask[k,:,:,0]) - mask_list[i]))
                        coords_in_patch.append(com)
                else:
                    coords_in_patch = []
                pixel_coords.append(coords_in_patch)
            n += 1

        dsc = np.delete(dsc, index_remove, axis=1)
        recall = np.delete(recall, index_remove, axis=1)
        precision = np.delete(precision, index_remove, axis=1)

        print(np.mean(precision, axis=1), np.mean(recall, axis=1), np.mean(dsc, axis=1))

        L = ['precision: {:.2f}%'.format(100*np.mean(precision, axis=1)[0]) + " \n", 
            'recall: {:.2f}%'.format(100*np.mean(recall, axis=1)[0]) +  " \n", 
            'dsc: {:.2f}%'.format(100*np.mean(dsc, axis=1)[0]) + " \n"] 

        file.writelines(L)
        file.close()

        np.save(self.dataset_path + 'prediction_mask_r%s_%s_%s'%(region, self.pre_output_name, self.output_name) , predicted_masks)
    
        return pixel_coords, np.mean(precision, axis=1), np.mean(recall, axis=1), np.mean(dsc, axis=1)


    def show_predictions(self, region, model, dataset, num=1, plot=True):
        try:
            milca_test = np.load(self.dataset_path + 'milca_test_pre_r%s_f%s_s%s_c%s_'%(region, 1022, self.npix, int(self.cold_cores)) + self.dataset + '.npz')['arr_0']
        except:
            milca_test = np.load(self.dataset_path + 'milca_test_pre_r%s_f%s_'%(region, 1022) + self.dataset + '.npz')['arr_0']

        n = 0
        pixel_coords = []
        predicted_masks = np.ndarray((len(milca_test),self.npix,self.npix,self.n_labels))
        final_masks = np.ndarray((len(milca_test),self.npix,self.npix,self.n_labels))

        precision = np.zeros((self.n_labels, len(milca_test)))
        recall = np.zeros((self.n_labels, len(milca_test)))
        dsc = np.zeros((self.n_labels, len(milca_test)))
        

        # index_remove = []

        for image, mask in tqdm(dataset.take(num)):
            pred_mask = model.predict(image)
            for k in range(len(pred_mask)):
                all_zero = np.zeros((self.npix, self.npix))
                for nl in range(self.n_labels):
                    predicted_masks[self.batch*n+k,:,:,nl] = pred_mask[k,:,:,nl]
                    all_zero += mask[k,:,:,nl] 
                    dsc[nl, self.batch*n+k] = losses.dsc(tf.convert_to_tensor(mask[k,:,:,nl], np.float64), tf.convert_to_tensor(pred_mask[k,:,:,nl], np.float64)).numpy()
                    precision[nl, self.batch*n+k] = losses.precision(tf.convert_to_tensor(mask[k,:,:,nl], np.float64), tf.convert_to_tensor(pred_mask[k,:,:,nl], np.float64)).numpy()
                    recall[nl, self.batch*n+k] = losses.recall(tf.convert_to_tensor(mask[k,:,:,nl], np.float64), tf.convert_to_tensor(pred_mask[k,:,:,nl], np.float64)).numpy()
                is_all_zero = np.all((all_zero == 0))
                        
                    # else:
                    #     index_remove.append(self.batch*n+k)
                # if plot == True:
                plt.figure(figsize=(36, 7), tight_layout=True)

                title = ['y map', 'True Mask', 'Predicted Mask', 'Detected clusters', 'Detected false clusters']

                plt.subplot(1, 3*self.n_labels+1, 1)
                plt.title(title[0], fontsize=20)
                plt.imshow(milca_test[self.batch*n+k,:,:,0], origin='lower')
                plt.axis('off')

                ## CLUSTERS
                plt.subplot(1, 3*self.n_labels+1, 2)
                plt.title(title[1], fontsize=20)
                plt.imshow(mask[k,:,:,0], origin='lower')
                plt.axis('off')
                            
                plt.subplot(1, 3*self.n_labels+1, 3)
                plt.title(title[2], fontsize=20)
                plt.imshow(pred_mask[k,:,:,0], origin='lower', vmin=0, vmax=1) #, norm=LogNorm(vmin=0.001, vmax=1)
                plt.axis('off')

                plt.subplot(1, 3*self.n_labels+1, 4)
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
                plt.title(title[3], fontsize=20)
                plt.axis('off')


                ## FALSE CLUSTERS
                if self.n_labels == 2:
                    plt.subplot(1, 3*self.n_labels+1, 5)
                    plt.title(title[1], fontsize=20)
                    plt.imshow(mask[k,:,:,1], origin='lower')
                    plt.axis('off')

                    plt.subplot(1, 3*self.n_labels+1, 6)
                    plt.title(title[2], fontsize=20)
                    plt.imshow(pred_mask[k,:,:,1], origin='lower', vmin=0, vmax=1) #, norm=LogNorm(vmin=0.001, vmax=1)
                    plt.axis('off')



                    # plt.subplot(1, 3*self.n_labels+1, 7)
                    # plt.title(title[4])
                    # mask_list, x_peak_list, y_peak_list = self.detect_clusters(im = pred_mask[k,:,:,1], pmax=self.pmax, dmin=self.dmin, dmax=self.dmax)
                    # if mask_list:
                    #     new_mask = np.zeros_like(pred_mask[k,:,:,1])
                    #     coords_in_patch = []
                    #     for i in range(len(mask_list)):
                    #         new_mask = new_mask + (np.ones_like(pred_mask[k,:,:,1]) - mask_list[i][:,:])
                            
                    #         plt.imshow(new_mask, origin='lower')
                    #         com = ndimage.measurements.center_of_mass(pred_mask[k,:,:,1]*(np.ones_like(pred_mask[k,:,:,1]) - mask_list[i]))
                    #         # plt.scatter(x_peak_list[i], y_peak_list[i], color='red')
                    #         plt.scatter(com[1], com[0], color='blue')
                    #         coords_in_patch.append(com)

                    #     final_masks[self.batch*n+k,:,:,1] = np.where(new_mask < 0, 0, new_mask)
                    # else:
                    #     plt.imshow(np.zeros_like(pred_mask[k,:,:,1]), origin='lower')
                    #     final_masks[self.batch*n+k,:,:,1] = np.zeros_like(pred_mask[k][:,:,1])
                    #     coords_in_patch = []
                    # pixel_coords.append(coords_in_patch)
                    # plt.axis('off')
                if not is_all_zero:
                    plt.savefig(self.output_path + 'figures/' + 'r%s/'%region + 'prediction_r%s_%s_%s'%(region, self.pre_output_name, self.output_name) + '/prediction_%s_%s'%(n, k)  + '.png', bbox_inches='tight', transparent=False)
                    plt.show()
                plt.close()
                # else:
                #     ## CLUSTERS
                #     mask_list, x_peak_list, y_peak_list = self.detect_clusters(im = pred_mask[k,:,:,0], pmax=self.pmax, dmin=self.dmin, dmax=self.dmax)
                #     if mask_list:
                #         new_mask = np.zeros_like(pred_mask[k,:,:,0])
                #         coords_in_patch = []
                #         for i in range(len(mask_list)):
                #             new_mask = new_mask + (np.ones_like(pred_mask[k,:,:,0]) - mask_list[i][:,:])
                #             com = ndimage.measurements.center_of_mass(pred_mask[k,:,:,0]*(np.ones_like(pred_mask[k,:,:,0]) - mask_list[i]))
                #             coords_in_patch.append(com)
                #         final_masks[self.batch*n+k,:,:,0] = np.where(new_mask < 0, 0, new_mask)
                #     else:
                #         final_masks[self.batch*n+k,:,:,0] =  np.zeros_like(pred_mask[k][:,:,0])
                #         coords_in_patch = []

                #     pixel_coords.append(coords_in_patch)

                #     ## SOURCES
                #     if self.n_labels == 2:
                #         mask_list, x_peak_list, y_peak_list = self.detect_clusters(im = pred_mask[k,:,:,1], pmax=self.pmax, dmin=self.dmin, dmax=self.dmax)
                #         if mask_list:
                #             new_mask = np.zeros_like(pred_mask[k,:,:,1])
                #             coords_in_patch = []
                #             for i in range(len(mask_list)):
                #                 new_mask = new_mask + (np.ones_like(pred_mask[k,:,:,1]) - mask_list[i][:,:])
                #                 com = ndimage.measurements.center_of_mass(pred_mask[k,:,:,1]*(np.ones_like(pred_mask[k,:,:,1]) - mask_list[i]))
                #                 coords_in_patch.append(com)
                #             final_masks[self.batch*n+k,:,:,1] = np.where(new_mask < 0, 0, new_mask)
                #         else:
                #             final_masks[self.batch*n+k,:,:,1] =  np.zeros_like(pred_mask[k][:,:,1])
                #             coords_in_patch = []

            n += 1

            # dsc = np.delete(dsc, index_remove, axis=1)
            # recall = np.delete(recall, index_remove, axis=1)
            # precision = np.delete(precision, index_remove, axis=1)

            # print(np.mean(precision, axis=1), np.mean(recall, axis=1), np.mean(dsc, axis=1))

            # file = open(self.output_path + 'figures/' + 'prediction_%s_%s_%s_r%s_f%s_'%(self.model_name, self.loss_name, self.optimizer_name, region, self.freq) + self.output_name + "/results.txt","w")

            # L = ['precision: {:.2f}%'.format(100*np.mean(precision, axis=1)[0]) + " \n", 
            #     'recall: {:.2f}%'.format(100*np.mean(recall, axis=1)[0]) +  " \n", 
            #     'dsc: {:.2f}%'.format(100*np.mean(dsc, axis=1)[0]) + " \n"] 

            # file.writelines(L)
            # file.close()
            np.save(self.dataset_path + 'prediction_mask_r%s_%s_%s'%(region, self.pre_output_name, self.output_name) , predicted_masks)
            np.save(self.dataset_path + 'final_mask_r%s_%s_%s'%(region, self.pre_output_name, self.output_name) , final_masks)
          
        return pixel_coords, np.mean(precision, axis=1), np.mean(recall, axis=1), np.mean(dsc, axis=1)

    def plot_metric(self, region, metric):

        data = pd.read_csv(self.path + "tf_saves/" + self.dataset + "/data_r%s_%s_%s"%(region, self.pre_output_name, self.output_name) + ".csv")
        
        try:
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
            plt.savefig(self.output_path + 'figures/' + 'r%s/'%region + 'prediction_r%s_%s_%s'%(region, self.pre_output_name, self.output_name) + '/' + metric  + '.png', bbox_inches='tight', transparent=False)
            plt.show()
            plt.close()
        except:
            pass

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

    def clusters_in_patch(self, region, catalog):

        #------------------------------------------------------------------#
        # # # # # Create common catalog # # # # #
        #------------------------------------------------------------------#


        x_left, x_right, y_up, y_down = self.test_regions[region]
        coord_catalog = catalog
        # planck_z = pd.read_csv(self.path + 'catalogs/planck_z' + '.csv')
        # planck_no_z = pd.read_csv(self.path + 'catalogs/planck_no-z' + '.csv')
        # MCXC = pd.read_csv(self.path + 'catalogs/MCXC_no_planck' + '.csv')
        # RM30 = pd.read_csv(self.path + 'catalogs/RM30_no_planck' + '.csv')
        # RM50 = pd.read_csv(self.path + 'catalogs/RM50_no_planck' + '.csv')
        # ACT = pd.read_csv(self.path + 'catalogs/ACT_no_planck' + '.csv')
        # SPT = pd.read_csv(self.path + 'catalogs/SPT_no_planck' + '.csv')

        # if catalog == 'planck_z':
        #     coord_catalog = planck_z[['RA', 'DEC', 'GLON', 'GLAT']].copy()
        # elif catalog == 'planck_no-z':
        #     coord_catalog = planck_no_z[['RA', 'DEC', 'GLON', 'GLAT']].copy()
        # elif catalog == 'MCXC':
        #     coord_catalog = MCXC[['RA', 'DEC', 'GLON', 'GLAT']].copy()
        # elif catalog == 'RM30':
        #     coord_catalog = RM30[['RA', 'DEC', 'GLON', 'GLAT']].copy()
        # elif catalog == 'RM50':
        #     coord_catalog = RM50[['RA', 'DEC', 'GLON', 'GLAT']].copy()
        # elif catalog == 'ACT':
        #     coord_catalog = ACT[['RA', 'DEC', 'GLON', 'GLAT']].copy()
        # elif catalog == 'SPT':
        #     coord_catalog = SPT[['RA', 'DEC', 'GLON', 'GLAT']].copy()

        test_coords, test_catalog = MakeData.test_coords(self, x_left, x_right, y_up, y_down)
        input_size = len(test_coords)

        coords_ns = SkyCoord(coord_catalog['GLON'].values, coord_catalog['GLAT'].values, unit='deg', frame='galactic')
        cluster_density = []
        coord_neighbours = []
        idx_cluster_list = []
        print("\n")
        print(len(coord_catalog))
        print("\n")
        if len(coord_catalog) < 100:
            limit = len(coord_catalog) 
        else:
            limit = 100
        for k in range(0,limit):
            if k == 0:
                idx_cluster_list.append(0)
            else:
                idx, _, _ = match_coordinates_sky(test_coords, coords_ns, nthneighbor=k)
                idx_cluster_list.append(idx)

        for i in range(input_size):
            ## Match between test patch center and galaxy clusters
            k = 1
            idx = idx_cluster_list[k]
            l_diff = np.abs(test_catalog['GLON'].values[i] - coord_catalog['GLON'].values[idx[i]])
            if np.abs(np.abs(test_catalog['GLON'].values[i] - coord_catalog['GLON'].values[idx[i]]) - 360) < l_diff:
                l_diff = np.abs(np.abs(test_catalog['GLON'].values[i] - coord_catalog['GLON'].values[idx[i]]) - 360)
            b_diff = np.abs(test_catalog['GLAT'].values[i] - coord_catalog['GLAT'].values[idx[i]])            
            if np.abs(np.abs(test_catalog['GLAT'].values[i] - coord_catalog['GLAT'].values[idx[i]]) - 180) < b_diff:
                b_diff = np.abs(np.abs(test_catalog['GLAT'].values[i] - coord_catalog['GLAT'].values[idx[i]]) - 180)
            k += 1
            neighb = [[coord_catalog['GLON'].values[idx[i]], coord_catalog['GLAT'].values[idx[i]]]]
            while l_diff <= 0.5*self.ndeg and b_diff <= 0.5*self.ndeg:
                idx = idx_cluster_list[k]
                l_diff = np.abs(test_catalog['GLON'].values[i] - coord_catalog['GLON'].values[idx[i]])
                if np.abs(np.abs(test_catalog['GLON'].values[i] - coord_catalog['GLON'].values[idx[i]]) - 360) < l_diff:
                    l_diff = np.abs(np.abs(test_catalog['GLON'].values[i] - coord_catalog['GLON'].values[idx[i]]) - 360)
                b_diff = np.abs(test_catalog['GLAT'].values[i] - coord_catalog['GLAT'].values[idx[i]])
                if np.abs(np.abs(test_catalog['GLAT'].values[i] - coord_catalog['GLAT'].values[idx[i]]) - 180) < b_diff:
                    b_diff = np.abs(np.abs(test_catalog['GLAT'].values[i] - coord_catalog['GLAT'].values[idx[i]]) - 180)
                neighb.append([coord_catalog['GLON'].values[idx[i]], coord_catalog['GLAT'].values[idx[i]]])
                k += 1
            coord_neighbours.append(neighb)
            cluster_density.append(k-2)

        sum_cluster = 0
        for i, coord in enumerate(test_coords):
            sum_cluster += cluster_density[i]

        return sum_cluster

    def false_clusters_in_patch(self, region, false_catalog):

        #------------------------------------------------------------------#
        # # # # # Create common catalog # # # # #
        #------------------------------------------------------------------#


        x_left, x_right, y_up, y_down = self.test_regions[region]

        # false_catalog = pd.read_csv(self.path + 'catalogs/False_SZ_catalog_f%s.csv'%self.planck_freq)
        # false_catalog.query("GLAT > %s"%y_down, inplace=True)
        # false_catalog.query("GLAT < %s"%y_up, inplace=True)
        # false_catalog.query("GLON > %s"%x_left, inplace=True)
        # false_catalog.query("GLON < %s"%x_right, inplace=True)

        test_coords, test_catalog = MakeData.test_coords(self, x_left, x_right, y_up, y_down)
        input_size = len(test_coords)


        false_coords = SkyCoord(false_catalog['GLON'].values, false_catalog['GLAT'].values, unit='deg', frame='galactic')
        false_cluster_density = []
        false_coord_neighbours = []
        idx_false_cluster_list = []
        print("\n")
        print(len(false_catalog))
        print("\n")
        if len(false_catalog) < 150:
            limit = len(false_catalog) 
        else:
            limit = 150
        for k in range(0,limit):
            if k == 0:
                idx_false_cluster_list.append(0)
            else:
                idx, _, _ = match_coordinates_sky(test_coords, false_coords, nthneighbor=k)
                idx_false_cluster_list.append(idx)

        for i in range(input_size):
            ## Match between test patch center and false clusters
            k = 1
            idx = idx_false_cluster_list[k]
            l_diff = np.abs(test_catalog['GLON'].values[i] - false_catalog['GLON'].values[idx[i]])
            if np.abs(np.abs(test_catalog['GLON'].values[i] - false_catalog['GLON'].values[idx[i]]) - 360) < l_diff:
                l_diff = np.abs(np.abs(test_catalog['GLON'].values[i] - false_catalog['GLON'].values[idx[i]]) - 360)
            b_diff = np.abs(test_catalog['GLAT'].values[i] - false_catalog['GLAT'].values[idx[i]])    
            if np.abs(np.abs(test_catalog['GLAT'].values[i] - false_catalog['GLAT'].values[idx[i]]) - 180) < b_diff:
                b_diff = np.abs(np.abs(test_catalog['GLAT'].values[i] - false_catalog['GLAT'].values[idx[i]]) - 180)        
            k += 1
            neighb = [[false_catalog['GLON'].values[idx[i]], false_catalog['GLAT'].values[idx[i]]]]
            while l_diff <= 0.5*self.ndeg and b_diff <= 0.5*self.ndeg:
                idx = idx_false_cluster_list[k]
                l_diff = np.abs(test_catalog['GLON'].values[i] - false_catalog['GLON'].values[idx[i]])
                if np.abs(np.abs(test_catalog['GLON'].values[i] - false_catalog['GLON'].values[idx[i]]) - 360) < l_diff:
                    l_diff = np.abs(np.abs(test_catalog['GLON'].values[i] - false_catalog['GLON'].values[idx[i]]) - 360)
                b_diff = np.abs(test_catalog['GLAT'].values[i] - false_catalog['GLAT'].values[idx[i]])
                if np.abs(np.abs(test_catalog['GLAT'].values[i] - false_catalog['GLAT'].values[idx[i]]) - 180) < b_diff:
                    b_diff = np.abs(np.abs(test_catalog['GLAT'].values[i] - false_catalog['GLAT'].values[idx[i]]) - 180)  
                neighb.append([false_catalog['GLON'].values[idx[i]], false_catalog['GLAT'].values[idx[i]]])
                k += 1
            false_coord_neighbours.append(neighb)
            false_cluster_density.append(k-2)

        sum_false = 0
        for i, coord in enumerate(test_coords):
            sum_false += false_cluster_density[i]

        return sum_false
