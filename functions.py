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
import shutil
import time
import glob
from joblib import Parallel, delayed
from scipy import ndimage
import losses
import models

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


from sklearn.model_selection import train_test_split

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

#------------------------------------------------------------------#
# # # # # Functions # # # # #
#------------------------------------------------------------------#


class GenerateFiles(object):
    """[summary]
    """

    def __init__(self, dataset, bands, output_path=None):
        """[summary]

        Args:
            dataset ([type]): [description]
            bands ([type]): [description]
            output_path ([type], optional): [description]. Defaults to None.
        """

        self.path = os.getcwd() +'/'
        self.dataset = dataset # 'planck_z', 'planck_z_no-z', 'MCXC', 'RM30', 'RM50'
        self.bands = bands
        self.temp_path = self.path + 'to_clean/'
        self.output_name = time.strftime("/%Y-%m-%d")
        if output_path is None:
            self.output_path = self.path
        else:
            self.output_path = output_path


    def make_directory(self, path_to_file):
        """[summary]

        Args:
            filename ([type]): [description]
        """

        try:
            os.mkdir(path_to_file)
        except OSError:
            pass
        else:
            print ("Successfully created the directory %s " % path_to_file)


    def make_directories(self, output=False, replace=False):
        """[summary]
        """

        if output == False:
            self.make_directory(self.temp_path)

        elif output == True:
            self.make_directory(self.output_path + 'output/')
            self.make_directory(self.output_path + 'tf_saves/')
            self.make_directory(self.output_path + 'catalogs/')
            self.make_directory(self.output_path + 'datasets/')
            self.make_directory(self.output_path + 'datasets/'+ self.dataset)
            self.make_directory(self.output_path + 'healpix/')
            self.make_directory(self.output_path + 'healpix/figures/')
            self.make_directory(self.output_path + 'healpix/figures/PSZ2')
            if replace == True:
                last_dir = '/' + os.listdir(self.output_path + 'output/' + self.dataset)[-1]
                if last_dir != self.output_name:
                    os.rename(self.output_path + 'output/' + self.dataset + last_dir, self.output_path + 'output/' + self.dataset + self.output_name)
                else:
                    self.make_directory(self.output_path + 'output/' + self.dataset)
            self.make_directory(self.output_path + 'output/' + self.dataset + self.output_name)
            self.make_directory(self.output_path + 'output/' + self.dataset + self.output_name + '/files')
            self.make_directory(self.output_path + 'output/' + self.dataset + self.output_name + '/figures')



    def remove_files_from_directory(self, folder):
        """Removes files for a given directory

        Args:
            folder ([str]): directory path
        """

        for file_name in os.listdir(folder):
            file_path = os.path.join(folder, file_name)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (file_path, e))


    def clean_temp_directories(self):
        """[summary]
        """

        if os.path.exists(self.temp_path) and os.path.isdir(self.temp_path):
            if not os.listdir(self.temp_path):
                print("Directory %s is empty"%(self.temp_path))
            else:
                self.remove_files_from_directory(self.temp_path)
                print("Successfully removed the directory %s " % (self.temp_path))
        else:
            print("Directory %s does not exist"%(self.temp_path))


    def is_directory_empty(self, path_to_dir):
        """Checks if given directory is empty or not.

        Args:
            path_to_dir ([str]): path to the directory

        Returns:
            [bool]: True if directory is empty, False if not.
        """

        if os.path.exists(path_to_dir) and os.path.isdir(path_to_dir):
            if not os.listdir(path_to_dir):
                print("Directory %s is empty"%path_to_dir)
                return True
            else:
                print("Directory %s is not empty"%path_to_dir)
                return False
        else:
            print("Directory %s don't exists"%path_to_dir)


class MakeData(object):

    def __init__(self, dataset, bands, planck_path, milca_path, disk_radius=None, output_path=None):
        """[summary]

        Args:
            dataset ([type]): [description]
            bands ([type]): [description]
            temp_path ([type]): [description]
        """

        self.path = os.getcwd() + '/'
        self.dataset = dataset # 'planck_z', 'planck_z_no-z', 'MCXC', 'RM30', 'RM50'
        self.bands = bands # '100GHz','143GHz','217GHz','353GHz','545GHz','857GHz'

        maps = []
        self.freq = 0
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
        if disk_radius is not None:
            self.disk_radius = disk_radius
        else:
            self.disk_radius = 'istri'
        self.npix = 64
        self.pixsize = 1.7
        self.nside = 2
        if output_path is None:
            self.output_path = self.path + 'output/' + self.dataset + time.strftime("/%Y-%m-%d/")
        else:
            self.output_path = output_path + 'output/' + self.dataset + time.strftime("/%Y-%m-%d/")
        self.dataset_path = self.path + 'datasets/' + self.dataset + '/'
        self.planck_path = planck_path
        self.milca_path = milca_path

    def plot_psz2_clusters(self, healpix_path):

        maps = self.maps

        PSZ2 = fits.open(self.planck_path + 'PSZ2v1.fits')
        glon = PSZ2[1].data['GLON']
        glat = PSZ2[1].data['GLAT']
        freq = ['100GHz','143GHz','217GHz','353GHz','545GHz','857GHz', 'y-map']

        for j in range(len(glon)):
                fig = plt.figure(figsize=(21,14), tight_layout=False)
                fig.suptitle(r'$glon=$ {:.2f} $^\circ$, $glat=$ {:.2f} $^\circ$'.format(glon[j], glat[j]), y=0.92, fontsize=20)
                cutsky = CutSky(maps, npix=self.npix, pixsize=self.pixsize, low_mem=False)
                coord = to_coord([glon[j], glat[j]])
                result = cutsky.cut_fits(coord)

                for i,nu in enumerate(freq):
                        ax = fig.add_subplot(3,4,1+i)
                        divider = make_axes_locatable(ax)
                        cax = divider.append_axes('right', size='5%', pad=0.05)
                        HDU = result[i]['fits']
                        im = ax.imshow(HDU.data, origin="lower")
                        w = WCS(HDU.header)
                        sky = w.world_to_pixel_values(glon[j], glat[j])
                        segmentation = plt.Circle((sky[0], sky[1]), 2.5/1.7, color='white', alpha=0.1)
                        ax.add_patch(segmentation)
                        ax.axvline(sky[0], ymin=0, ymax=(self.npix//2-10)/self.npix, color='white', linestyle='--')
                        ax.axvline(sky[0], ymin=(self.npix//2+10)/self.npix, ymax=1, color='white', linestyle='--')
                        ax.axhline(sky[1], xmin=0, xmax=(self.npix//2-10)/self.npix, color='white', linestyle='--')
                        ax.axhline(sky[1], xmin=(self.npix//2+10)/self.npix, xmax=1, color='white', linestyle='--')
                        # ax.scatter(sky[0], sky[1], color='red')
                        ax.set_title(r'%s'%nu)
                        fig.colorbar(im, cax=cax, orientation='vertical')
                plt.savefig(healpix_path + 'PSZ2/PSZ2_skycut_%s.png'%j, bbox_inches='tight', transparent=False)
                plt.show()
                plt.close()


    def create_catalogs(self, plot=True):
        PSZ2 = fits.open(self.planck_path + 'PSZ2v1.fits')
        df_psz2 = pd.DataFrame(data={'RA': PSZ2[1].data['RA'].tolist(), 'DEC': PSZ2[1].data['DEC'].tolist(), 'GLON': PSZ2[1].data['GLON'].tolist(), 'GLAT':PSZ2[1].data['GLAT'].tolist(),
            'M500': PSZ2[1].data['MSZ'].tolist(), 'R500': PSZ2[1].data['Y5R500'].tolist(), 'REDMAPPER': PSZ2[1].data['REDMAPPER'].tolist(), 'MCXC': PSZ2[1].data['MCXC'].tolist(),
            'Z': PSZ2[1].data['REDSHIFT'].tolist()})
        df_psz2 = df_psz2.replace([-1, -10, -99], np.nan)
        planck_no_z = df_psz2.query('Z.isnull()', engine='python')
        planck_z = df_psz2.query('Z.notnull()', engine='python')
        # planck_no_z = planck_no_z[['RA', 'DEC']].copy()
        # planck_z = planck_z[['RA', 'DEC']].copy()
        planck_no_z.to_csv(self.path + 'catalogs/planck_no-z' + '.csv', index=False)
        planck_z.to_csv(self.path + 'catalogs/planck_z' + '.csv', index=False)

        MCXC = fits.open(self.planck_path + 'MCXC-Xray-clusters.fits')
        df_MCXC = pd.DataFrame(data={'RA': MCXC[1].data['RA'].tolist(), 'DEC': MCXC[1].data['DEC'].tolist(), 'R500': MCXC[1].data['RADIUS_500'].tolist(), 'M500': MCXC[1].data['MASS_500'].tolist(),
            'Z': MCXC[1].data['REDSHIFT'].tolist()})

        REDMAPPER = fits.open(self.planck_path + 'redmapper_dr8_public_v6.3_catalog.fits')
        df_REDMAPPER = pd.DataFrame(data={'RA': REDMAPPER[1].data['RA'].tolist(), 'DEC': REDMAPPER[1].data['DEC'].tolist(), 'LAMBDA': REDMAPPER[1].data['LAMBDA'].tolist(),
        'Z': REDMAPPER[1].data['Z_SPEC'].tolist()})

        df_REDMAPPER_30 = df_REDMAPPER.query("LAMBDA > 30")
        df_REDMAPPER_50 = df_REDMAPPER.query("LAMBDA > 50")

        MCXC_no_planck = self.remove_duplicates_on_radec(df_MCXC, df_psz2, output_name='MCXC_no_planck', plot=plot)
        RedMaPPer_no_planck = self.remove_duplicates_on_radec(df_REDMAPPER_30, df_psz2, output_name='RM30_no_planck', plot=plot)
        RedMaPPer_no_planck = self.remove_duplicates_on_radec(df_REDMAPPER_50, df_psz2, output_name='RM50_no_planck', plot=plot)

        return planck_z, planck_no_z, MCXC_no_planck, RedMaPPer_no_planck

        # mv = missing_data(df_psz2)
        # print(mv.head())


    def remove_duplicates_on_radec(self, df_main, df_with_dup, output_name, plot=False):
        ID = np.arange(0, len(df_with_dup))
        df_with_dup = df_with_dup[['RA', 'DEC']].copy()
        df_with_dup.insert(loc=0, value=ID, column='ID')

        scatalog_sub = SkyCoord(ra=df_main['RA'].values, dec=df_main['DEC'].values, unit='deg')
        pcatalog_sub = SkyCoord(ra=df_with_dup['RA'].values, dec=df_with_dup['DEC'].values, unit='deg')
        idx, d2d, _ = match_coordinates_sky(scatalog_sub, pcatalog_sub, nthneighbor=1)

        tol = 7
        ismatched = d2d < tol*u.arcminute #threshold to consider whether or not two galaxies are the same

        df_d2d = pd.DataFrame(data={'ismatched': ismatched, 'idx': idx, 'd2d': d2d})
        if plot == True:
            fig = plt.figure(figsize=(8,8), tight_layout=False)
            ax = fig.add_subplot(111)
            ax.set_facecolor('white')
            ax.grid(True, color='grey', lw=0.5)
            ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
            ax.set_xlabel(r'$\mathrm{angular\;distance\;\left(arcmin\right)}$', fontsize=20)
            ax.set_ylabel(output_name, fontsize=20)
            ax.hist(np.array(df_d2d['d2d'].values)*60, bins = 400)
            ax.axvline(tol, color='k', linestyle='--')
            ax.set_xlim(0, 2*tol)
            plt.savefig(self.output_path + 'figures/' + 'd2d_' + output_name + '.png', bbox_inches='tight', transparent=False)
            plt.show()
            plt.close()

        df_d2d.query("ismatched == True", inplace=True)
        df_d2d.drop(columns=['ismatched'], inplace=True)

        df_main['ismatched'], df_main['ID'] = ismatched, idx

        df_with_dup.drop(columns=['RA', 'DEC'], inplace=True)

        df_wo_dup = pd.merge(df_main, df_with_dup, indicator=True, on='ID', how='outer').query('_merge=="both"').drop('_merge', axis=1)

        df_wo_dup.query("ismatched == False", inplace=True)
        df_wo_dup.drop(columns=['ismatched', 'ID'], inplace=True)
        df_wo_dup = df_wo_dup.replace([-1, -10, -99], np.nan)
        # df_wo_dup = df_wo_dup[['RA', 'DEC']].copy()
        df_wo_dup.to_csv(self.path + 'catalogs/' + output_name + '.csv', index=False)



        return df_wo_dup


    def missing_data(self, dataset):
        all_data_na = (dataset.isnull().sum() / len(dataset)) * 100
        all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending=False)[:30]
        missing_data = pd.DataFrame({'Missing Ratio' :all_data_na})
        return missing_data

    def create_circular_mask(self, h, w, center, ang_center, radius):
        if radius is None:
            size_distribution = fits.open(self.path + 'catalogs/exp_joined_ami_carma_plck_psz1_psz2_act_spt_YT.fits')[1].data['T500']
            heights, bins = np.histogram(size_distribution, bins=8, density=False, range=[0,15])
            heights = heights/sum(heights)
            bins = bins[1:]
            radius = np.random.choice(bins, p=heights)/self.pixsize
        else:
            radius = radius/self.pixsize

        Y, X = np.ogrid[:h, :w]
        mask = np.zeros((h,w))
        count = 0
        ra, dec = [], []
        for i,c in enumerate(center):
            dist_from_center = np.sqrt((X - c[0])**2 + (Y-c[1])**2)
            mask += (dist_from_center <= radius).astype(int)
            is_all_zero = np.all(((dist_from_center <= radius).astype(int) == 0))
            if is_all_zero == False:
                count += 1
                ra.append(ang_center[i][0])
                dec.append(ang_center[i][1])
        return np.where(mask > 1, 1, mask), count, ra, dec


    def create_input(self, p, plot=False, verbose=False):

        #------------------------------------------------------------------#
        # # # # # Create common catalog # # # # #
        #------------------------------------------------------------------#
        if p != 0:
            plot = False

        if self.dataset == 'planck_z':
            planck_z = pd.read_csv(self.path + 'catalogs/planck_z' + '.csv')
            coord_catalog = planck_z[['RA', 'DEC']].copy()
        elif self.dataset == 'planck_no-z':
            planck_z = pd.read_csv(self.path + 'catalogs/planck_z' + '.csv')
            planck_no_z = pd.read_csv(self.path + 'catalogs/planck_no-z' + '.csv')
            coord_catalog = pd.concat([planck_z[['RA', 'DEC']].copy(), planck_no_z[['RA', 'DEC']].copy()], ignore_index=True)
        elif self.dataset == 'MCXC':
            planck_z = pd.read_csv(self.path + 'catalogs/planck_z' + '.csv')
            planck_no_z = pd.read_csv(self.path + 'catalogs/planck_no-z' + '.csv')
            MCXC = pd.read_csv(self.path + 'catalogs/MCXC_no_planck' + '.csv')
            coord_catalog = pd.concat([planck_z[['RA', 'DEC']].copy(), planck_no_z[['RA', 'DEC']].copy(), MCXC[['RA', 'DEC']].copy()],
                ignore_index=True)
        elif self.dataset == 'RM30':
            planck_z = pd.read_csv(self.path + 'catalogs/planck_z' + '.csv')
            planck_no_z = pd.read_csv(self.path + 'catalogs/planck_no-z' + '.csv')
            MCXC = pd.read_csv(self.path + 'catalogs/MCXC_no_planck' + '.csv')
            RM30 = pd.read_csv(self.path + 'catalogs/RM30_no_planck' + '.csv')
            coord_catalog = pd.concat([planck_z[['RA', 'DEC']].copy(), planck_no_z[['RA', 'DEC']].copy(), MCXC[['RA', 'DEC']].copy(),
                RM30[['RA', 'DEC']].copy()], ignore_index=True)
        elif self.dataset == 'RM50':
            planck_z = pd.read_csv(self.path + 'catalogs/planck_z' + '.csv')
            planck_no_z = pd.read_csv(self.path + 'catalogs/planck_no-z' + '.csv')
            MCXC = pd.read_csv(self.path + 'catalogs/MCXC_no_planck' + '.csv')
            RM50 = pd.read_csv(self.path + 'catalogs/RM50_no_planck' + '.csv')
            coord_catalog = pd.concat([planck_z[['RA', 'DEC']].copy(), planck_no_z[['RA', 'DEC']].copy(), MCXC[['RA', 'DEC']].copy(),
                RM50[['RA', 'DEC']].copy()], ignore_index=True)

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

        if p == 0:
            test_positions = np.array([coord_catalog['RA'].values -30*1.7/60 + (60*1.7/60)*random_coord_x, coord_catalog['DEC'].values -30*1.7/60 + (60*1.7/60)*random_coord_y])
            np.save(self.dataset_path + 'test_coordinates_f%s_'%(self.freq) + self.dataset, test_positions)

        #------------------------------------------------------------------#
        # # # # # Check for potential neighbours # # # # #
        #------------------------------------------------------------------#

        scatalog = SkyCoord(ra=coord_catalog['RA'].values, dec=coord_catalog['DEC'].values, unit='deg')
        cluster_density = []
        coord_neighbours = []
        for i in range(input_size):
            idx, d2d, _ = match_coordinates_sky(scatalog, scatalog, nthneighbor=2)
            ra_diff = np.abs(coord_catalog['RA'].values[i] - coord_catalog['RA'].values[idx[i]])
            dec_diff = np.abs(coord_catalog['DEC'].values[i] - coord_catalog['DEC'].values[idx[i]])
            k = 3
            neighb = [[coord_catalog['RA'].values[idx[i]], coord_catalog['DEC'].values[idx[i]]]]
            while ra_diff < 1.76 and dec_diff < 1.76:
                idx, d2d, _ = match_coordinates_sky(scatalog, scatalog, nthneighbor=k)
                ra_diff = np.abs(coord_catalog['RA'].values[i] - coord_catalog['RA'].values[idx[i]])
                dec_diff = np.abs(coord_catalog['DEC'].values[i] - coord_catalog['DEC'].values[idx[i]])
                neighb.append([coord_catalog['RA'].values[idx[i]], coord_catalog['DEC'].values[idx[i]]])
                k += 1
            coord_neighbours.append(neighb)
            cluster_density.append(k-2)

        if plot == True:
            fig = plt.figure(figsize=(7,7), tight_layout=False)
            ax = fig.add_subplot(111)
            ax.set_facecolor('white')
            ax.grid(True, color='grey', lw=0.5)
            ax.set_xlabel('Neighbours per patch', fontsize=20)
            ax.set_ylabel('Cluster number', fontsize=20)
            ax.hist(cluster_density)
            ax.set_yscale('log')

            plt.savefig(self.output_path + 'figures/' + 'cluster_density' + '.png', bbox_inches='tight', transparent=False)
            plt.show()
            plt.close()

        #------------------------------------------------------------------#
        # # # # # Create patch & masks # # # # #
        #------------------------------------------------------------------#

        maps = self.maps

        cutsky = CutSky(maps, npix=self.npix, pixsize=self.pixsize, low_mem=False)

        labels = np.ndarray((input_size,self.npix,self.npix,1))
        inputs = np.ndarray((input_size,self.npix,self.npix,len(self.bands)))
        milca = np.ndarray((input_size,self.npix,self.npix,1))
        dataset_type = []

        hpi = ahp.HEALPix(nside=self.nside, order='ring', frame=ICRS())
        test_coords = [hpi.healpix_to_skycoord(healpix_index = 6), hpi.healpix_to_skycoord(healpix_index = 7)]
        val_coords = [hpi.healpix_to_skycoord(healpix_index = 9), hpi.healpix_to_skycoord(healpix_index = 38), hpi.healpix_to_skycoord(healpix_index = 41),
                      hpi.healpix_to_skycoord(healpix_index = 12), hpi.healpix_to_skycoord(healpix_index = 14), hpi.healpix_to_skycoord(healpix_index = 19),
                      hpi.healpix_to_skycoord(healpix_index = 21), hpi.healpix_to_skycoord(healpix_index = 23), hpi.healpix_to_skycoord(healpix_index = 25),
                      hpi.healpix_to_skycoord(healpix_index = 27), hpi.healpix_to_skycoord(healpix_index = 29)]


        
        for i, coord in enumerate(coords):
            
            # dataset_bool = False
            # for h in range(len(test_coords)):
            #     if np.abs(coord.ra.degree - test_coords[h].ra.degree) < 14.5 and np.abs(coord.dec.degree - test_coords[h].dec.degree) < 14.5:
            #         dataset_type.append('test')
            #         dataset_bool = True
            #         break
            #     else:
            #         pass
            # if dataset_bool == False:
            #     for h in range(len(val_coords)):
            #         if np.abs(coord.ra.degree - val_coords[h].ra.degree) < 14.5 and np.abs(coord.dec.degree - val_coords[h].dec.degree) < 14.5:
            #             dataset_type.append('val')
            #             dataset_bool = True
            #             break
            #         else:
            #             pass
            # if dataset_bool == False:
            #     dataset_type.append('train')

            if hp.ang2pix(self.nside, coord.galactic.l.degree, coord.galactic.b.degree, lonlat=True) == 6 or hp.ang2pix(self.nside, coord.galactic.l.degree, coord.galactic.b.degree, lonlat=True) == 7:
                dataset_type.append('test')
            elif hp.ang2pix(self.nside, coord.galactic.l.degree, coord.galactic.b.degree, lonlat=True) == 9 or hp.ang2pix(self.nside, coord.galactic.l.degree, coord.galactic.b.degree, lonlat=True) == 38 or hp.ang2pix(self.nside, coord.galactic.l.degree, coord.galactic.b.degree, lonlat=True) == 41 or hp.ang2pix(self.nside, coord.galactic.l.degree, coord.galactic.b.degree, lonlat=True) == 25:
                dataset_type.append('val')
            else:
                dataset_type.append('train')


            patch = cutsky.cut_fits(coord)
            HDU = patch[-1]['fits']
            wcs = WCS(HDU.header)
            x,y = wcs.world_to_pixel_values(coord_catalog['RA'].values[i], coord_catalog['DEC'].values[i])
            h, w = self.npix, self.npix
            center = [(x,y)]
            ang_center = [(coord_catalog['RA'].values[i], coord_catalog['DEC'].values[i])]
            if cluster_density[i] == 1:
                mask, _, _, _ = self.create_circular_mask(h, w, center=center, ang_center= ang_center, radius=self.disk_radius)
                labels[i,:,:,0] = mask.astype(int)
            else:
                for j in range(cluster_density[i]-1):
                    center.append(wcs.world_to_pixel_values(coord_neighbours[i][j][0], coord_neighbours[i][j][1]))
                    ang_center.append((coord_neighbours[i][j][0], coord_neighbours[i][j][1]))
                mask, _, _, _ = self.create_circular_mask(h, w, center=center, ang_center= ang_center, radius=self.disk_radius)
                labels[i,:,:,0] = mask.astype(int)

                if verbose:
                    print('\n')
                    print(i)
                    print('cluster density: %s'%cluster_density[i])
                    print('coords no shift: {:.2f}, {:.2f}'.format(coord_catalog['RA'].values[i], coord_catalog['DEC'].values[i]))
                    print('coords shift: {:.2f}, {:.2f}'.format(coord_catalog['RA'].values[i] -30*1.7/60 + (60*1.7/60)*random_coord_x[i], coord_catalog['DEC'].values[i] -30*1.7/60 + (60*1.7/60)*random_coord_y[i]))
                    print(coord_neighbours[i])
                    print(center)
                    print('\n')
            
            milca[i,:,:,0] = patch[-1]['fits'].data
            for j in range(len(self.bands)):
                inputs[i,:,:,j] = patch[j]['fits'].data

                #------------------------------------------------------------------#
                # # # # # Plots # # # # #
                #------------------------------------------------------------------#

                # if plot == True:
                #     fig = plt.figure(figsize=(20,5), tight_layout=False)
                #     ax = fig.add_subplot(131)
                #     divider = make_axes_locatable(ax)
                #     cax = divider.append_axes('right', size='5%', pad=0.05)
                #     im = ax.imshow(HDU.data, origin='lower')
                #     ax.scatter(x,y)
                #     ax.set_title('x={:.2f}, y={:.2f}'.format(x,y))
                #     fig.colorbar(im, cax=cax, orientation='vertical')

                #     ax = fig.add_subplot(132)
                #     divider = make_axes_locatable(ax)
                #     cax = divider.append_axes('right', size='5%', pad=0.05)
                #     im = ax.imshow(mask, origin='lower')
                #     ax.set_title('x={:.2f}, y={:.2f}'.format(x,y))
                #     fig.colorbar(im, cax=cax, orientation='vertical')

                #     ax = fig.add_subplot(133)
                #     divider = make_axes_locatable(ax)
                #     cax = divider.append_axes('right', size='5%', pad=0.05)
                #     patch_ns = cutsky.cut_fits(coords_ns[i])
                #     HDU_ns = patch_ns[6]['fits']
                #     im = ax.imshow(HDU_ns.data, origin='lower')
                #     ax.scatter(32,32)
                #     ax.set_title('x={:.0f}, y={:.0f}'.format(32,32))
                #     fig.colorbar(im, cax=cax, orientation='vertical')

                #     plt.savefig(self.temp_path + 'random_mask_milca-y_%s'%i + '.png', bbox_inches='tight', transparent=False)
                #     plt.show()
                #     plt.close()

        #------------------------------------------------------------------#
        # # # # # Save files # # # # #
        #------------------------------------------------------------------#

        assert len(coords) == len(dataset_type)

        GenerateFiles.make_directory(self, path_to_file = self.output_path + 'files/' + 'f%s'%(self.freq))
        np.savez_compressed(self.output_path + 'files/f%s/'%self.freq + 'milca_n%s_f%s_'%(p, self.freq) + self.dataset, milca)
        np.savez_compressed(self.output_path + 'files/f%s/'%self.freq + 'type_n%s_f%s_'%(p, self.freq) + self.dataset, np.array(dataset_type))
        if p == 0:
            np.savez_compressed(self.dataset_path + 'type_test_f%s_'%(self.freq) + self.dataset, np.array(dataset_type))
        np.savez_compressed(self.output_path + 'files/f%s/'%self.freq + 'input_n%s_f%s_'%(p, self.freq) + self.dataset, inputs)
        np.savez_compressed(self.output_path + 'files/f%s/'%self.freq + 'label_n%s_f%s_'%(p, self.freq) + self.dataset, labels)


    def train_data_generator(self, loops, n_jobs = 1, plot=True):

        all_files = glob.glob(os.path.join(self.output_path + "files/*.npz"))
        for f in all_files:
            os.remove(f)

        Parallel(n_jobs=n_jobs)(delayed(self.create_input)(p, plot=plot) for p in tqdm(range(loops)))

        all_type = glob.glob(os.path.join(self.output_path + 'files/f%s/'%self.freq, "type_n*.npz"))
        X = []
        for f in all_type:
            X.append(np.load(f)['arr_0'])
        dataset_type = np.concatenate(X, axis=0)
        np.savez_compressed(self.dataset_path + 'type_f%s_'%(self.freq) + self.dataset, dataset_type)

        if plot == True:
            counts = Counter(dataset_type)
            df = pd.DataFrame.from_dict(counts, orient='index')
            ax = df.plot(kind='bar')
            ax.figure.savefig(self.output_path + 'figures/' + 'dataset_type_density' + '.png', bbox_inches='tight', transparent=False)

        all_type = glob.glob(os.path.join(self.output_path + 'files/f%s/'%self.freq, "input_n*.npz"))
        X = []
        for f in all_type:
            X.append(np.load(f)['arr_0'])
        inputs = np.concatenate(X, axis=0)
        np.savez_compressed(self.dataset_path + 'input_f%s_'%(self.freq) + self.dataset, inputs)

        all_type = glob.glob(os.path.join(self.output_path + 'files/f%s/'%self.freq, "label_n*.npz"))
        X = []
        for f in all_type:
            X.append(np.load(f)['arr_0'])
        labels = np.concatenate(X, axis=0)
        np.savez_compressed(self.dataset_path + 'label_f%s_'%(self.freq) + self.dataset, labels)
        
        all_type = glob.glob(os.path.join(self.output_path + 'files/f%s/'%self.freq, "milca_n*.npz"))
        X = []
        for f in all_type:
            X.append(np.load(f)['arr_0'])
        milca = np.concatenate(X, axis=0)
        np.savez_compressed(self.dataset_path + 'milca_f%s_'%(self.freq) + self.dataset, milca)

    def fit_gaussian_up_to_mode(self, dataset, index, slice='train', plot=True):
        from scipy.optimize import leastsq

        dataset = np.sort(dataset)
        band = self.bands[index]
        print(band)
        density = False
        range_list = [(-0.0005, 0.0005), (-0.0005, 0.0005), (-0.0005, 0.001), (-0.0001, 0.004), (0.25, 1), (0.25, 4)]

        if plot == True:
            fig = plt.figure(figsize=(8,8), tight_layout=False)
            ax = fig.add_subplot(111)
            ax.set_facecolor('white')
            ax.grid(True, color='grey', lw=0.5)
            ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
            ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
            ax.set_xlabel('Pixel value', fontsize=20)
            ax.set_ylabel('Counts', fontsize=20)
            ax.set_title('%s'%band, fontsize=20)

            # the histogram of the data
            if band in self.bands:
                _n,_b,_ = ax.hist(dataset, bins=200, density=density, facecolor='grey', range=range_list[index])
                elem = np.argmax(_n)
                mode = _b[elem]
                mode_index = next(x[0] for x in enumerate(dataset) if x[1] > mode)
                ax.axvline(mode, color='k', linestyle='--', label='mode = {:.2e}'.format(mode))
                dataset_up_to_mode = np.sort(np.concatenate((dataset[:mode_index], (-1)*(dataset[:mode_index] - mode) + mode)))
                n, b = np.histogram(dataset_up_to_mode, bins=200, density=density, range=range_list[index])

            else:
                _n,_b,_ = ax.hist(dataset, bins=200, density=density, facecolor='grey')
                elem = np.argmax(_n)
                mode = _b[elem]
                mode_index = next(x[0] for x in enumerate(dataset) if x[1] > mode)
                ax.axvline(mode, color='k', linestyle='--', label='mode = {:.2e}'.format(mode))
                dataset_up_to_mode = np.sort(np.concatenate((dataset[:mode_index], (-1)*(dataset[:mode_index] - mode) + mode)))
                n, b = np.histogram(dataset_up_to_mode, bins=200, density=density, range=range_list[index])

            # add a 'best fit' line
            xdata = (b[1:] + b[:-1])/2
            ydata = n

            fitfunc  = lambda p, x: p[0]* np.exp(-0.5*((x-p[1])/p[2])**2)
            errfunc  = lambda p, x, y: (y - fitfunc(p, x))
            if band == '353GHz' or band == '545GHz' or band == '857GHz':
                init  = [n[elem], mode, 0.1]
            else:
                init  = [n[elem], 0.1, 0.1]

            out = leastsq(errfunc, init, args=(xdata, ydata))
            c = out[0]

            ax.plot(xdata, fitfunc(c, xdata), color='k', label=r'$\sigma = %.2e$'%(abs(c[2])) + '\n' + r'$\sigma_{\mathrm{MAD}} = %.2e$'%(mad_std(dataset_up_to_mode)))

            legend = ax.legend(loc='best', shadow=True, fontsize='x-large')

            plt.savefig(self.output_path + 'figures/' + 'preprocessing_gaussian_normalization_%s_%s'%(slice, band)  + '.png', bbox_inches='tight', transparent=False)
            plt.show()
            plt.close()
        
        if plot == False:
          
            _n,_b = np.histogram(dataset, bins=200, density=density, range=range_list[index])
            elem = np.argmax(_n)
            mode = _b[elem]
            mode_index = next(x[0] for x in enumerate(dataset) if x[1] > mode)
            dataset_up_to_mode = np.sort(np.concatenate((dataset[:mode_index], (-1)*(dataset[:mode_index] - mode) + mode)))
            n, b = np.histogram(dataset_up_to_mode, bins=200, density=density, range=range_list[index])

            # add a 'best fit' line
            xdata = (b[1:] + b[:-1])/2
            ydata = n
            fitfunc  = lambda p, x: p[0]* np.exp(-0.5*((x-p[1])/p[2])**2)
            errfunc  = lambda p, x, y: (y - fitfunc(p, x))
            if band == '353GHz' or band == '545GHz' or band == '857GHz':
                init  = [n[elem], mode, 0.1]
            else:
                init  = [n[elem], 0.1, 0.1]
            out = leastsq(errfunc, init, args=(xdata, ydata))
            c = out[0]

        print('[leastsq fit] A=%.0f (expected %.0f), mu=%.2e (expected %.2e), sigma=%.2e'%(c[0],n[elem],c[1], mode,abs(c[2])))
        print('[mad std] sigma=%.2e'%(mad_std(dataset_up_to_mode)))

        return abs(c[2])

    def train_val_test_split(self, inputs_train, labels_train, dataset_type_train, inputs_test, labels_test, dataset_type_test, milca_test):

        type_count_train = Counter(dataset_type_train)
        type_count_test = Counter(dataset_type_test)
        print("[Inputs] training: {:.0f}, validation: {:.0f}, test: {:.0f}".format(type_count_train['train'], type_count_train['val'], type_count_test['test']))

        X_train = np.ndarray((type_count_train['train'], self.npix, self.npix, len(self.bands)))
        X_val = np.ndarray((type_count_train['val'], self.npix, self.npix, len(self.bands)))
        X_test = np.ndarray((type_count_test['test'], self.npix, self.npix, len(self.bands)))
        M_test = np.ndarray((type_count_test['test'], self.npix, self.npix, len(self.bands)))
        output_train = np.ndarray((type_count_train['train'], self.npix, self.npix, 1))
        output_val = np.ndarray((type_count_train['val'], self.npix, self.npix, 1))
        output_test = np.ndarray((type_count_test['test'], self.npix, self.npix, 1))

        index_train, index_val, index_test = 0, 0, 0
        for i in range(len(inputs_train)):
            if dataset_type_train[i] == 'train':
                X_train[index_train,:,:,:] = inputs_train[i,:,:,:]
                output_train[index_train,:,:,:] = labels_train[i,:,:,:]
                index_train += 1
            if dataset_type_train[i] == 'val':
                X_val[index_val,:,:,:] = inputs_train[i,:,:,:]
                output_val[index_val,:,:,:] = labels_train[i,:,:,:]
                index_val += 1

        for i in range(len(inputs_test)):
            if dataset_type_test[i] == 'test':
                X_test[index_test,:,:,:] = inputs_test[i,:,:,:]
                output_test[index_test,:,:,:] = labels_test[i,:,:,:]
                M_test[index_test,:,:,:] = milca_test[i,:,:,:]
                index_test += 1

        return X_train, X_val, X_test, output_train, output_val, output_test, M_test

        
    def preprocess(self, leastsq=False, range_comp=True, plot=True):
        inputs_train = np.load(self.dataset_path + 'input_f%s_'%self.freq + self.dataset + '.npz')['arr_0']
        labels_train = np.load(self.dataset_path + 'label_f%s_'%self.freq + self.dataset + '.npz')['arr_0']
        dataset_type_train = np.load(self.dataset_path + 'type_f%s_'%self.freq + self.dataset + '.npz')['arr_0']

        inputs_test = np.load(self.output_path + 'files/f%s/'%self.freq + 'input_n0_f%s_'%self.freq + self.dataset + '.npz')['arr_0']
        labels_test = np.load(self.output_path + 'files/f%s/'%self.freq + 'label_n0_f%s_'%self.freq + self.dataset + '.npz')['arr_0']
        dataset_type_test = np.load(self.output_path + 'files/f%s/'%self.freq + 'type_n0_f%s_'%self.freq + self.dataset + '.npz')['arr_0']
        milca_test = np.load(self.output_path + 'files/f%s/'%self.freq + 'milca_n0_f%s_'%self.freq + self.dataset + '.npz')['arr_0']

        X_train, X_val, X_test, output_train, output_val, output_test, M_test = self.train_val_test_split(inputs_train, labels_train, dataset_type_train, inputs_test, labels_test, dataset_type_test, milca_test)

        np.savez_compressed(self.dataset_path + 'label_train_pre_f%s_d%s_'%(self.freq, self.disk_radius) + self.dataset, output_train)
        np.savez_compressed(self.dataset_path + 'label_val_pre_f%s_d%s_'%(self.freq, self.disk_radius) + self.dataset, output_val)
        np.savez_compressed(self.dataset_path + 'label_test_pre_f%s_d%s_'%(self.freq, self.disk_radius) + self.dataset, output_test)
        np.savez_compressed(self.dataset_path + 'milca_test_pre_f%s_'%(self.freq) + self.dataset, M_test)

        scaling_train, scaling_val, scaling_test = [], [], []

        for i in range(len(self.bands)):
            if leastsq == False:
                sigma_train = mad_std(X_train[...,i].flatten())
                scaling_train.append(sigma_train)
                sigma_val = mad_std(X_val[...,i].flatten())
                scaling_val.append(sigma_val)
                sigma_test = mad_std(X_test[...,i].flatten())
                scaling_test.append(sigma_test)
            if leastsq == True:
                gsigma_train = self.fit_gaussian_up_to_mode(X_train[...,i].flatten(), index=i, slice='train', plot=True)
                scaling_train.append(gsigma_train)
                gsigma_val = self.fit_gaussian_up_to_mode(X_val[...,i].flatten(), index=i, slice='val', plot=True)
                scaling_val.append(gsigma_val)
                gsigma_test = self.fit_gaussian_up_to_mode(X_test[...,i].flatten(), index=i, slice='test', plot=True)
                scaling_test.append(gsigma_test)

        if range_comp == True:
            input_train = np.ndarray((len(X_train),self.npix,self.npix,len(self.bands)))
            for i in range(len(X_train)):
                for j in range(len(self.bands)):
                    input_train[i,:,:,j] = np.arcsinh(X_train[i,:,:,j]/ scaling_train[j])
            np.savez_compressed(self.dataset_path + 'input_train_pre_f%s_'%self.freq + self.dataset, input_train)

            input_val = np.ndarray((len(X_val),self.npix,self.npix,len(self.bands)))
            for i in range(len(X_val)):
                for j in range(len(self.bands)):
                    input_val[i,:,:,j] = np.arcsinh(X_val[i,:,:,j]/ scaling_test[j])
            np.savez_compressed(self.dataset_path + 'input_val_pre_f%s_'%self.freq + self.dataset, input_val)

            input_test = np.ndarray((len(X_test),self.npix,self.npix,len(self.bands)))
            for i in range(len(X_test)):
                for j in range(len(self.bands)):
                    input_test[i,:,:,j] = np.arcsinh(X_test[i,:,:,j]/ scaling_test[j])
            np.savez_compressed(self.dataset_path + 'input_test_pre_f%s_'%self.freq + self.dataset, input_test)

        else:
            np.savez_compressed(self.dataset_path + 'input_train_pre_f%s_'%self.freq + self.dataset, X_train)
            np.savez_compressed(self.dataset_path + 'input_val_pre_f%s_'%self.freq + self.dataset, X_val)
            np.savez_compressed(self.dataset_path + 'input_test_pre_f%s_'%self.freq + self.dataset, X_test)


        if plot == True:
            density = True

            fig = plt.figure(figsize=(8,8), tight_layout=False)
            ax = fig.add_subplot(111)
            ax.set_facecolor('white')
            ax.grid(True, color='grey', lw=0.5)
            ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
            ax.set_xlabel('Pixel value', fontsize=20)
            ax.set_ylabel('Counts (normalized)', fontsize=20)
            for i,b in enumerate(self.bands):
                ax.hist(X_train[...,len(self.bands)-(i+1)].flatten()/scaling_train[len(self.bands)-(i+1)], bins=200, label=self.bands[len(self.bands)-(i+1)], density=density, range=[-3,10])
                legend = ax.legend(loc='best', shadow=True, fontsize='x-large')
            plt.savefig(self.output_path + 'figures/' + 'preprocessing_normalization_train'  + '.png', bbox_inches='tight', transparent=False)
            plt.show()
            plt.close()

            fig = plt.figure(figsize=(8,8), tight_layout=False)
            ax = fig.add_subplot(111)
            ax.set_facecolor('white')
            ax.grid(True, color='grey', lw=0.5)
            ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
            ax.set_xlabel('Pixel value', fontsize=20)
            ax.set_ylabel('Counts (normalized)', fontsize=20)
            for i,b in enumerate(self.bands):
                ax.hist(np.arcsinh(X_train[...,len(self.bands)-(i+1)].flatten()/scaling_train[len(self.bands)-(i+1)]/3), bins=200, label=self.bands[len(self.bands)-(i+1)], density=density, range=[-1.5,3])
                legend = ax.legend(loc='best', shadow=True, fontsize='x-large')
            plt.savefig(self.output_path + 'figures/' + 'preprocessing_range_compression_train'  + '.png', bbox_inches='tight', transparent=False)
            plt.show()
            plt.close()

            fig = plt.figure(figsize=(8,8), tight_layout=False)
            ax = fig.add_subplot(111)
            ax.set_facecolor('white')
            ax.grid(True, color='grey', lw=0.5)
            ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
            ax.set_xlabel('Pixel value', fontsize=20)
            ax.set_ylabel('Counts (normalized)', fontsize=20)
            for i,b in enumerate(self.bands):
                ax.hist(X_test[...,len(self.bands)-(i+1)].flatten()/scaling_test[len(self.bands)-(i+1)], bins=200, label=self.bands[len(self.bands)-(i+1)], density=density, range=[-3,10])
                legend = ax.legend(loc='best', shadow=True, fontsize='x-large')
            plt.savefig(self.output_path + 'figures/' + 'preprocessing_normalization_test'  + '.png', bbox_inches='tight', transparent=False)
            plt.show()
            plt.close()

            fig = plt.figure(figsize=(8,8), tight_layout=False)
            ax = fig.add_subplot(111)
            ax.set_facecolor('white')
            ax.grid(True, color='grey', lw=0.5)
            ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
            ax.set_xlabel('Pixel value', fontsize=20)
            ax.set_ylabel('Counts (normalized)', fontsize=20)
            for i,b in enumerate(self.bands):
                ax.hist(np.arcsinh(X_test[...,len(self.bands)-(i+1)].flatten()/scaling_test[len(self.bands)-(i+1)]/3), bins=200, label=self.bands[len(self.bands)-(i+1)], density=density, range=[-1.5,3])
                legend = ax.legend(loc='best', shadow=True, fontsize='x-large')
            plt.savefig(self.output_path + 'figures/' + 'preprocessing_range_compression_test'  + '.png', bbox_inches='tight', transparent=False)
            plt.show()
            plt.close()

        all_files = glob.glob(os.path.join(self.output_path + "files/f%s/"%self.freq + "*.npz"))
        for f in all_files:
            os.remove(f)
        os.rmdir(self.output_path + "files/f%s"%self.freq)
        os.remove(self.dataset_path + 'input_f%s_'%self.freq + self.dataset + '.npz')
        os.remove(self.dataset_path + 'label_f%s_'%self.freq + self.dataset + '.npz')
        os.remove(self.dataset_path + 'type_f%s_'%self.freq + self.dataset + '.npz')
        

        print('[preprocessing] Done!')


class CNNSegmentation(object):
    def __init__(self, model, dataset, bands, planck_path, milca_path, epochs, batch, lr, patience, loss, optimizer, size=64, disk_radius = None, drop_out=False, output_path = None):
        
        self.path = os.getcwd() + '/'
        self.dataset = dataset # 'planck_z', 'planck_z_no-z', 'MCXC', 'RM30', 'RM50'
        self.bands = bands # '100GHz','143GHz','217GHz','353GHz','545GHz','857GHz', 'y-map'
        maps = []
        self.freq = 0
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
        self.pmax=0.5
        self.dmin=3
        self.dmax=20

        self.output_name = 'e%s_b%s_lr%s_p%s_d%s'%(epochs, batch, lr, patience, disk_radius)

        optimizers_dict = {'sgd': SGD(lr=self.lr, momentum=0.9), 'adam': Adam(learning_rate=self.lr)}
        self.optimizer =  optimizers_dict[optimizer]
        self.optimizer_name =  optimizer
        losses_dict = {'binary_crossentropy': 'binary_crossentropy', 'tversky_loss': losses.tversky_loss, 'focal_tversky_loss': losses.focal_tversky_loss(gamma=0.75), 'dice_loss': losses.dice_loss, 
                       'combo_loss': losses.combo_loss(alpha=0.5,beta=0.5), 'cosine_tversky_loss': losses.cosine_tversky_loss(gamma=1), 'focal_dice_loss': losses.focal_dice_loss(delta=0.7, gamma_fd=0.75), 
                       'focal_loss': losses.focal_loss(alpha=None, beta=None, gamma_f=2.), 'mixed_focal_loss': losses.mixed_focal_loss(weight=None, alpha=None, beta=None, delta=0.7, gamma_f=2.,gamma_fd=0.75)}
        self.loss = losses_dict[loss]
        self.loss_name = loss
        model_dict = {'unet': models.unet, 'attn_unet': models.attn_unet, 'attn_reg_ds': models.attn_reg_ds, 'attn_reg': models.attn_reg}
        self.model = model_dict[model]
        self.model_name = model

    def npy_to_tfdata(self, batch_size=20, buffer_size=1000, input_train=None, input_val=None, input_test=None, output_train=None, output_val=None, output_test=None):
        if input_train is None:
            input_train = np.load(self.dataset_path + 'input_train_pre_f%s_'%self.freq + self.dataset + '.npz')['arr_0']
        if input_val is None:
            input_val = np.load(self.dataset_path + 'input_val_pre_f%s_'%self.freq + self.dataset + '.npz')['arr_0']
        if input_test is None:
            input_test = np.load(self.dataset_path + 'input_test_pre_f%s_'%self.freq + self.dataset + '.npz')['arr_0']
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
        val_dataset = val_dataset.shuffle(buffer_size).batch(batch_size).repeat()
        test_dataset = test_dataset.batch(batch_size)


        return train_dataset, val_dataset, test_dataset

    def train_model(self):
        from keras_unet_collection import models
        input_train = np.load(self.dataset_path + 'input_train_pre_f%s_'%self.freq + self.dataset + '.npz')['arr_0']
        input_val = np.load(self.dataset_path + 'input_val_pre_f%s_'%self.freq + self.dataset + '.npz')['arr_0']
        train_dataset, valid_dataset, test_dataset = self.npy_to_tfdata(batch_size=self.batch, buffer_size=1000)

        callbacks = [
            ModelCheckpoint(self.path + "tf_saves/" + self.dataset + "/model_%s_l%s_o%s_f%s_"%(self.model_name, self.loss_name, self.optimizer_name, self.freq) + self.output_name + ".h5", save_best_only=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=self.patience),
            CSVLogger(self.path + "tf_saves/" + self.dataset + "/data_%s_l%s_o%s_f%s_"%(self.model_name, self.loss_name, self.optimizer_name, self.freq) + self.output_name + ".csv"),
            TensorBoard(),
            EarlyStopping(monitor='val_loss', patience=self.patience, restore_best_weights=True)
        ]


        input_size = (self.npix, self.npix, len(self.bands))
        model = models.att_unet_2d(input_size, filter_num=[64, 128, 256, 512, 1024], n_labels=2, 
                           stack_num_down=2, stack_num_up=2, activation='ReLU', 
                           atten_activation='ReLU', attention='add', output_activation='Sigmoid', 
                           batch_norm=True, pool=False, unpool=False, freeze_batch_norm=True, 
                           name='attunet')

        metrics = [losses.dsc, tf.keras.metrics.Recall(), tf.keras.metrics.Precision(), losses.iou]
        model.compile(loss=self.loss, optimizer=self.optimizer, metrics=metrics) #tf.keras.losses.categorical_crossentropy

        # model = self.model(self.optimizer, input_size, self.loss)

        model_history = model.fit(train_dataset,
            validation_data=valid_dataset,
            epochs=self.epochs,
            steps_per_epoch = len(input_train)/self.batch,
            validation_steps= len(input_val)//self.batch//5,
            callbacks=callbacks)

    def pixel_neighbours(self, im, center, p, pmax, dmax):

        rows, cols, _ = np.shape(im)

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

        print('duplicate detections (total): %s'%(len(df_duplicates)),
              'planck_z: %s(%s)/%s(%s)'%(len(df_planck_z_detections), planck_z_detections_number, npz[0], npz[1]), 
              'planck_no-z: %s(%s)/%s(%s)'%(len(df_planck_no_z_detections), planck_no_z_detections_number, npnz[0], npnz[1]), 
              'MCXC: %s(%s)/%s(%s)'%(len(df_MCXC_detections), MCXC_detections_number, nmcxc[0], nmcxc[1]), 
              'RM50: %s(%s)/%s(%s)'%(len(df_RM50_detections), RM50_detections_number, nrm50[0], nrm50[1]), 
              'RM30: %s(%s)/%s(%s)'%(len(df_RM30_detections), RM30_detections_number, nrm30[0], nrm30[1]), 
              'Unknown: %s(%s)'%(len(df_detections) - len(df_duplicates) + (planck_z_detections_number-len(df_planck_z_detections) - (planck_no_z_detections_number-len(df_planck_no_z_detections))
                                 - (MCXC_detections_number-len(df_MCXC_detections)) - (RM50_detections_number-len(df_RM50_detections)) - (RM30_detections_number-len(df_RM30_detections)))
                                 - len(df_planck_z_detections)-len(df_planck_no_z_detections)-len(df_MCXC_detections)-len(df_RM50_detections)-len(df_RM30_detections), 
                                 len(df_detections)-len(df_planck_z_detections)-len(df_planck_no_z_detections)-len(df_MCXC_detections)-len(df_RM50_detections)-len(df_RM30_detections)))

        file = open(self.output_path + 'figures/' + 'prediction_%s_l%s_o%s_f%s_'%(self.model_name, self.loss_name, self.optimizer_name, self.freq) + self.output_name + "/results.txt","w")
        L = ['duplicate detections (total): %s'%(len(df_duplicates))+ " \n",
             'planck_z: %s(%s)/%s(%s)'%(len(df_planck_z_detections), planck_z_detections_number, npz[0], npz[1])+ " \n", 
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

    def remove_duplicates(self, df, tol=3):
        coords = SkyCoord(ra=df['RA'].values, dec=df['DEC'].values, unit='deg')
        _, d2d, _ = match_coordinates_sky(coords, coords, nthneighbor=2)
        isdup = d2d < tol*u.arcmin
        df['isdup'] = isdup
        df.query("isdup == False", inplace=True)
        df.drop(columns=['isdup'], inplace=True)

        return df

    def return_dup_coords(self, df_detections, tol=3, plot=True):

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
            plt.savefig(self.output_path + 'figures/' + 'd2d_detections_duplicates_' + self.dataset + '.png', bbox_inches='tight', transparent=False)
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
            plt.savefig(self.output_path + 'figures/' + 'd2d_detections_%s'%output_name + '.png', bbox_inches='tight', transparent=False)
            plt.show()
            plt.close()

        df_main['ismatched'], df_main['ID'] = ismatched, idx

        df_catalog.drop(columns=['RA', 'DEC'], inplace=True)

        df_common = pd.merge(df_main, df_catalog, indicator=True, on='ID', how='outer').query('_merge=="both"').drop('_merge', axis=1)

        df_common.query("ismatched == True", inplace=True)
        df_common.drop(columns=['ismatched', 'ID'], inplace=True)

        size = len(df_common)
        if len(df_common) != 0 and len(df_common) != 1:
            df_common = self.remove_duplicates(df_common)

        return df_common, size
        


    def evaluate_prediction(self, plot=True, plot_patch=True):
        train_dataset, valid_dataset, test_dataset = self.npy_to_tfdata(batch_size=self.batch, buffer_size=1000)

        from tensorflow.keras.utils import CustomObjectScope

        with CustomObjectScope({'iou': losses.iou, 'f1': losses.f1, 'dsc': losses.dsc, self.loss_name: self.loss, 'loss_function': self.loss}):
            model = tf.keras.models.load_model(self.path + "tf_saves/" + self.dataset + "/model_%s_l%s_o%s_f%s_"%(self.model_name, self.loss_name, self.optimizer_name, self.freq) + self.output_name + ".h5")

        if plot == True:
            GenerateFiles.make_directory(self, path_to_file = self.output_path + 'figures/' + 'prediction_%s_l%s_o%s_f%s_'%(self.model_name, self.loss_name, self.optimizer_name, self.freq) + self.output_name)
            for metric in ['dsc', 'precision', 'recall', 'loss', 'lr']:#['f1', 'acc', 'iou', 'precision', 'recall', 'loss', 'lr']:
                self.plot_metric(metric)
            pixel_coords = self.show_predictions(model, dataset = test_dataset, num=30, plot=plot_patch)
        else:
            self.make_predictions(model, dataset = test_dataset, num=30)
            pixel_coords = self.load_predictions()


        self.match_detections_against_catalogs(pixel_coords, plot=plot)

    

    def load_predictions(self):
        pixel_coords = []
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

        np.save(self.dataset_path + 'prediction_mask_%s_l%s_o%s_f%s_%s'%(self.model_name, self.loss_name, self.optimizer_name, self.freq, self.output_name) , predicted_masks)

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
                    plt.imshow(pred_mask[k], origin='lower', vmin=0, vmax=1) #, norm=LogNorm(vmin=0.001, vmax=1)
                    plt.axis('off')
                    plt.colorbar()

                    plt.subplot(1, 4, 4)
                    plt.title(title[3])
                    mask_list, x_peak_list, y_peak_list = self.detect_clusters(im = pred_mask[k], pmax=self.pmax, dmin=self.dmin, dmax=self.dmax)
                    if mask_list:
                        new_mask = np.zeros_like(pred_mask[k,:,:,0])
                        coords_in_patch = []
                        for i in range(len(mask_list)):
                            new_mask = new_mask + (np.ones_like(pred_mask[k,:,:,0]) - mask_list[i][:,:,0])
                            
                            plt.imshow(new_mask, origin='lower')
                            com = ndimage.measurements.center_of_mass(pred_mask[k]*(np.ones_like(pred_mask[k]) - mask_list[i]))
                            # plt.scatter(x_peak_list[i], y_peak_list[i], color='red')
                            plt.scatter(com[1], com[0], color='blue')
                            coords_in_patch.append(com)

                        final_masks[self.batch*n+k,:,:,0] = np.where(new_mask < 0, 0, new_mask)
                    else:
                        plt.imshow(np.zeros_like(pred_mask[k]), origin='lower')
                        final_masks[self.batch*n+k,:,:,0] = np.zeros_like(pred_mask[k][:,:,0])
                        coords_in_patch = []
                    pixel_coords.append(coords_in_patch)
                    plt.axis('off')

                    plt.savefig(self.output_path + 'figures/prediction_%s_l%s_o%s_f%s_%s/'%(self.model_name, self.loss_name, self.optimizer_name, self.freq, self.output_name) + 'prediction_%s_%s'%(n, k)  + '.png', bbox_inches='tight', transparent=False)
                    plt.show()
                    plt.close()
                else:
                    mask_list, x_peak_list, y_peak_list = self.detect_clusters(im = pred_mask[k], pmax=self.pmax, dmin=self.dmin, dmax=self.dmax)
                    if mask_list:
                        new_mask = np.zeros_like(pred_mask[k,:,:,0])
                        coords_in_patch = []
                        for i in range(len(mask_list)):
                            new_mask = new_mask + (np.ones_like(pred_mask[k,:,:,0]) - mask_list[i][:,:,0])
                            com = ndimage.measurements.center_of_mass(pred_mask[k]*(np.ones_like(pred_mask[k]) - mask_list[i]))
                            coords_in_patch.append(com)
                        final_masks[self.batch*n+k,:,:,0] = np.where(new_mask < 0, 0, new_mask)
                    else:
                        final_masks[self.batch*n+k,:,:,0] =  np.zeros_like(pred_mask[k][:,:,0])
                        coords_in_patch = []

                    pixel_coords.append(coords_in_patch)
            n += 1
            np.save(self.dataset_path + 'prediction_mask_%s_l%s_o%s_f%s_%s'%(self.model_name, self.loss_name, self.optimizer_name, self.freq, self.output_name) , predicted_masks)
            np.save(self.dataset_path + 'final_mask_%s_l%s_o%s_f%s_%s'%(self.model_name, self.loss_name, self.optimizer_name, self.freq, self.output_name) , final_masks)

        return pixel_coords

    def plot_metric(self, metric):

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
        plt.savefig(self.output_path + 'figures/prediction_%s_l%s_o%s_f%s_%s/'%(self.model_name, self.loss_name, self.optimizer_name, self.freq, self.output_name) + metric  + '.png', bbox_inches='tight', transparent=False)
        plt.show()
        plt.close()

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

    def get_focal_params(self, y_pred):
        epsilon = tf.constant(1e-9)
        gamma = tf.constant(3.)
        y_pred = y_pred + epsilon
        pinv = 1./y_pred
        pos_weight_f = (pinv - 1)**gamma
        weight_f = y_pred**gamma
        return pos_weight_f, weight_f

    def custom_loss(self, y_true,y_pred):
        y_pred_prob = tf.keras.backend.sigmoid(y_pred)    
        pos_weight_f, weight_f = self.get_focal_params(y_pred_prob)
        alpha = tf.constant(.35)
        alpha_ = 1 - alpha
        alpha_div = alpha / alpha_
        pos_weight = pos_weight_f * alpha_div
        weight = weight_f * alpha_

        l2 = weight * tf.nn.weighted_cross_entropy_with_logits\
        (labels=y_true, logits=y_pred, pos_weight=pos_weight)
        return l2

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
