#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 01 10:00:58 2021

@author: Xavier Jimenez
"""


#------------------------------------------------------------------#
# # # # # Imports # # # # #
#------------------------------------------------------------------#
from math import e
import numpy as np
import pandas as pd
import os
import time
import glob
import itertools
from joblib import Parallel, delayed
from generate_files import GenerateFiles

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import LogNorm
import seaborn as sns
import matplotlib.style as style
style.use('seaborn-poster') #sets the size of the charts
style.use('ggplot')

from scipy import ndimage

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

class MakeData(object):
    """Class to create and preprocess input/output files from full sky-maps.
    """

    def __init__(self, dataset, npix, loops, planck_path, milca_path, disk_radius=None, output_path=None):
        """
        Args:
            dataset (str): file name for the cluster catalog that will used.
                        Options are 'planck_z', 'planck_z_no-z', 'MCXC', 'RM30', 'RM50'.
            bands (list): list of full sky-maps that will be used for the input file.
            loops (int): number of times the dataset containing patches with at least one cluster within will be added 
                        again to training set with random variations (translations/rotations).
                        Options are 100GHz','143GHz','217GHz','353GHz','545GHz','857GHz', and 'y-map'.
                        More full sky-maps will be added later on (e.g. CO2, X-ray, density maps).
            planck_path (str): path to directory containing planck HFI 6 frequency maps.
                        Files should be named as following
                        'HFI_SkyMap_100-field-IQU_2048_R3.00_full.fits', 'HFI_SkyMap_143-field-IQU_2048_R3.00_full.fits', 
                        'HFI_SkyMap_143-field-IQU_2048_R3.00_full.fits', 'HFI_SkyMap_353-psb-field-IQU_2048_R3.00_full.fits', 
                        'HFI_SkyMap_545-field-Int_2048_R3.00_full.fits', 'HFI_SkyMap_857-field-Int_2048_R3.00_full.fits'.
            milca_path (str): path to directory containing MILCA full sky map. File should be named 'milca_ymaps.fits'.
            disk_radius (float, optional): Disk radius that will be used to create segmentation masks for output files. 
                        Defaults to None.
            output_path (str, optional): Path to output directory. Output directory needs be created beforehand using 
                        'python xcluster.py -m True' selecting same output directory in 'params.py'.
                        If None, xcluster path will be used. Defaults to None.
        """

        self.path = os.getcwd() + '/'
        self.dataset = dataset # 'planck_z', 'planck_z_no-z', 'MCXC', 'RM30', 'RM50'
        self.bands =  ['100GHz','143GHz','217GHz','353GHz','545GHz','857GHz','y-map','CO','p-noise']
        self.loops = loops
        self.n_labels = 2

        maps = []
        self.freq = 1022
        self.planck_freq = 126
        if '100GHz' in  self.bands:
            maps.append((planck_path + "HFI_SkyMap_100-field-IQU_2048_R3.00_full.fits", {'legend': 'HFI 100', 'docontour': True}))
            # self.freq += 2
            # self.planck_freq += 2
        if '143GHz' in self.bands:
            maps.append((planck_path + "HFI_SkyMap_143-field-IQU_2048_R3.00_full.fits", {'legend': 'HFI 143', 'docontour': True}))
            # self.freq += 4
            # self.planck_freq += 4
        if '217GHz' in self.bands:
            maps.append((planck_path + "HFI_SkyMap_217-field-IQU_2048_R3.00_full.fits", {'legend': 'HFI 217', 'docontour': True}))
            # self.freq += 8
            # self.planck_freq += 8
        if '353GHz' in self.bands:
            maps.append((planck_path + "HFI_SkyMap_353-psb-field-IQU_2048_R3.00_full.fits", {'legend': 'HFI 353', 'docontour': True}))
            # self.freq += 16
            # self.planck_freq += 16
        if '545GHz' in self.bands:
            maps.append((planck_path + "HFI_SkyMap_545-field-Int_2048_R3.00_full.fits", {'legend': 'HFI 545', 'docontour': True}))
            # self.freq += 32
            # self.planck_freq += 32
        if '857GHz' in self.bands:
            maps.append((planck_path + "HFI_SkyMap_857-field-Int_2048_R3.00_full.fits", {'legend': 'HFI 857', 'docontour': True}))
            # self.freq += 64
            # self.planck_freq += 64
        if 'y-map' in self.bands:
            maps.append((milca_path + "milca_ymaps.fits", {'legend': 'MILCA y-map', 'docontour': True}))
            # self.freq += 128
        if 'CO' in self.bands:
            maps.append((planck_path + "COM_CompMap_CO21-commander_2048_R2.00.fits", {'legend': 'CO', 'docontour': True}))
            # self.freq += 256
        if 'p-noise' in self.bands:
            maps.append((planck_path + 'COM_CompMap_Compton-SZMap-milca-stddev_2048_R2.00.fits', {'legend': 'noise', 'docontour': True}))
            # self.freq += 512
        maps.append((milca_path + "milca_ymaps.fits", {'legend': 'MILCA y-map', 'docontour': True})) #used for plots only
        
        self.maps = maps

        self.temp_path = self.path + 'to_clean/'

        self.disk_radius = disk_radius

        self.npix = npix #in pixels
        self.pixsize = 1.7 #in arcmin
        self.ndeg = (self.npix * self.pixsize)/60 #in deg
        self.nside = 2
        if output_path is None:
            self.output_path = self.path + 'output/' + self.dataset + time.strftime("/%Y-%m-%d/")
        else:
            self.output_path = output_path + 'output/' + self.dataset + time.strftime("/%Y-%m-%d/")
        self.dataset_path = self.path + 'datasets/' + self.dataset + '/'
        self.planck_path = planck_path
        self.milca_path = milca_path

        self.test_regions = [[0, 360, 90, 70], 
                        [0, 120, 70, 40], [120, 240, 70, 40], [240, 360, 70, 40],
                        [0, 120, 40, 18], [120, 240, 40, 18], [240, 360, 40, 18],
                        [0, 120, -18, -40], [120, 240, -18, -40], [240, 360, -18, -40],
                        [0, 120, -40, -70], [120, 240, -40, -70], [240, 360, -40, -70], 
                        [0, 360, -70, -90]]

        self.val_regions = [[0, 180, -20, -40],
                        [0, 180, -20, -40], [0, 180, -20, -40], [0, 180, -20, -40],  
                        [0, 360, -40, -60], [0, 360, -40, -60], [0, 360, -40, -60], 
                        [0, 360, 60, 40], [0, 360, 60, 40], [0, 360, 60, 40],
                        [0, 180, 40, 20], [0, 180, 40, 20], [0, 180, 40, 20],
                        [0, 180, 40, 20]]


    def plot_psz2_clusters(self, healpix_path):
        """Saves plots containing patches for planck frequency maps and y-map.
        Function is deprecated and will be removed in later versions.

        Args:
            healpix_path (str): output path for plots (deprecated).
        """

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


    def create_catalogs(self, plot=False):
        """Creates the following catalogs using 'PSZ2v1.fits', 'MCXC-Xray-clusters.fits', and 'redmapper_dr8_public_v6.3_catalog.fits'
            (see V. Bonjean 2018 for more details):

            planck_z (pd.DataFrame): dataframe with the following columns for PSZ2 clusters with known redshift:
                                    'RA', 'DEC', 'GLON', 'GLAT', 'M500', 'R500', 'Y5R500', 'REDMAPPER', 'MCXC', 'Z'
            planck_no_z (pd.DataFrame): dataframe with the following columns for PSZ2 clusters with unknown redshift:
                                    'RA', 'DEC', 'GLON', 'GLAT', 'M500', 'R500', 'Y5R500', 'REDMAPPER', 'MCXC'
            MCXC_no_planck (pd.DataFrame): dataframe with the following columns for MCXC clusters:
                                    'RA', 'DEC', 'R500', 'M500', 'Z'
            RM50_no_planck (pd.DataFrame): dataframe with the following columns for RedMaPPer clusters with lambda>50:
                                    'RA', 'DEC', 'LAMBDA', 'Z'
            RM30_no_planck (pd.DataFrame): dataframe with the following columns for RedMaPPer clusters with lambda>30:
                                    'RA', 'DEC', 'LAMBDA', 'Z'

            Catalogs are saved in output_path + /catalogs/. Input catalogs are in planck_path.

        Args:
            plot (bool, optional): If True, will save duplicates distance from each other distribution plots. Defaults to False.

        """

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
        MCXC_skycoord = SkyCoord(ra=MCXC[1].data['RA'].tolist(), dec=MCXC[1].data['DEC'].tolist(), unit=u.degree)
        MCXC_GLON = list(MCXC_skycoord.galactic.l.degree)
        MCXC_GLAT = list(MCXC_skycoord.galactic.b.degree)
        df_MCXC = pd.DataFrame(data={'RA': MCXC[1].data['RA'].tolist(), 'DEC': MCXC[1].data['DEC'].tolist(), 'R500': MCXC[1].data['RADIUS_500'].tolist(), 'M500': MCXC[1].data['MASS_500'].tolist(),
            'GLON': MCXC_GLON, 'GLAT': MCXC_GLAT, 'Z': MCXC[1].data['REDSHIFT'].tolist()})

        REDMAPPER = fits.open(self.planck_path + 'redmapper_dr8_public_v6.3_catalog.fits')
        REDMAPPER_skycoord = SkyCoord(ra=REDMAPPER[1].data['RA'].tolist(), dec=REDMAPPER[1].data['DEC'].tolist(), unit=u.degree)
        REDMAPPER_GLON = list(REDMAPPER_skycoord.galactic.l.degree)
        REDMAPPER_GLAT = list(REDMAPPER_skycoord.galactic.b.degree)
        df_REDMAPPER = pd.DataFrame(data={'RA': REDMAPPER[1].data['RA'].tolist(), 'DEC': REDMAPPER[1].data['DEC'].tolist(), 'LAMBDA': REDMAPPER[1].data['LAMBDA'].tolist(),
            'GLON': REDMAPPER_GLON, 'GLAT': REDMAPPER_GLAT, 'Z': REDMAPPER[1].data['Z_SPEC'].tolist()})

        df_REDMAPPER_30 = df_REDMAPPER.query("LAMBDA > 30")
        df_REDMAPPER_50 = df_REDMAPPER.query("LAMBDA > 50")

        ACT = fits.open(self.planck_path + 'sptecs_catalog_oct919_forSZDB.fits')
        SPT = fits.open(self.planck_path + 'DR5_cluster-catalog_v1.1_forSZDB.fits')
        df_act = pd.DataFrame(data={'RA': list(ACT[1].data['RA']), 'DEC': list(ACT[1].data['DEC']), 'GLON': list(ACT[1].data['GLON']), 'GLAT': list(ACT[1].data['GLAT'])})
        df_spt = pd.DataFrame(data={'RA': list(SPT[1].data['RA']), 'DEC': list(SPT[1].data['DEC']), 'GLON': list(SPT[1].data['GLON']), 'GLAT': list(SPT[1].data['GLAT'])})

        self.remove_duplicates_on_radec(df_MCXC, df_psz2, output_name='MCXC_no_planck', plot=plot)
        self.remove_duplicates_on_radec(df_REDMAPPER_30, df_psz2, output_name='RM30_no_planck', plot=plot)
        self.remove_duplicates_on_radec(df_REDMAPPER_50, df_psz2, output_name='RM50_no_planck', plot=plot)
        self.remove_duplicates_on_radec(df_act, df_psz2, output_name='ACT_no_planck', plot=plot)
        self.remove_duplicates_on_radec(df_spt, df_psz2, output_name='SPT_no_planck', plot=plot)

        PSZ2.close()
        MCXC.close()
        MCXC.close()
        REDMAPPER.close()
        ACT.close()
        SPT.close()


    def create_fake_source_catalog(self):
        
        PGCC = fits.open(self.planck_path + 'HFI_PCCS_GCC_R2.02.fits')
        df_pgcc = pd.DataFrame(data={'RA': list(PGCC[1].data['RA']), 'DEC': list(PGCC[1].data['DEC']), 'GLON': list(PGCC[1].data['GLON']), 'GLAT': list(PGCC[1].data['GLAT'])})
        PGCC.close()
        df_pgcc.to_csv(self.path + 'catalogs/' + 'PGCC' + '.csv', index=False)

        df = pd.DataFrame(columns=['RA','DEC','GLON','GLAT'])

        bands = ['100GHz', '143GHz', '217GHz', '353GHz', '545GHz', '857GHz']
        cs_100 = fits.open(self.planck_path + 'COM_PCCS_100_R2.01.fits')
        cs_143 = fits.open(self.planck_path + 'COM_PCCS_143_R2.01.fits')
        cs_217 = fits.open(self.planck_path + 'COM_PCCS_217_R2.01.fits')
        cs_353 = fits.open(self.planck_path + 'COM_PCCS_353_R2.01.fits')
        cs_545 = fits.open(self.planck_path + 'COM_PCCS_545_R2.01.fits')
        cs_857 = fits.open(self.planck_path + 'COM_PCCS_857_R2.01.fits')

        df_cs_100 = pd.DataFrame(data={'RA': list(cs_100[1].data['RA']), 'DEC': list(cs_100[1].data['DEC']), 'GLON': list(cs_100[1].data['GLON']), 'GLAT': list(cs_100[1].data['GLAT'])})
        df_cs_100.to_csv(self.path + 'catalogs/' + 'cs_100' + '.csv', index=False)
        df_cs_143 = pd.DataFrame(data={'RA': list(cs_143[1].data['RA']), 'DEC': list(cs_143[1].data['DEC']), 'GLON': list(cs_143[1].data['GLON']), 'GLAT': list(cs_143[1].data['GLAT'])})
        df_cs_143.to_csv(self.path + 'catalogs/' + 'cs_143' + '.csv', index=False)
        df_cs_217 = pd.DataFrame(data={'RA': list(cs_217[1].data['RA']), 'DEC': list(cs_217[1].data['DEC']), 'GLON': list(cs_217[1].data['GLON']), 'GLAT': list(cs_217[1].data['GLAT'])})
        df_cs_217.to_csv(self.path + 'catalogs/' + 'cs_217' + '.csv', index=False)
        df_cs_353 = pd.DataFrame(data={'RA': list(cs_353[1].data['RA']), 'DEC': list(cs_353[1].data['DEC']), 'GLON': list(cs_353[1].data['GLON']), 'GLAT': list(cs_353[1].data['GLAT'])})
        df_cs_353.to_csv(self.path + 'catalogs/' + 'cs_353' + '.csv', index=False)
        df_cs_545 = pd.DataFrame(data={'RA': list(cs_545[1].data['RA']), 'DEC': list(cs_545[1].data['DEC']), 'GLON': list(cs_545[1].data['GLON']), 'GLAT': list(cs_545[1].data['GLAT'])})
        df_cs_545.to_csv(self.path + 'catalogs/' + 'cs_545' + '.csv', index=False)
        df_cs_857 = pd.DataFrame(data={'RA': list(cs_857[1].data['RA']), 'DEC': list(cs_857[1].data['DEC']), 'GLON': list(cs_857[1].data['GLON']), 'GLAT': list(cs_857[1].data['GLAT'])})
        df_cs_857.to_csv(self.path + 'catalogs/' + 'cs_857' + '.csv', index=False)

        
        freq = 0
        if '100GHz' in bands:
            freq += 2
            df = pd.concat((df, pd.DataFrame(data={'RA': list(cs_100[1].data['RA']), 'DEC': list(cs_100[1].data['DEC']), 'GLON': list(cs_100[1].data['GLON']), 'GLAT': list(cs_100[1].data['GLAT'])})))
        if '143GHz' in bands:
            freq += 4
            df = pd.concat((df, pd.DataFrame(data={'RA': list(cs_143[1].data['RA']), 'DEC': list(cs_143[1].data['DEC']), 'GLON': list(cs_143[1].data['GLON']), 'GLAT': list(cs_143[1].data['GLAT'])})))
        if '217GHz' in bands:
            freq += 8
            df = pd.concat((df, pd.DataFrame(data={'RA': list(cs_217[1].data['RA']), 'DEC': list(cs_217[1].data['DEC']), 'GLON': list(cs_217[1].data['GLON']), 'GLAT': list(cs_217[1].data['GLAT'])})))
        if '353GHz' in bands:
            freq += 16
            df = pd.concat((df, pd.DataFrame(data={'RA': list(cs_353[1].data['RA']), 'DEC': list(cs_353[1].data['DEC']), 'GLON': list(cs_353[1].data['GLON']), 'GLAT': list(cs_353[1].data['GLAT'])})))
        if '545GHz' in bands:
            freq += 32
            df = pd.concat((df, pd.DataFrame(data={'RA': list(cs_545[1].data['RA']), 'DEC': list(cs_545[1].data['DEC']), 'GLON': list(cs_545[1].data['GLON']), 'GLAT': list(cs_545[1].data['GLAT'])})))
        if '857GHz' in bands:
            freq += 64
            df = pd.concat((df, pd.DataFrame(data={'RA': list(cs_857[1].data['RA']), 'DEC': list(cs_857[1].data['DEC']), 'GLON': list(cs_857[1].data['GLON']), 'GLAT': list(cs_857[1].data['GLAT'])})))
        
        df = pd.concat((df_pgcc, df))
        df = self.remove_duplicates_on_radec(df, with_itself=True, tol=2)
        df.to_csv(self.path + 'catalogs/' + 'False_SZ_catalog_f%s'%freq + '.csv', index=False)

        df = pd.DataFrame(columns=['RA','DEC','GLON','GLAT'])
        for L in range(1, len(bands)):
            for subset in tqdm(itertools.combinations(bands, L)):
                freq = 0
                if '100GHz' in subset:
                    freq += 2
                    df = pd.concat((df, pd.DataFrame(data={'RA': list(cs_100[1].data['RA']), 'DEC': list(cs_100[1].data['DEC']), 'GLON': list(cs_100[1].data['GLON']), 'GLAT': list(cs_100[1].data['GLAT'])})))
                if '143GHz' in subset:
                    freq += 4
                    df = pd.concat((df, pd.DataFrame(data={'RA': list(cs_143[1].data['RA']), 'DEC': list(cs_143[1].data['DEC']), 'GLON': list(cs_143[1].data['GLON']), 'GLAT': list(cs_143[1].data['GLAT'])})))
                if '217GHz' in subset:
                    freq += 8
                    df = pd.concat((df, pd.DataFrame(data={'RA': list(cs_217[1].data['RA']), 'DEC': list(cs_217[1].data['DEC']), 'GLON': list(cs_217[1].data['GLON']), 'GLAT': list(cs_217[1].data['GLAT'])})))
                if '353GHz' in subset:
                    freq += 16
                    df = pd.concat((df, pd.DataFrame(data={'RA': list(cs_353[1].data['RA']), 'DEC': list(cs_353[1].data['DEC']), 'GLON': list(cs_353[1].data['GLON']), 'GLAT': list(cs_353[1].data['GLAT'])})))
                if '545GHz' in subset:
                    freq += 32
                    df = pd.concat((df, pd.DataFrame(data={'RA': list(cs_545[1].data['RA']), 'DEC': list(cs_545[1].data['DEC']), 'GLON': list(cs_545[1].data['GLON']), 'GLAT': list(cs_545[1].data['GLAT'])})))
                if '857GHz' in subset:
                    freq += 64
                    df = pd.concat((df, pd.DataFrame(data={'RA': list(cs_857[1].data['RA']), 'DEC': list(cs_857[1].data['DEC']), 'GLON': list(cs_857[1].data['GLON']), 'GLAT': list(cs_857[1].data['GLAT'])})))
                
                df = pd.concat((df_pgcc, df))
                df = self.remove_duplicates_on_radec(df, with_itself=True, tol=2)
                df.to_csv(self.path + 'catalogs/' + 'False_SZ_catalog_f%s'%freq + '.csv', index=False)
    
        cs_100.close()
        cs_143.close()
        cs_217.close()
        cs_353.close()
        cs_545.close()
        cs_857.close()


    def remove_duplicates_on_radec(self, df_main, df_with_dup=None, output_name=None, with_itself=False, tol=5, plot=False):
        """"Takes two different dataframes with columns 'RA' & 'DEC' and performs a spatial
        coordinate match with a tol=5 arcmin tolerance. Saves a .csv file containing df_main 
        without objects in common from df_with_dup.
        
        Args:
            df_main (pd.DataFrame): main dataframe.
            df_with_dup (pd.DataFrame): dataframe that contains objects from df_main. Defaults to None.
            output_name (str): name that will be used in the saved/plot file name. If None, no file will be saved. Defaults to None.
            with_itself (bool, optional): If True, the spatial coordinates match will be performed with df_main. Defaults to False.
            tol (int, optional): tolerance for spatial coordinates match in arcmin. Defaults to 5.
            plot (bool, optional): If True, will save duplicates distance from each other distribution plots. Defaults to False.
        """

        if with_itself == True:
            scatalog_sub = SkyCoord(ra=df_main['RA'].values, dec=df_main['DEC'].values, unit='deg')
            idx, d2d, _ = match_coordinates_sky(scatalog_sub, scatalog_sub, nthneighbor=2)
            ismatched = d2d < tol*u.arcminute #threshold to consider whether or not two galaxies are the same
            df_d2d = pd.DataFrame(data={'ismatched': ismatched, 'idx': idx, 'd2d': d2d})
            df_main['ismatched'], df_main['ID'] = ismatched, idx
            df_main.query("ismatched == False", inplace=True)
            df_main.drop(columns=['ismatched', 'ID'], inplace=True)
            df_main = df_main.replace([-1, -10, -99], np.nan)
            if output_name is not None:
                df_main.to_csv(self.path + 'catalogs/' + output_name + '.csv', index=False)

        elif with_itself == False:
            assert df_with_dup is not None

            ID = np.arange(0, len(df_with_dup))
            df_with_dup = df_with_dup[['RA', 'DEC']].copy()
            df_with_dup.insert(loc=0, value=ID, column='ID')

            scatalog_sub = SkyCoord(ra=df_main['RA'].values, dec=df_main['DEC'].values, unit='deg')
            pcatalog_sub = SkyCoord(ra=df_with_dup['RA'].values, dec=df_with_dup['DEC'].values, unit='deg')
            idx, d2d, _ = match_coordinates_sky(scatalog_sub, pcatalog_sub, nthneighbor=1)

            ismatched = d2d < tol*u.arcminute #threshold to consider whether or not two galaxies are the same

            df_d2d = pd.DataFrame(data={'ismatched': ismatched, 'idx': idx, 'd2d': d2d})

            df_main['ismatched'], df_main['ID'] = ismatched, idx

            df_with_dup.drop(columns=['RA', 'DEC'], inplace=True)

            df_wo_dup = pd.merge(df_main, df_with_dup, indicator=True, on='ID', how='outer').query('_merge=="both"').drop('_merge', axis=1)

            df_wo_dup.query("ismatched == False", inplace=True)
            df_wo_dup.drop(columns=['ismatched', 'ID'], inplace=True)
            df_wo_dup = df_wo_dup.replace([-1, -10, -99], np.nan)
            if output_name is not None:
                df_wo_dup.to_csv(self.path + 'catalogs/' + output_name + '.csv', index=False)
            df_main = df_wo_dup.copy()

        if plot == True and output_name is not None:
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
    
        return df_main


    def remove_duplicates_on_lonlat(self, df_main, df_with_dup=None, output_name=None, with_itself=False, tol=2, plot=False):
        """"Takes two different dataframes with columns 'GLON' & 'GLAT' and performs a spatial
        coordinate match with a tol=2 arcmin tolerance. Saves a .csv file containing df_main 
        without objects in common from df_with_dup.
        
        Args:
            df_main (pd.DataFrame): main dataframe.
            output_name (str): name that will be used in the saved/plot file name. If None, no file will be saved. Defaults to None.
            df_with_dup (pd.DataFrame): dataframe that contains objects from df_main. Defaults to None.
            with_itself (bool, optional): If True, the spatial coordinates match will be performed with df_main. Defaults to False.
            tol (int, optional): tolerance for spatial coordinates match in arcmin. Defaults to 2.
            plot (bool, optional): If True, will save duplicates distance from each other distribution plots. Defaults to False.
        """


        if with_itself == True:
            scatalog_sub = SkyCoord(df_main['GLON'].values, df_main['GLAT'].values, unit='deg', frame='galactic')
            idx, d2d, _ = match_coordinates_sky(scatalog_sub, scatalog_sub, nthneighbor=2)
            ismatched = d2d < tol*u.arcminute #threshold to consider whether or not two galaxies are the same
            df_d2d = pd.DataFrame(data={'ismatched': ismatched, 'idx': idx, 'd2d': d2d})
            df_main['ismatched'], df_main['ID'] = ismatched, idx
            df_main.query("ismatched == False", inplace=True)
            df_main.drop(columns=['ismatched', 'ID'], inplace=True)
            df_main = df_main.replace([-1, -10, -99], np.nan)
            if output_name is not None:
                df_main.to_csv(self.path + 'catalogs/' + output_name + '.csv', index=False)

        elif with_itself == False:
            assert df_with_dup is not None

            ID = np.arange(0, len(df_with_dup))
            df_with_dup = df_with_dup[['GLON', 'GLAT']].copy()
            df_with_dup.insert(loc=0, value=ID, column='ID')

            scatalog_sub = SkyCoord(df_main['GLON'].values, df_main['GLAT'].values, unit='deg', frame='galactic')
            pcatalog_sub = SkyCoord(df_with_dup['GLON'].values, df_with_dup['GLAT'].values, unit='deg', frame='galactic')
            idx, d2d, _ = match_coordinates_sky(scatalog_sub, pcatalog_sub, nthneighbor=1)

            ismatched = d2d < tol*u.arcminute #threshold to consider whether or not two galaxies are the same

            df_d2d = pd.DataFrame(data={'ismatched': ismatched, 'idx': idx, 'd2d': d2d})

            df_main['ismatched'], df_main['ID'] = ismatched, idx

            df_with_dup.drop(columns=['GLON', 'GLAT'], inplace=True)

            df_wo_dup = pd.merge(df_main, df_with_dup, indicator=True, on='ID', how='outer').query('_merge=="both"').drop('_merge', axis=1)

            df_wo_dup.query("ismatched == False", inplace=True)
            df_wo_dup.drop(columns=['ismatched', 'ID'], inplace=True)
            df_wo_dup = df_wo_dup.replace([-1, -10, -99], np.nan)
            if output_name is not None:
                df_wo_dup.to_csv(self.path + 'catalogs/' + output_name + '.csv', index=False)
            df_main = df_wo_dup.copy()

        if plot == True and output_name is not None:
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

        return df_main



    def create_circular_mask(self, h, w, center, ang_center, radius):
        """Takes a list of center positions and returns a segmentation mask with circulat masks at the center's 
        position.

        Args:
            h (int): patch height.
            w (int): patch width.
            center (list of tuples): In pixels. List of tupples containing center coordinates to mask.
            ang_center (list of tuples): In ICRS. Same as center
            radius ([type]): In arcmin. Disk radius for mask

        Returns:
            np.ndarray: ndarray with shape (h,w) filled with zeros except at centers position where circular masks 
            with size radius are equal to one.
        """

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
            if np.isnan(c[0]):
                continue
            elif np.isnan(c[1]):
                continue
            else:
                dist_from_center = np.sqrt((X - int(c[0]))**2 + (Y -  int(c[1]))**2)
                mask += (dist_from_center <= radius).astype(int)
                is_all_zero = np.all(((dist_from_center <= radius).astype(int) == 0))
                if is_all_zero == False:
                    count += 1
                    ra.append(ang_center[i][0])
                    dec.append(ang_center[i][1])
        return np.where(mask > 1, 1, mask), count, ra, dec

    def return_coord_catalog(self):
        """
        Returns coordinate catalogs

        Returns:
            DataFrame: cluster coordinate catalog
            DataFrame: other sources coordinate catalog
        """
        if self.dataset == 'planck_z':
            planck_z = pd.read_csv(self.path + 'catalogs/planck_z' + '.csv')
            coord_catalog = planck_z[['RA', 'DEC', 'GLON', 'GLAT']].copy()
        elif self.dataset == 'planck_no-z':
            planck_z = pd.read_csv(self.path + 'catalogs/planck_z' + '.csv')
            planck_no_z = pd.read_csv(self.path + 'catalogs/planck_no-z' + '.csv')
            coord_catalog = pd.concat([planck_z[['RA', 'DEC', 'GLON', 'GLAT']].copy(), planck_no_z[['RA', 'DEC', 'GLON', 'GLAT']].copy()], ignore_index=True)
        elif self.dataset == 'MCXC':
            planck_z = pd.read_csv(self.path + 'catalogs/planck_z' + '.csv')
            planck_no_z = pd.read_csv(self.path + 'catalogs/planck_no-z' + '.csv')
            MCXC = pd.read_csv(self.path + 'catalogs/MCXC_no_planck' + '.csv')
            coord_catalog = pd.concat([planck_z[['RA', 'DEC', 'GLON', 'GLAT']].copy(), planck_no_z[['RA', 'DEC', 'GLON', 'GLAT']].copy(), MCXC[['RA', 'DEC', 'GLON', 'GLAT']].copy()],
                ignore_index=True)
        elif self.dataset == 'RM30':
            planck_z = pd.read_csv(self.path + 'catalogs/planck_z' + '.csv')
            planck_no_z = pd.read_csv(self.path + 'catalogs/planck_no-z' + '.csv')
            MCXC = pd.read_csv(self.path + 'catalogs/MCXC_no_planck' + '.csv')
            RM30 = pd.read_csv(self.path + 'catalogs/RM30_no_planck' + '.csv')
            coord_catalog = pd.concat([planck_z[['RA', 'DEC', 'GLON', 'GLAT']].copy(), planck_no_z[['RA', 'DEC', 'GLON', 'GLAT']].copy(), MCXC[['RA', 'DEC', 'GLON', 'GLAT']].copy(),
                RM30[['RA', 'DEC']].copy()], ignore_index=True)
        elif self.dataset == 'RM50':
            planck_z = pd.read_csv(self.path + 'catalogs/planck_z' + '.csv')
            planck_no_z = pd.read_csv(self.path + 'catalogs/planck_no-z' + '.csv')
            MCXC = pd.read_csv(self.path + 'catalogs/MCXC_no_planck' + '.csv')
            RM50 = pd.read_csv(self.path + 'catalogs/RM50_no_planck' + '.csv')
            coord_catalog = pd.concat([planck_z[['RA', 'DEC', 'GLON', 'GLAT']].copy(), planck_no_z[['RA', 'DEC', 'GLON', 'GLAT']].copy(), MCXC[['RA', 'DEC', 'GLON', 'GLAT']].copy(),
                RM50[['RA', 'DEC']].copy()], ignore_index=True)

        false_catalog = pd.read_csv(self.path + 'catalogs/False_SZ_catalog_f%s.csv'%self.planck_freq)
        cold_cores = pd.read_csv(self.path + 'catalogs/PGCC.csv')
        # cs_100 = pd.read_csv(self.path + 'catalogs/cs_100.csv')
        # cs_143 = pd.read_csv(self.path + 'catalogs/cs_143.csv')
        # cs_217 = pd.read_csv(self.path + 'catalogs/cs_217.csv')
        # cs_343 = pd.read_csv(self.path + 'catalogs/cs_353.csv')
        # cs_545 = pd.read_csv(self.path + 'catalogs/cs_545.csv')
        # cs_857 = pd.read_csv(self.path + 'catalogs/cs_857.csv')

        return coord_catalog, false_catalog, cold_cores#, cs_100

    def rotate(self, origin, point, angle):
        """
        Rotate a point clockwise by a given angle around a given origin.
        The angle should be given in radians.

        Args:
            origin ([type]): [description]
            point ([type]): [description]
            angle ([type]): [description]

        Returns:
            [type]: [description]
        """
        angle = -np.radians(angle) #transform in radians, - sign is there to ensure clockwise

        ox, oy = origin
        px, py = point

        qx = ox + np.cos(angle) * (px - ox) - np.sin(angle) * (py - oy)
        qy = oy + np.sin(angle) * (px - ox) + np.cos(angle) * (py - oy)
        return qx, qy

    def rotate_patch(self, p, i, band, patch_rot, coord_catalog, random_angle, n_rot):
        """[summary]

        Args:
            p (int): [description]
            i (int): [description]
            band (int): [description]
            patch_rot ([type]): [description]
            coord_catalog ([type]): [description]
            random_angle ([type]): [description]
            n_rot ([type]): [description]

        Returns:
            [type]: [description]
        """

        HDU_rot = patch_rot[band]['fits']
        HDU_rot_data = ndimage.rotate(np.array(HDU_rot.data), random_angle, reshape=False)
        wcs_rot = WCS(HDU_rot.header)
        x_rot,y_rot = wcs_rot.world_to_pixel_values(coord_catalog['RA'].values[i], coord_catalog['DEC'].values[i])
        x_rot, y_rot = self.rotate(origin=(0.5*self.npix*np.sqrt(2), 0.5*self.npix*np.sqrt(2)), point=(x_rot,y_rot), angle=random_angle)
        x_rot, y_rot = x_rot-int(0.5*self.npix*(np.sqrt(2)-1)), y_rot-int(0.5*self.npix*(np.sqrt(2)-1))

        if x_rot < 0 or x_rot > self.npix or y_rot < 0 or y_rot > self.npix:
            np.random.seed(p+i+200)
            random_int = np.random.randint(300,400)
            random_index = 0
            while x_rot < 0 or x_rot > self.npix or y_rot < 0 or y_rot > self.npix:
                np.random.seed(random_int+random_index)
                random_angle = 360*float(np.random.rand(1))

                HDU_rot_data = ndimage.rotate(np.array(HDU_rot.data), random_angle, reshape=False)
                wcs_rot = WCS(HDU_rot.header)
                x_rot,y_rot = wcs_rot.world_to_pixel_values(coord_catalog['RA'].values[i], coord_catalog['DEC'].values[i])
                x_rot, y_rot = self.rotate(origin=(0.5*self.npix*np.sqrt(2), 0.5*self.npix*np.sqrt(2)), point=(x_rot,y_rot), angle=random_angle)
                x_rot, y_rot = x_rot-int(0.5*self.npix*(np.sqrt(2)-1)), y_rot-int(0.5*self.npix*(np.sqrt(2)-1))
                random_index += 1

        if len(HDU_rot_data[n_rot:-n_rot, n_rot:-n_rot]) != self.npix:
            return HDU_rot_data[n_rot:-n_rot-1, n_rot:-n_rot-1]
        else:
            return HDU_rot_data[n_rot:-n_rot, n_rot:-n_rot]


    def center_of_mass(self, input):
        """
        Calculate the center of mass of the values of a coordinate array.

        Args:
            input (ndarray): Data from which to calculate center-of-mass.

        Returns:
            tuple: Coordinates of centers-of-mass.
        """

        input_mod_sphere = np.zeros_like(input)
        for i,coordinate in enumerate(input):
            if np.abs(coordinate[0]) > 360 - 0.1*self.ndeg:
                input_mod_sphere[i,0] = coordinate[0] - 360
            else: 
                input_mod_sphere[i,0] = coordinate[0]
            if coordinate[1] < -90 + 0.1*self.ndeg:
                input_mod_sphere[i,1] = coordinate[1] + 180
            else:
                input_mod_sphere[i,1] = coordinate[1]

        results = np.mean(input_mod_sphere, axis=0)

        if np.isscalar(results[0]):
            if results[0] > 360 and results[1] > 90:
                return tuple((results[1]-360, results[0]-180))
            elif results[0] > 360 and results[1] < 90:
                return tuple((results[1]-360, results[0]))
            elif results[0] < 360 and results[1] > 90:
                return tuple((results[1], results[0]-180))
            elif results[0] < 360 and results[1] < 90:
                return tuple((results[0], results[1]))
            else:
                raise ValueError("coordinates are not in the right format")
        else:
            raise ValueError("input has wrong dimensions")


    def neighbours(self, catalog_list, coord_list):

        assert len(catalog_list) == len(coord_list)

        close_neighbours_id_list, close_coord_neighbours_list, coord_neighbours_list, cluster_density_list = [], [], [], []
        for h in range(len(catalog_list)):
            close_neighbours_id, close_coord_neighbours, coord_neighbours, cluster_density = [], [], [], []
            for j in range(len(catalog_list)):

                idx_cluster_list = []
                idx_false_cluster_list = []
                if h == j:
                    ini = 2
                else:
                    ini = 1
                for k in range(ini,100):
                    idx, _, _ = match_coordinates_sky(coord_list[h], coord_list[j], nthneighbor=k)
                    idx_cluster_list.append(idx)

                for i in range(len(catalog_list[h])):
                    close_neighb_id = [i]
                    close_neighb = [[catalog_list[h]['RA'].values[i], catalog_list[j]['DEC'].values[i]]]
                    ## Match between galaxy clusters and themselves 
                    k = 0
                    idx = idx_cluster_list[k]
                    ra_diff = np.abs(catalog_list[h]['RA'].values[i] - catalog_list[j]['RA'].values[idx[i]])
                    if np.abs(np.abs(catalog_list[h]['RA'].values[i] - catalog_list[j]['RA'].values[idx[i]])-360) < ra_diff:
                        ra_diff = np.abs(np.abs(catalog_list[h]['RA'].values[i] - catalog_list[j]['RA'].values[idx[i]]) - 360)
                    dec_diff = np.abs(catalog_list[h]['DEC'].values[i] - catalog_list[j]['DEC'].values[idx[i]])     
                    if np.abs(np.abs(catalog_list[h]['DEC'].values[i] - catalog_list[j]['DEC'].values[idx[i]]) - 180) < dec_diff:
                        dec_diff = np.abs(np.abs(catalog_list[h]['DEC'].values[i] - catalog_list[j]['DEC'].values[idx[i]]) - 180)

                    if ra_diff < 0.7*self.ndeg and dec_diff < 0.7*self.ndeg:
                        close_neighb_id.append(idx[i])
                        close_neighb.append([catalog_list[h]['RA'].values[idx[i]], catalog_list[j]['DEC'].values[idx[i]]])
                    k += 1
                    neighb = [[catalog_list[h]['RA'].values[idx[i]], catalog_list[j]['DEC'].values[idx[i]]]]
                    while ra_diff < 1.5*self.ndeg and dec_diff < 1.5*self.ndeg:
                        # idx, _, _ = match_coordinates_sky(coords_ns, coords_ns, nthneighbor=k)
                        idx = idx_cluster_list[k]
                        ra_diff = np.abs(catalog_list[h]['RA'].values[i] - catalog_list[j]['RA'].values[idx[i]])
                        if np.abs(np.abs(catalog_list[h]['RA'].values[i] - catalog_list[j]['RA'].values[idx[i]]) - 360) < ra_diff:
                            ra_diff = np.abs(np.abs(catalog_list[h]['RA'].values[i] - catalog_list[j]['RA'].values[idx[i]]) - 360)
                        dec_diff = np.abs(catalog_list[h]['DEC'].values[i] - catalog_list[j]['DEC'].values[idx[i]])
                        if np.abs(np.abs(catalog_list[h]['DEC'].values[i] - catalog_list[j]['DEC'].values[idx[i]]) - 180) < dec_diff:
                            dec_diff = np.abs(np.abs(catalog_list[h]['DEC'].values[i] - catalog_list[j]['DEC'].values[idx[i]]) - 180)

                        if ra_diff < 0.7*self.ndeg and dec_diff < 0.7*self.ndeg:
                            close_neighb_id.append(idx[i])
                            close_neighb.append([catalog_list[h]['RA'].values[idx[i]], catalog_list[j]['DEC'].values[idx[i]]])
                        neighb.append([catalog_list[h]['RA'].values[idx[i]], catalog_list[j]['DEC'].values[idx[i]]])
                        k += 1
                    close_neighbours_id.append(close_neighb_id)
                    close_coord_neighbours.append(close_neighb)
                    coord_neighbours.append(neighb)
                    cluster_density.append(k-ini+1)
                close_neighbours_id_list.append(close_neighbours_id)
                close_coord_neighbours_list.append(close_coord_neighbours)
                coord_neighbours_list.append(coord_neighbours)
                cluster_density_list.append(cluster_density)

            return close_neighbours_id_list, close_coord_neighbours_list, coord_neighbours_list, cluster_density_list

            ## Match between galaxy clusters and cold cores
            # idx, _, _ = match_coordinates_sky(coords_ns, cold_cores_coords, nthneighbor=1)
            k = 0
            idx = idx_false_cluster_list[k]
            ra_diff = np.abs(coord_catalog['RA'].values[i] - cold_cores_catalog['RA'].values[idx[i]])
            if np.abs(np.abs(coord_catalog['RA'].values[i] - cold_cores_catalog['RA'].values[idx[i]]) - 360) < ra_diff:
                ra_diff = np.abs(np.abs(coord_catalog['RA'].values[i] - cold_cores_catalog['RA'].values[idx[i]]) - 360)
            dec_diff = np.abs(coord_catalog['DEC'].values[i] - cold_cores_catalog['DEC'].values[idx[i]])     
            if np.abs(np.abs(coord_catalog['DEC'].values[i] - cold_cores_catalog['DEC'].values[idx[i]]) - 180) < dec_diff:
                dec_diff = np.abs(np.abs(coord_catalog['DEC'].values[i] - cold_cores_catalog['DEC'].values[idx[i]]) - 180)       
            k += 1
            neighb = [[cold_cores_catalog['RA'].values[idx[i]], cold_cores_catalog['DEC'].values[idx[i]]]]
            while ra_diff < 1.5*self.ndeg and dec_diff < 1.5*self.ndeg:
                idx = idx_false_cluster_list[k]
                # idx, _, _ = match_coordinates_sky(coords_ns, cold_cores_coords, nthneighbor=k)
                ra_diff = np.abs(coord_catalog['RA'].values[i] - cold_cores_catalog['RA'].values[idx[i]])
                if np.abs(np.abs(coord_catalog['RA'].values[i] - cold_cores_catalog['RA'].values[idx[i]]) - 360) < ra_diff:
                    ra_diff = np.abs(np.abs(coord_catalog['RA'].values[i] - cold_cores_catalog['RA'].values[idx[i]]) - 360)
                dec_diff = np.abs(coord_catalog['DEC'].values[i] - cold_cores_catalog['DEC'].values[idx[i]])
                if np.abs(np.abs(coord_catalog['DEC'].values[i] - cold_cores_catalog['DEC'].values[idx[i]]) - 180) < dec_diff:
                    dec_diff = np.abs(np.abs(coord_catalog['DEC'].values[i] - cold_cores_catalog['DEC'].values[idx[i]]) - 180) 
                neighb.append([cold_cores_catalog['RA'].values[idx[i]], cold_cores_catalog['DEC'].values[idx[i]]])
                k += 1
            false_coord_neighbours.append(neighb)
            false_cluster_density.append(k-1)

    def make_input(self, p, cold_cores=False, label_only=False, save_files=True, plot=False, verbose=False):
        """
        Creates input/output datasets for all clusters in the selected cluster catalog. Patches contain at least 
        one cluster that underwent a random translation. It also saves a .npz file containing a list of str in which
        training, validation and test belonging is specified.

        Args:
            p (int): loop number.
            plot (bool, optional): If True, will plot the number of potential objects per patch. Defaults to False.
            verbose (bool, optional): If True, will print additional information. Defaults to False.
        """

        #------------------------------------------------------------------#
        # # # # # Create common catalog # # # # #
        #------------------------------------------------------------------#

        if p != 0:
            plot = False

        coord_catalog, false_catalog, cold_cores_catalog = self.return_coord_catalog()
        false_catalog_list = [cold_cores_catalog]#, cs_100]

        input_size = len(coord_catalog)
        coords_ns = SkyCoord(ra=coord_catalog['RA'].values, dec=coord_catalog['DEC'].values, unit='deg')
    

        #------------------------------------------------------------------#
        # # # # # Check for potential neighbours # # # # #
        #------------------------------------------------------------------#

        # false_coords = SkyCoord(ra=false_catalog['RA'].values, dec=false_catalog['DEC'].values, unit='deg')
        cold_cores_coords = SkyCoord(ra=cold_cores_catalog['RA'].values, dec=cold_cores_catalog['DEC'].values, unit='deg')
        # cs_100_coords = SkyCoord(ra=cs_100['RA'].values, dec=cs_100['DEC'].values, unit='deg')
        false_coords_list = [cold_cores_coords]#, cs_100_coords]

        cluster_density = []
        false_cluster_density = []
        coord_neighbours = []
        close_coord_neighbours = []
        false_coord_neighbours = []
        close_neighbours_id_list = []

        idx_cluster_list = []
        idx_false_cluster_list = []
        for k in range(2,10):
            idx, _, _ = match_coordinates_sky(coords_ns, coords_ns, nthneighbor=k)
            idx_cluster_list.append(idx)
        for k in range(1,100):
            idx, _, _ = match_coordinates_sky(coords_ns, cold_cores_coords, nthneighbor=k)
            idx_false_cluster_list.append(idx)

        for i in range(input_size):
            close_neighbours_id = [i]
            close_neighb = [[coord_catalog['RA'].values[i], coord_catalog['DEC'].values[i]]]
            ## Match between galaxy clusters and themselves 
            # idx, _, _ = match_coordinates_sky(coords_ns, coords_ns, nthneighbor=2)
            k = 0
            idx = idx_cluster_list[k]
            ra_diff = np.abs(coord_catalog['RA'].values[i] - coord_catalog['RA'].values[idx[i]])
            if np.abs(np.abs(coord_catalog['RA'].values[i] - coord_catalog['RA'].values[idx[i]])-360) < ra_diff:
                ra_diff = np.abs(np.abs(coord_catalog['RA'].values[i] - coord_catalog['RA'].values[idx[i]]) - 360)
            dec_diff = np.abs(coord_catalog['DEC'].values[i] - coord_catalog['DEC'].values[idx[i]])     
            if np.abs(np.abs(coord_catalog['DEC'].values[i] - coord_catalog['DEC'].values[idx[i]]) - 180) < dec_diff:
                dec_diff = np.abs(np.abs(coord_catalog['DEC'].values[i] - coord_catalog['DEC'].values[idx[i]]) - 180)

            if ra_diff < 0.7*self.ndeg and dec_diff < 0.7*self.ndeg:
                close_neighbours_id.append(idx[i])
                close_neighb.append([coord_catalog['RA'].values[idx[i]], coord_catalog['DEC'].values[idx[i]]])
            k += 1
            neighb = [[coord_catalog['RA'].values[idx[i]], coord_catalog['DEC'].values[idx[i]]]]
            while ra_diff < 1.5*self.ndeg and dec_diff < 1.5*self.ndeg:
                # idx, _, _ = match_coordinates_sky(coords_ns, coords_ns, nthneighbor=k)
                idx = idx_cluster_list[k]
                ra_diff = np.abs(coord_catalog['RA'].values[i] - coord_catalog['RA'].values[idx[i]])
                if np.abs(np.abs(coord_catalog['RA'].values[i] - coord_catalog['RA'].values[idx[i]]) - 360) < ra_diff:
                    ra_diff = np.abs(np.abs(coord_catalog['RA'].values[i] - coord_catalog['RA'].values[idx[i]]) - 360)
                dec_diff = np.abs(coord_catalog['DEC'].values[i] - coord_catalog['DEC'].values[idx[i]])
                if np.abs(np.abs(coord_catalog['DEC'].values[i] - coord_catalog['DEC'].values[idx[i]]) - 180) < dec_diff:
                    dec_diff = np.abs(np.abs(coord_catalog['DEC'].values[i] - coord_catalog['DEC'].values[idx[i]]) - 180)

                if ra_diff < 0.7*self.ndeg and dec_diff < 0.7*self.ndeg:
                    close_neighbours_id.append(idx[i])
                    close_neighb.append([coord_catalog['RA'].values[idx[i]], coord_catalog['DEC'].values[idx[i]]])
                neighb.append([coord_catalog['RA'].values[idx[i]], coord_catalog['DEC'].values[idx[i]]])
                k += 1
            close_neighbours_id_list.append(close_neighbours_id)
            close_coord_neighbours.append(close_neighb)
            coord_neighbours.append(neighb)
            cluster_density.append(k)

            ## Match between galaxy clusters and cold cores
            # idx, _, _ = match_coordinates_sky(coords_ns, cold_cores_coords, nthneighbor=1)
            k = 0
            idx = idx_false_cluster_list[k]
            ra_diff = np.abs(coord_catalog['RA'].values[i] - cold_cores_catalog['RA'].values[idx[i]])
            if np.abs(np.abs(coord_catalog['RA'].values[i] - cold_cores_catalog['RA'].values[idx[i]]) - 360) < ra_diff:
                ra_diff = np.abs(np.abs(coord_catalog['RA'].values[i] - cold_cores_catalog['RA'].values[idx[i]]) - 360)
            dec_diff = np.abs(coord_catalog['DEC'].values[i] - cold_cores_catalog['DEC'].values[idx[i]])     
            if np.abs(np.abs(coord_catalog['DEC'].values[i] - cold_cores_catalog['DEC'].values[idx[i]]) - 180) < dec_diff:
                dec_diff = np.abs(np.abs(coord_catalog['DEC'].values[i] - cold_cores_catalog['DEC'].values[idx[i]]) - 180)       
            k += 1
            neighb = [[cold_cores_catalog['RA'].values[idx[i]], cold_cores_catalog['DEC'].values[idx[i]]]]
            while ra_diff < 1.5*self.ndeg and dec_diff < 1.5*self.ndeg:
                idx = idx_false_cluster_list[k]
                # idx, _, _ = match_coordinates_sky(coords_ns, cold_cores_coords, nthneighbor=k)
                ra_diff = np.abs(coord_catalog['RA'].values[i] - cold_cores_catalog['RA'].values[idx[i]])
                if np.abs(np.abs(coord_catalog['RA'].values[i] - cold_cores_catalog['RA'].values[idx[i]]) - 360) < ra_diff:
                    ra_diff = np.abs(np.abs(coord_catalog['RA'].values[i] - cold_cores_catalog['RA'].values[idx[i]]) - 360)
                dec_diff = np.abs(coord_catalog['DEC'].values[i] - cold_cores_catalog['DEC'].values[idx[i]])
                if np.abs(np.abs(coord_catalog['DEC'].values[i] - cold_cores_catalog['DEC'].values[idx[i]]) - 180) < dec_diff:
                    dec_diff = np.abs(np.abs(coord_catalog['DEC'].values[i] - cold_cores_catalog['DEC'].values[idx[i]]) - 180) 
                neighb.append([cold_cores_catalog['RA'].values[idx[i]], cold_cores_catalog['DEC'].values[idx[i]]])
                k += 1
            false_coord_neighbours.append(neighb)
            false_cluster_density.append(k-1)

        if cold_cores:
            idx_cluster_list = []
            idx_false_cluster_list = []
            for k in range(1,10):
                idx, _, _ = match_coordinates_sky(cold_cores_coords, coords_ns, nthneighbor=k)
                idx_cluster_list.append(idx)
            for k in range(2,200):
                idx, _, _ = match_coordinates_sky(cold_cores_coords, cold_cores_coords, nthneighbor=k)
                idx_false_cluster_list.append(idx)

            for i in range(len(cold_cores_catalog)):
                # close_neighbours_id = [i]
                # close_neighb = [[coord_catalog['RA'].values[i], coord_catalog['DEC'].values[i]]]
                ## Match between galaxy clusters and themselves 
                # idx, _, _ = match_coordinates_sky(cold_cores_coords, coords_ns, nthneighbor=1)
                k = 0
                idx = idx_cluster_list[k]
                ra_diff = np.abs(coord_catalog['RA'].values[idx[i]] - cold_cores_catalog['RA'].values[i])
                if np.abs(np.abs(coord_catalog['RA'].values[idx[i]] - cold_cores_catalog['RA'].values[i])-360) < ra_diff:
                    ra_diff = np.abs(np.abs(coord_catalog['RA'].values[idx[i]] - cold_cores_catalog['RA'].values[i]) - 360)
                dec_diff = np.abs(coord_catalog['DEC'].values[idx[i]] - cold_cores_catalog['DEC'].values[i])     
                if np.abs(np.abs(coord_catalog['DEC'].values[idx[i]] - cold_cores_catalog['DEC'].values[i]) - 180) < dec_diff:
                    dec_diff = np.abs(np.abs(coord_catalog['DEC'].values[idx[i]] - cold_cores_catalog['DEC'].values[i]) - 180)

                k += 1
                neighb = [[coord_catalog['RA'].values[idx[i]], coord_catalog['DEC'].values[idx[i]]]]
                while ra_diff < 1.5*self.ndeg and dec_diff < 1.5*self.ndeg:
                    # idx, _, _ = match_coordinates_sky(cold_cores_coords, coords_ns,  nthneighbor=k)
                    idx = idx_cluster_list[k]
                    ra_diff = np.abs(coord_catalog['RA'].values[idx[i]] - cold_cores_catalog['RA'].values[i])
                    if np.abs(np.abs(coord_catalog['RA'].values[idx[i]] - cold_cores_catalog['RA'].values[i]) - 360) < ra_diff:
                        ra_diff = np.abs(np.abs(coord_catalog['RA'].values[idx[i]] - cold_cores_catalog['DEC'].values[i]) - 360)
                    dec_diff = np.abs(coord_catalog['DEC'].values[idx[i]] - cold_cores_catalog['DEC'].values[i])
                    if np.abs(np.abs(coord_catalog['DEC'].values[idx[i]] - cold_cores_catalog['DEC'].values[i]) - 180) < dec_diff:
                        dec_diff = np.abs(np.abs(coord_catalog['DEC'].values[idx[i]] - cold_cores_catalog['DEC'].values[i]) - 180)

                    
                    neighb.append([coord_catalog['RA'].values[idx[i]], coord_catalog['DEC'].values[idx[i]]])
                    k += 1
                
                coord_neighbours.append(neighb)
                cluster_density.append(k-1)

                ## Match between galaxy clusters and cold cores
                close_neighbours_id = [i]
                close_neighb = [[cold_cores_catalog['RA'].values[i], cold_cores_catalog['DEC'].values[i]]]
                k = 0
                idx = idx_false_cluster_list[k]
                # idx, _, _ = match_coordinates_sky(cold_cores_coords, cold_cores_coords, nthneighbor=2)
                ra_diff = np.abs(cold_cores_catalog['RA'].values[i] - cold_cores_catalog['RA'].values[idx[i]])
                if np.abs(np.abs(cold_cores_catalog['RA'].values[i] - cold_cores_catalog['RA'].values[idx[i]]) - 360) < ra_diff:
                    ra_diff = np.abs(np.abs(cold_cores_catalog['RA'].values[i] - cold_cores_catalog['RA'].values[idx[i]]) - 360)
                dec_diff = np.abs(cold_cores_catalog['DEC'].values[i] - cold_cores_catalog['DEC'].values[idx[i]])     
                if np.abs(np.abs(cold_cores_catalog['DEC'].values[i] - cold_cores_catalog['DEC'].values[idx[i]]) - 180) < dec_diff:
                    dec_diff = np.abs(np.abs(cold_cores_catalog['DEC'].values[i] - cold_cores_catalog['DEC'].values[idx[i]]) - 180)   

                if ra_diff < 0.5*self.ndeg and dec_diff < 0.5*self.ndeg:
                    close_neighbours_id.append(idx[i])
                    close_neighb.append([cold_cores_catalog['RA'].values[idx[i]], cold_cores_catalog['DEC'].values[idx[i]]])

                k += 1
                neighb = [[cold_cores_catalog['RA'].values[idx[i]], cold_cores_catalog['DEC'].values[idx[i]]]]
                while ra_diff < 1.5*self.ndeg and dec_diff < 1.5*self.ndeg:
                    # idx, _, _ = match_coordinates_sky(cold_cores_coords, cold_cores_coords, nthneighbor=k)
                    idx = idx_false_cluster_list[k]
                    ra_diff = np.abs(cold_cores_catalog['RA'].values[i] - cold_cores_catalog['RA'].values[idx[i]])
                    if np.abs(np.abs(cold_cores_catalog['RA'].values[i] - cold_cores_catalog['RA'].values[idx[i]]) - 360) < ra_diff:
                        ra_diff = np.abs(np.abs(cold_cores_catalog['RA'].values[i] - cold_cores_catalog['RA'].values[idx[i]]) - 360)
                    dec_diff = np.abs(cold_cores_catalog['DEC'].values[i] - cold_cores_catalog['DEC'].values[idx[i]])
                    if np.abs(np.abs(cold_cores_catalog['DEC'].values[i] - cold_cores_catalog['DEC'].values[idx[i]]) - 180) < dec_diff:
                        dec_diff = np.abs(np.abs(cold_cores_catalog['DEC'].values[i] - cold_cores_catalog['DEC'].values[idx[i]]) - 180) 

                    if ra_diff < 0.5*self.ndeg and dec_diff < 0.5*self.ndeg:
                        close_neighbours_id.append(idx[i])
                        close_neighb.append([cold_cores_catalog['RA'].values[idx[i]], cold_cores_catalog['DEC'].values[idx[i]]])

                    neighb.append([cold_cores_catalog['RA'].values[idx[i]], cold_cores_catalog['DEC'].values[idx[i]]])
                    k += 1
                close_neighbours_id_list.append(close_neighbours_id)
                close_coord_neighbours.append(close_neighb)
                false_coord_neighbours.append(neighb)
                false_cluster_density.append(k)

        #------------------------------------------------------------------------------------------------#
        # # # # # Replace cluster coords that are too close form each other with center of mass # # # # #
        #------------------------------------------------------------------------------------------------#


        df = coord_catalog.copy()
        df = pd.concat((df, cold_cores_catalog), ignore_index=True)
        df = pd.concat((df, pd.DataFrame(data={'coord_neighbours': coord_neighbours, 'cluster_density': cluster_density, 'false_coord_neighbours': false_coord_neighbours,
                                                'false_cluster_density': false_cluster_density})), axis=1)

        df_clusters = df.iloc[:input_size,:].copy()
        if cold_cores:
            df_cold_cores = df.iloc[input_size:,:].copy()
            df_cold_cores.reset_index(drop=True, inplace=True)

        
        removed_index_list = []
        for i in range(input_size):
            if i in removed_index_list:
                continue
            else:
                if len(close_neighbours_id_list[i]) > 1:
                    ra, dec = self.center_of_mass(close_coord_neighbours[i])
                    df_clusters.loc[close_neighbours_id_list[i][0],'RA'] = ra
                    df_clusters.loc[close_neighbours_id_list[i][0],'DEC'] = dec
                    for j,row in enumerate(close_neighbours_id_list[i]):
                        if j > 0:
                            try:
                                df_clusters.drop(row, inplace=True)
                                removed_index_list.append(row)
                            except:
                                pass
        print(df_clusters['false_cluster_density'].isnull().sum())
        print(len(df_clusters))
        print(len(set(removed_index_list)))
        print(df_clusters.head(10))

        if cold_cores:
            removed_index_list = []
            skip_index_list = []
            for i in range(input_size, len(cold_cores_catalog)+input_size):
                if i in removed_index_list:
                    continue
                else:
                    if len(close_neighbours_id_list[i]) > 1 and close_neighbours_id_list[i][0] not in skip_index_list:
                        skip_index_list.append(close_neighbours_id_list[i][0])
                        ra, dec = self.center_of_mass(close_coord_neighbours[i])
                        df_cold_cores.loc[close_neighbours_id_list[i][0],'RA'] = ra
                        df_cold_cores.loc[close_neighbours_id_list[i][0],'DEC'] = dec
                        for j,row in enumerate(close_neighbours_id_list[i]):
                            if j > 0:
                                try:
                                    removed_index_list.append(row)
                                except:
                                    pass

            df_cold_cores.drop(removed_index_list, inplace=True)
            print(df_cold_cores['false_cluster_density'].isnull().sum())
            print(len(df_cold_cores))
            print(len(set(removed_index_list)))
            print(df_cold_cores.head(10))
            
            df_cold_cores = df_cold_cores.sample(frac=0.2, random_state=p)
            df = pd.concat((df_clusters, df_cold_cores))

        if not cold_cores:
            df = df_clusters.copy()


    
        coord_catalog = df[['RA', 'DEC']].copy()
        coord_catalog.reset_index(drop=True, inplace=True)
        coord_neighbours = df['coord_neighbours'].copy()
        coord_neighbours.reset_index(drop=True, inplace=True)
        cluster_density = df['cluster_density'].copy()
        cluster_density.reset_index(drop=True, inplace=True)
        false_coord_neighbours = df['false_coord_neighbours'].copy()
        false_coord_neighbours.reset_index(drop=True, inplace=True)
        false_cluster_density = df['false_cluster_density'].copy()
        false_cluster_density.reset_index(drop=True, inplace=True)

        input_size = len(coord_catalog)

        # if plot == True:
        #     fig = plt.figure(figsize=(7,7), tight_layout=False)
        #     ax = fig.add_subplot(111)
        #     ax.set_facecolor('white')
        #     ax.grid(True, color='grey', lw=0.5)
        #     ax.set_xlabel('Neighbours per patch', fontsize=20)
        #     ax.set_ylabel('Cluster number', fontsize=20)
        #     ax.hist(cluster_density)
        #     ax.set_yscale('log')

        #     plt.savefig(self.output_path + 'figures/' + 'cluster_density' + '.png', bbox_inches='tight', transparent=False)
        #     plt.show()
        #     plt.close()

        #     fig = plt.figure(figsize=(7,7), tight_layout=False)
        #     ax = fig.add_subplot(111)
        #     ax.set_facecolor('white')
        #     ax.grid(True, color='grey', lw=0.5)
        #     ax.set_xlabel('Neighbours per patch', fontsize=20)
        #     ax.set_ylabel('False cluster number', fontsize=20)
        #     ax.hist(false_cluster_density, bins=14)
        #     ax.set_yscale('log')

        #     plt.savefig(self.output_path + 'figures/' + 'false_cluster_density' + '.png', bbox_inches='tight', transparent=False)
        #     plt.show()
        #     plt.close()

        #------------------------------------------------------------------#
        # # # # # Create ramdon coordinate translations # # # # #
        #------------------------------------------------------------------#

        np.random.seed(p)
        random_coord_x = 2*np.random.rand(1, input_size).flatten() - np.ones_like(np.random.rand(1, input_size).flatten())
        np.random.seed(p+100)
        random_coord_y = 2*np.random.rand(1, input_size).flatten() - np.ones_like(np.random.rand(1, input_size).flatten())
        coords = SkyCoord(ra=coord_catalog['RA'].values + 0.5*(self.ndeg-0.2)*random_coord_x,
                          dec=coord_catalog['DEC'].values + 0.5*(self.ndeg-0.2)*random_coord_y, unit='deg') 

        #------------------------------------------------------------------#
        # # # # # Create patch & masks # # # # #
        #------------------------------------------------------------------#

        maps = self.maps
        cutsky = CutSky(maps, npix=self.npix, pixsize=self.pixsize, low_mem=False)
        cutsky_rot = CutSky(maps, npix=int(np.sqrt(2)*self.npix), pixsize=self.pixsize, low_mem=False)

        # skip_regions_up = [[0, 360, -60, -62],
        #                 [0, 360, -60, -62], [0, 360, -60, -62], 
        #                 [0, 360, -60, -62], [0, 360, -60, -62], [0, 360, -60, -62], 
        #                 [0, 360, 62, 60], [0, 360, 62, 60], [0, 360, 62, 60],
        #                 [0, 360, 62, 60], [0, 360, 62, 60], 
        #                 [0, 360, 62, 60]]

        # skip_regions_down = [[0, 360, -38, -40],
        #                 [0, 360, -38, -40], [0, 360, -38, -40], 
        #                 [0,0,0,0], [0,0,0,0], [0,0,0,0], 
        #                 [0,0,0,0], [0,0,0,0], [0,0,0,0], 
        #                 [0, 360, 42, 40], [0, 360, 42, 40],
        #                 [0, 360, 42, 40]]

        ## Skip for test set is done within the region itself, not in skip_regions

        for region,(x_left, x_right, y_up, y_down) in enumerate(self.test_regions):
            if not label_only:
                inputs = np.ndarray((input_size,self.npix,self.npix,len(self.bands)))
            labels = np.ndarray((input_size,self.npix,self.npix,self.n_labels))
            milca = np.ndarray((input_size,self.npix,self.npix,1))
            dataset_type = []

            sum_cluster, sum_false = 0, 0
            for i, coord in enumerate(coords):
                if coord.galactic.l.degree > x_left and coord.galactic.l.degree < x_right and coord.galactic.b.degree < y_up and coord.galactic.b.degree > y_down :
                    dataset_type.append('test')
                    labels[i,:,:,0] = np.zeros((self.npix, self.npix))
                    labels[i,:,:,1] = np.zeros((self.npix, self.npix))
                    milca[i,:,:,0] = np.zeros((self.npix, self.npix))
                    if not label_only:
                        for j in range(len(self.bands)):
                            inputs[i,:,:,j] = np.zeros((self.npix, self.npix))
                    continue

                elif coord.galactic.l.degree > self.val_regions[region][0] and coord.galactic.l.degree < self.val_regions[region][1] and coord.galactic.b.degree < self.val_regions[region][2] and coord.galactic.b.degree > self.val_regions[region][3]:
                    dataset_type.append('val')
                # elif coord.galactic.l.degree > skip_regions_up[region][0] and coord.galactic.l.degree < skip_regions_up[region][1] and coord.galactic.b.degree < skip_regions_up[region][2] and coord.galactic.b.degree > skip_regions_up[region][3]:
                #     dataset_type.append('skip')
                #     labels[i,:,:,0] = np.zeros((self.npix, self.npix))
                #     labels[i,:,:,1] = np.zeros((self.npix, self.npix))
                #     milca[i,:,:,0] = np.zeros((self.npix, self.npix))
                #     if not label_only:
                #         for j in range(len(self.bands)):
                #             inputs[i,:,:,j] = np.zeros((self.npix, self.npix))
                #     continue
                # elif coord.galactic.l.degree > skip_regions_down[region][0] and coord.galactic.l.degree < skip_regions_down[region][1] and coord.galactic.b.degree < skip_regions_down[region][2] and coord.galactic.b.degree > skip_regions_down[region][3]:
                #     dataset_type.append('skip')
                #     labels[i,:,:,0] = np.zeros((self.npix, self.npix))
                #     labels[i,:,:,1] = np.zeros((self.npix, self.npix))
                #     milca[i,:,:,0] = np.zeros((self.npix, self.npix))
                #     if not label_only:
                #         for j in range(len(self.bands)):
                #             inputs[i,:,:,j] = np.zeros((self.npix, self.npix))
                #     continue
                else:
                    dataset_type.append('train')

                #------------------------------------------------------------------#
                # # # # # Rotations # # # # #
                #------------------------------------------------------------------#

                np.random.seed(p+i+200)
                random_angle = 360*float(np.random.rand(1))

                patch = cutsky.cut_fits(coord)
                HDU = patch[-1]['fits']
                wcs = WCS(HDU.header)
                x,y = wcs.world_to_pixel_values(coord_catalog['RA'].values[i], coord_catalog['DEC'].values[i])
                patch_rot = cutsky_rot.cut_fits(coord)
                HDU_rot = patch_rot[-1]['fits']
                HDU_rot_data = ndimage.rotate(np.array(HDU_rot.data), random_angle, reshape=False)

                HDU_rot_data = ndimage.rotate(np.array(HDU_rot.data), random_angle, reshape=False)
                wcs_rot = WCS(HDU_rot.header)
                x_rot,y_rot = wcs_rot.world_to_pixel_values(coord_catalog['RA'].values[i], coord_catalog['DEC'].values[i])
                x_rot, y_rot = self.rotate(origin=(0.5*self.npix*np.sqrt(2), 0.5*self.npix*np.sqrt(2)), point=(x_rot,y_rot), angle=random_angle)
                x_rot, y_rot = x_rot-int(0.5*self.npix*(np.sqrt(2)-1)), y_rot-int(0.5*self.npix*(np.sqrt(2)-1))

                if x_rot < 0 or x_rot > self.npix or y_rot < 0 or y_rot > self.npix:
                    np.random.seed(p+i+200)
                    random_int = np.random.randint(300,400)
                    random_index = 0
                    while x_rot < 0 or x_rot > self.npix or y_rot < 0 or y_rot > self.npix:
                        np.random.seed(random_int+random_index)
                        random_angle = 360*float(np.random.rand(1))

                        HDU_rot_data = ndimage.rotate(np.array(HDU_rot.data), random_angle, reshape=False)
                        wcs_rot = WCS(HDU_rot.header)
                        x_rot,y_rot = wcs_rot.world_to_pixel_values(coord_catalog['RA'].values[i], coord_catalog['DEC'].values[i])
                        x_rot, y_rot = self.rotate(origin=(0.5*self.npix*np.sqrt(2), 0.5*self.npix*np.sqrt(2)), point=(x_rot,y_rot), angle=random_angle)
                        x_rot, y_rot = x_rot-int(0.5*self.npix*(np.sqrt(2)-1)), y_rot-int(0.5*self.npix*(np.sqrt(2)-1))
                        random_index += 1

                h, w = self.npix, self.npix

                if plot == True:
                    if i < len(df_clusters):
                        center = [(x,y)]
                        ang_center = [(coord_catalog['RA'].values[i], coord_catalog['DEC'].values[i])]
                    else:
                        center = []
                        ang_center = []
                    if cluster_density[i] == 0:
                        mask = np.zeros((self.npix, self.npix))
                    elif cluster_density[i] == 1:
                        mask, _, _, _ = self.create_circular_mask(h, w, center=center, ang_center= ang_center, radius=self.disk_radius)
                    else:
                        for j in range(int(cluster_density[i])-1):
                            center.append(wcs.world_to_pixel_values(coord_neighbours[i][j][0], coord_neighbours[i][j][1]))
                            ang_center.append((coord_neighbours[i][j][0], coord_neighbours[i][j][1]))
                        mask, _, _, _ = self.create_circular_mask(h, w, center=center, ang_center= ang_center, radius=self.disk_radius)

                ## CLUSTERS
                if i < len(df_clusters):
                    center = [(x_rot,y_rot)]
                    ang_center = [(coord_catalog['RA'].values[i], coord_catalog['DEC'].values[i])]
                else:
                    center = []
                    ang_center = []
                if cluster_density[i] == 0:
                    mask_rot = np.zeros((self.npix, self.npix))
                    labels[i,:,:,0] = mask_rot.astype(int)
                elif cluster_density[i] == 1:
                    mask_rot, _, _, _ = self.create_circular_mask(h, w, center=center, ang_center= ang_center, radius=self.disk_radius)
                    labels[i,:,:,0] = mask_rot.astype(int)
                else:
                    for j in range(int(cluster_density[i])-1):
                        x_rotn, y_rotn = wcs_rot.world_to_pixel_values(coord_neighbours[i][j][0], coord_neighbours[i][j][1])
                        x_rotn, y_rotn = self.rotate(origin=(0.5*self.npix*np.sqrt(2), 0.5*self.npix*np.sqrt(2)), point=(x_rotn,y_rotn), angle=random_angle)
                        x_rotn, y_rotn = x_rotn-int(0.5*self.npix*(np.sqrt(2)-1)), y_rotn-int(0.5*self.npix*(np.sqrt(2)-1))
                        center.append((x_rotn, y_rotn))
                        ang_center.append((coord_neighbours[i][j][0], coord_neighbours[i][j][1]))
                    mask_rot, _, _, _ = self.create_circular_mask(h, w, center=center, ang_center= ang_center, radius=self.disk_radius)
                    labels[i,:,:,0] = mask_rot.astype(int)

                ## FALSE CLUSTERS
                if i >= len(df_clusters):
                    center = [(x_rot,y_rot)]
                    ang_center = [(coord_catalog['RA'].values[i], coord_catalog['DEC'].values[i])]
                else:
                    center = []
                    ang_center = []
                if false_cluster_density[i] == 0:
                    mask_rot_false = np.zeros((self.npix, self.npix))
                    labels[i,:,:,1] = mask_rot_false
                else:
                    for j in range(int(false_cluster_density[i])):
                        x_rotn, y_rotn = wcs_rot.world_to_pixel_values(false_coord_neighbours[i][j][0], false_coord_neighbours[i][j][1])
                        x_rotn, y_rotn = self.rotate(origin=(0.5*self.npix*np.sqrt(2), 0.5*self.npix*np.sqrt(2)), point=(x_rotn,y_rotn), angle=random_angle)
                        x_rotn, y_rotn = x_rotn-int(0.5*self.npix*(np.sqrt(2)-1)), y_rotn-int(0.5*self.npix*(np.sqrt(2)-1))
                        center.append((x_rotn, y_rotn))
                        ang_center.append((false_coord_neighbours[i][j][0], false_coord_neighbours[i][j][1]))
                    mask_rot_false, _, _, _ = self.create_circular_mask(h, w, center=center, ang_center= ang_center, radius=self.disk_radius)
                    labels[i,:,:,1] = mask_rot_false.astype(int)

                    if verbose:
                        print('\n')
                        print(i)
                        print('cluster density: %s'%cluster_density[i])
                        print('coords no shift: {:.2f}, {:.2f}'.format(coord_catalog['RA'].values[i], coord_catalog['DEC'].values[i]))
                        print('coords shift: {:.2f}, {:.2f}'.format(coord_catalog['RA'].values[i] -30*self.pixsize/60 + (60*self.pixsize/60)*random_coord_x[i], coord_catalog['DEC'].values[i] -30*self.pixsize/60 + (60*self.pixsize/60)*random_coord_y[i]))
                        print(coord_neighbours[i])
                        print(center)
                        print('\n')
                
                n_rot  = int(0.5*self.npix*(np.sqrt(2)-1))
                if len(HDU_rot_data[n_rot:-n_rot, n_rot:-n_rot]) != self.npix:
                    milca[i,:,:,0] = HDU_rot_data[n_rot:-n_rot-1, n_rot:-n_rot-1]
                else:
                    milca[i,:,:,0] = HDU_rot_data[n_rot:-n_rot, n_rot:-n_rot]
                if not label_only:
                    for j in range(len(self.bands)):
                        inputs[i,:,:,j] = self.rotate_patch(p, i, j, patch_rot, coord_catalog, random_angle, n_rot)

                #------------------------------------------------------------------#
                # # # # # Plots # # # # #
                #------------------------------------------------------------------#

                if plot == True:
                    fig = plt.figure(figsize=(25,5), tight_layout=False)

                    ax = fig.add_subplot(151)
                    im = ax.imshow(HDU.data, origin='lower')
                    if i < len(df_clusters):
                        ax.scatter(x,y)
                    ax.set_title('x={:.2f}, y={:.2f}'.format(x,y))

                    ax = fig.add_subplot(152)
                    im = ax.imshow(mask, origin='lower')
                    ax.set_title('x={:.2f}, y={:.2f}'.format(x,y))

                    ax = fig.add_subplot(153)
                    im = ax.imshow(milca[i,:,:,0], origin='lower') #[n_rot:-n_rot, n_rot:-n_rot]
                    if i < len(df_clusters):
                        ax.scatter(x_rot,y_rot)
                    ax.set_title('xrot={:.2f}, yrot={:.2f}'.format(x_rot,y_rot))

                    ax = fig.add_subplot(154)
                    im = ax.imshow(labels[i,:,:,0], origin='lower')
                    ax.set_title('Potential clusters: {:.0f}'.format(cluster_density[i]))

                    ax = fig.add_subplot(155)
                    im = ax.imshow(labels[i,:,:,1], origin='lower')
                    ax.set_title('Potential sources: {:.0f}'.format(false_cluster_density[i]))

                    GenerateFiles.make_directory(self, path_to_file = self.temp_path + 'training_set_r%s_f%s_d%s_s%s_c%s/'%(region, self.freq, self.disk_radius, self.npix, int(cold_cores)))
                    plt.savefig(self.temp_path + 'training_set_r%s_f%s_d%s_s%s_c%s/'%(region, self.freq, self.disk_radius, self.npix, int(cold_cores)) + 'training_%s'%(i) + '.png', bbox_inches='tight', transparent=False)
                    plt.show()
                    plt.close()

                sum_cluster += cluster_density[i]
                sum_false += false_cluster_density[i]

            print(sum_cluster, sum_false)

            #------------------------------------------------------------------#
            # # # # # Save files # # # # #
            #------------------------------------------------------------------#

            assert len(coords) == len(dataset_type)

            # if plot == True:
            #     counts = Counter(dataset_type)
            #     df = pd.DataFrame.from_dict(counts, orient='index')
            #     ax = df.plot(kind='bar')
            #     ax.figure.savefig(self.output_path + 'figures/' + 'dataset_type_density_r%s_f%s_d%s'%(region, self.freq, self.disk_radius) + '.png', bbox_inches='tight', transparent=False)

            if save_files:
                GenerateFiles.make_directory(self, path_to_file = self.output_path + 'files/' + 'r%s_f%s_d%s_s%s_c%s'%(region, self.freq, self.disk_radius, self.npix, int(cold_cores)))
                if not label_only:
                    np.savez_compressed(self.output_path + 'files/r%s_f%s_d%s_s%s_c%s/'%(region, self.freq, self.disk_radius, self.npix, int(cold_cores)) + 'input_n%s_f%s_'%(p, self.freq) + self.dataset, inputs)
                np.savez_compressed(self.output_path + 'files/r%s_f%s_d%s_s%s_c%s/'%(region, self.freq, self.disk_radius, self.npix, int(cold_cores)) + 'milca_n%s_f%s_'%(p, self.freq) + self.dataset, milca)
                np.savez_compressed(self.output_path + 'files/r%s_f%s_d%s_s%s_c%s/'%(region, self.freq, self.disk_radius, self.npix, int(cold_cores)) + 'type_n%s_f%s_'%(p, self.freq) + self.dataset, np.array(dataset_type))
                if p == 0:
                    np.savez_compressed(self.dataset_path + 'type_test_r%s_f%s_c%s_'%(region, self.freq, int(cold_cores)) + self.dataset, np.array(dataset_type))
                np.savez_compressed(self.output_path + 'files/r%s_f%s_d%s_s%s_c%s/'%(region, self.freq, self.disk_radius, self.npix, int(cold_cores)) + 'label_n%s_f%s_'%(p, self.freq) + self.dataset, labels)



    def test_coords(self, x_left, x_right, y_up, y_down):

        width = np.abs(x_right - x_left)
        width_contour = width%self.ndeg
        n_width = width//self.ndeg

        height = np.abs(y_up - y_down)
        height_contour = height%self.ndeg
        n_height = height//self.ndeg

        l, b = [], []
        x = x_left + 0.5*width_contour + 0.5*self.ndeg
        y = y_down + 0.5*height_contour + 0.5*self.ndeg
        l.append(x)
        b.append(y)
        for i in range(int(n_height)):
            if i > 0:
                x = x_left + 0.5*width_contour + 0.5*self.ndeg
                y += self.ndeg
            for j in range(int(n_width)):
                x += self.ndeg
                if i > 0 or j > 0:
                    l.append(x)
                    b.append(y)

        print(self.ndeg, n_height, n_width, n_height * n_width, len(l))
        assert int(n_height * n_width) == len(l)

        coords = SkyCoord(l, b, unit='deg', frame='galactic') 
        catalog = pd.DataFrame(data={'GLON': l, 'GLAT': b})

        # print(catalog.head(60))

        return coords, catalog


    def make_test_input(self, region, x_left, x_right, y_up, y_down, cold_cores=False, label_only=False, plot=False, verbose=False):
        test_coords, test_catalog = self.test_coords(x_left, x_right, y_up, y_down)
        input_size = len(test_coords)
        maps = self.maps
        cutsky = CutSky(maps, npix=self.npix, pixsize=self.pixsize, low_mem=False)

        if not label_only:
            inputs = np.ndarray((len(test_coords),self.npix,self.npix,len(self.bands)))
        labels = np.ndarray((len(test_coords),self.npix,self.npix,2))
        milca = np.ndarray((len(test_coords),self.npix,self.npix,1))

        #------------------------------------------------------------------#
        # # # # # Check for potential neighbours # # # # #
        #------------------------------------------------------------------#

        


        coord_catalog, false_catalog, cold_cores_catalog = self.return_coord_catalog()
        # false_catalog_list = [cold_cores_catalog]
        cold_cores_catalog.query("GLAT > %s"%y_down, inplace=True)
        cold_cores_catalog.query("GLAT < %s"%y_up, inplace=True)
        cold_cores_catalog.query("GLON > %s"%x_left, inplace=True)
        cold_cores_catalog.query("GLON < %s"%x_right, inplace=True)

        # false_coords = SkyCoord(false_catalog['GLON'].values, false_catalog['GLAT'].values, unit='deg', frame='galactic')
        cold_cores_coords = SkyCoord(cold_cores_catalog['GLON'].values, cold_cores_catalog['GLAT'].values, unit='deg', frame='galactic')
        # false_coords_list = [cold_cores_catalog]


        coords_ns = SkyCoord(coord_catalog['GLON'].values, coord_catalog['GLAT'].values, unit='deg', frame='galactic')

        cluster_density = []
        false_cluster_density = []
        coord_neighbours = []
        false_coord_neighbours = []
        for i in range(input_size):
            ## Match between test patch center and galaxy clusters
            idx, _, _ = match_coordinates_sky(test_coords, coords_ns, nthneighbor=1)
            l_diff = np.abs(test_catalog['GLON'].values[i] - coord_catalog['GLON'].values[idx[i]])
            if np.abs(np.abs(test_catalog['GLON'].values[i] - coord_catalog['GLON'].values[idx[i]]) - 360) < l_diff:
                l_diff = np.abs(np.abs(test_catalog['GLON'].values[i] - coord_catalog['GLON'].values[idx[i]]) - 360)
            b_diff = np.abs(test_catalog['GLAT'].values[i] - coord_catalog['GLAT'].values[idx[i]])            
            if np.abs(np.abs(test_catalog['GLAT'].values[i] - coord_catalog['GLAT'].values[idx[i]]) - 180) < b_diff:
                b_diff = np.abs(np.abs(test_catalog['GLAT'].values[i] - coord_catalog['GLAT'].values[idx[i]]) - 180)
            k = 2
            neighb = [[coord_catalog['GLON'].values[idx[i]], coord_catalog['GLAT'].values[idx[i]]]]
            while l_diff <= 0.5*self.ndeg and b_diff <= 0.5*self.ndeg:
                idx, _, _ = match_coordinates_sky(test_coords, coords_ns, nthneighbor=k)
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

            ## Match between test patch center and false clusters
            idx, _, _ = match_coordinates_sky(test_coords, cold_cores_coords, nthneighbor=1)
            l_diff = np.abs(test_catalog['GLON'].values[i] - cold_cores_catalog['GLON'].values[idx[i]])
            if np.abs(np.abs(test_catalog['GLON'].values[i] - cold_cores_catalog['GLON'].values[idx[i]]) - 360) < l_diff:
                l_diff = np.abs(np.abs(test_catalog['GLON'].values[i] - cold_cores_catalog['GLON'].values[idx[i]]) - 360)
            b_diff = np.abs(test_catalog['GLAT'].values[i] - cold_cores_catalog['GLAT'].values[idx[i]])    
            if np.abs(np.abs(test_catalog['GLAT'].values[i] - cold_cores_catalog['GLAT'].values[idx[i]]) - 180) < b_diff:
                b_diff = np.abs(np.abs(test_catalog['GLAT'].values[i] - cold_cores_catalog['GLAT'].values[idx[i]]) - 180)        
            k = 2
            neighb = [[cold_cores_catalog['GLON'].values[idx[i]], cold_cores_catalog['GLAT'].values[idx[i]]]]
            while l_diff <= 0.5*self.ndeg and b_diff <= 0.5*self.ndeg:
                idx, _, _ = match_coordinates_sky(test_coords, cold_cores_coords, nthneighbor=k)
                l_diff = np.abs(test_catalog['GLON'].values[i] - cold_cores_catalog['GLON'].values[idx[i]])
                if np.abs(np.abs(test_catalog['GLON'].values[i] - cold_cores_catalog['GLON'].values[idx[i]]) - 360) < l_diff:
                    l_diff = np.abs(np.abs(test_catalog['GLON'].values[i] - cold_cores_catalog['GLON'].values[idx[i]]) - 360)
                b_diff = np.abs(test_catalog['GLAT'].values[i] - cold_cores_catalog['GLAT'].values[idx[i]])
                if np.abs(np.abs(test_catalog['GLAT'].values[i] - cold_cores_catalog['GLAT'].values[idx[i]]) - 180) < b_diff:
                    b_diff = np.abs(np.abs(test_catalog['GLAT'].values[i] - cold_cores_catalog['GLAT'].values[idx[i]]) - 180)  
                neighb.append([cold_cores_catalog['GLON'].values[idx[i]], cold_cores_catalog['GLAT'].values[idx[i]]])
                k += 1
            false_coord_neighbours.append(neighb)
            false_cluster_density.append(k-2)


        sum_cluster, sum_false = 0, 0
        ra_list, dec_list = [], []
        for i, coord in enumerate(test_coords):
            patch = cutsky.cut_fits(coord)
            HDU = patch[-1]['fits']
            wcs = WCS(HDU.header)


            center = []
            ang_center = []
            if cluster_density[i] == 0:
                mask = np.zeros((self.npix, self.npix))
                labels[i,:,:,0] = mask
            else:
                for j in range(cluster_density[i]):
                    center.append(wcs.world_to_pixel_values(coord_neighbours[i][j][0], coord_neighbours[i][j][1]))
                    ang_center.append((coord_neighbours[i][j][0], coord_neighbours[i][j][1]))
                mask, _, ra, dec = self.create_circular_mask(self.npix, self.npix, center=center, ang_center= ang_center, radius=self.disk_radius)
                ra_list = np.concatenate((ra_list, ra))
                dec_list = np.concatenate((dec_list, dec))
                labels[i,:,:,0] = mask.astype(int)

            center = []
            ang_center = []
            if false_cluster_density[i] == 0:
                mask_false = np.zeros((self.npix, self.npix))
                labels[i,:,:,1] = mask_false
            else:
                for j in range(false_cluster_density[i]):
                    center.append(wcs.world_to_pixel_values(false_coord_neighbours[i][j][0], false_coord_neighbours[i][j][1]))
                    ang_center.append((false_coord_neighbours[i][j][0], false_coord_neighbours[i][j][1]))
                mask_false, _, ra, dec = self.create_circular_mask(self.npix, self.npix, center=center, ang_center= ang_center, radius=self.disk_radius)
                ra_list = np.concatenate((ra_list, ra))
                dec_list = np.concatenate((dec_list, dec))
                labels[i,:,:,1] = mask_false.astype(int)

            milca[i,:,:,0] = patch[-1]['fits'].data
            if not label_only:
                for j in range(len(self.bands)):
                    inputs[i,:,:,j] = patch[j]['fits'].data

            if verbose:
                print('\n')
                print(i)
                print('cluster density: %s'%cluster_density[i])
                print('false cluster density: %s'%false_cluster_density[i])
                print('test centers: {:.2f}, {:.2f}'.format(test_catalog['GLON'].values[i], test_catalog['GLAT'].values[i]))
                print(false_coord_neighbours[i])
                print(coord_neighbours[i])
                print('\n')

            if plot == True:
                fig = plt.figure(figsize=(12,5), tight_layout=False)

                ax = fig.add_subplot(131)
                im = ax.imshow(HDU.data, origin='lower')
                ax.set_title('%s: '%i + 'l={:.2f}, b={:.2f}'.format(test_catalog['GLON'].values[i], test_catalog['GLAT'].values[i]))

                ax = fig.add_subplot(132)
                im = ax.imshow(mask, origin='lower')
                ax.set_title('potential clusters: %s'%cluster_density[i])

                ax = fig.add_subplot(133)
                im = ax.imshow(mask_false, origin='lower')
                ax.set_title('potential sources: %s'%false_cluster_density[i])
                
                GenerateFiles.make_directory(self, path_to_file = self.temp_path + 'test_set_r%s_f%s_d%s_s%s_c%s/'%(region, self.freq, self.disk_radius, self.npix, int(cold_cores)))
                plt.savefig(self.temp_path + 'test_set_r%s_f%s_d%s_s%s/'%(region, self.freq, self.disk_radius, self.npix) + 'test_%s'%i + '.png', bbox_inches='tight', transparent=False)
                plt.show()
                plt.close()

            sum_cluster += cluster_density[i]
            sum_false += false_cluster_density[i]

        # cluster_coords = np.array([ra_list, dec_list])
        print(sum_cluster, sum_false)

        # np.savez_compressed(self.dataset_path + 'test_cluster_coordinates_f%s_'%self.freq + self.dataset, cluster_coords)
        if not label_only:
            np.savez_compressed(self.output_path + 'files/r%s_f%s_d%s_s%s_c%s/'%(region, self.freq, self.disk_radius, self.npix, int(cold_cores)) + 'input_test_f%s_'%(self.freq) + self.dataset, inputs)
        np.savez_compressed(self.output_path + 'files/r%s_f%s_d%s_s%s_c%s/'%(region, self.freq, self.disk_radius, self.npix, int(cold_cores)) + 'milca_test_f%s_'%(self.freq) + self.dataset, milca)
        np.savez_compressed(self.output_path + 'files/r%s_f%s_d%s_s%s_c%s/'%(region, self.freq, self.disk_radius, self.npix, int(cold_cores)) + 'label_test_f%s_'%(self.freq) + self.dataset, labels)

    def test_data_generator(self, cold_cores=False, label_only=False, n_jobs=1, plot=False, verbose=False):

        Parallel(n_jobs=n_jobs)(delayed(self.make_test_input)(region, x_left, x_right, y_up, y_down, cold_cores=cold_cores, label_only=label_only, plot=plot, verbose=verbose) for region,(x_left, x_right, y_up, y_down) in tqdm(enumerate(self.test_regions)))

        # for region,(x_left, x_right, y_up, y_down) in tqdm(enumerate(self.test_regions)):
        #     self.make_test_input(region, x_left, x_right, y_up, y_down, label_only, plot, verbose)
            
        
            



        

    def train_data_generator(self, loops, cold_cores=False, label_only=False, n_jobs = 1, plot=False):
        """Calls make_input n=loops times to create input/output datasets for all clusters in the selected cluster
        catalog. Patches contain at least one cluster and for each loop, patches undergo random translations. 
        Function saves the following datasets as .npz files:
            - dataset_type: list of str containing either 'train', 'val' or 'test'. 
            - inputs: np.ndarray with shape (loops*len(self.dataset), self.npix, self.npix, len(self.bands)) containing input patches
            - labels: np.ndarray with shape (loops*len(self.dataset), self.npix, self.npix, 1) containing segmentation masks
            - milca:  np.ndarray with shape (loops*len(self.dataset), self.npix, self.npix, 1) containing milca patches.

        Args:
            loops (int): number of times the dataset containing patches with at least one cluster within will be 
                         added again to training set with random variations (translations)
            n_jobs (int, optional): Core numbers that will be used. Core numbers cannot exceed loops. Defaults to 1.
            plot (bool, optional): If True, will plot the number of potential objects per patch. Defaults to True.
        """

        for region in range(0,len(self.test_regions)):
            all_files = glob.glob(os.path.join(self.output_path + 'files/r%s_f%s_d%s_s%s_c%s/*.npz'%(region, self.freq, self.disk_radius, self.npix, int(cold_cores))))
            for f in all_files:
                os.remove(f)

        Parallel(n_jobs=n_jobs)(delayed(self.make_input)(p, cold_cores=cold_cores, label_only=label_only, plot=plot) for p in tqdm(range(loops)))

        for region in range(0,len(self.test_regions)):
            ## dataset type to know if data is in training, validation or test set
            all_type = glob.glob(os.path.join(self.output_path + 'files/r%s_f%s_d%s_s%s_c%s/'%(region, self.freq, self.disk_radius, self.npix, int(cold_cores)), "type_n*.npz"))
            X = []
            for f in all_type:
                X.append(np.load(f)['arr_0'])
            dataset_type = np.concatenate(X, axis=0)
            np.savez_compressed(self.dataset_path + 'type_r%s_f%s_d%s_s%s_c%s_'%(region, self.freq, self.disk_radius, self.npix, int(cold_cores)) + self.dataset, dataset_type)

            ## dataset type bar plot
            # if plot == True:
            #     counts = Counter(dataset_type)
            #     df = pd.DataFrame.from_dict(counts, orient='index')
            #     ax = df.plot(kind='bar')
            #     ax.figure.savefig(self.output_path + 'figures/' + 'dataset_type_density' + '.png', bbox_inches='tight', transparent=False)

            ## input file
            if not label_only:
                all_type = glob.glob(os.path.join(self.output_path + 'files/r%s_f%s_d%s_s%s_c%s/'%(region, self.freq, self.disk_radius, self.npix, int(cold_cores)), "input_n*.npz"))
                X = []
                for f in all_type:
                    X.append(np.load(f)['arr_0'])
                inputs = np.concatenate(X, axis=0)
                np.savez_compressed(self.dataset_path + 'input_r%s_f%s_d%s_s%s_c%s_'%(region, self.freq, self.disk_radius, self.npix, int(cold_cores)) + self.dataset, inputs)

            ## label file
            all_type = glob.glob(os.path.join(self.output_path + 'files/r%s_f%s_d%s_s%s_c%s/'%(region, self.freq, self.disk_radius, self.npix, int(cold_cores)), "label_n*.npz"))
            X = []
            for f in all_type:
                X.append(np.load(f)['arr_0'])
            labels = np.concatenate(X, axis=0)
            np.savez_compressed(self.dataset_path + 'label_r%s_f%s_d%s_s%s_c%s_'%(region, self.freq, self.disk_radius, self.npix, int(cold_cores)) + self.dataset, labels)
            
            ## milca file
            all_type = glob.glob(os.path.join(self.output_path + 'files/r%s_f%s_d%s_s%s_c%s/'%(region, self.freq, self.disk_radius, self.npix, int(cold_cores)), "milca_n*.npz"))
            X = []
            for f in all_type:
                X.append(np.load(f)['arr_0'])
            milca = np.concatenate(X, axis=0)
            np.savez_compressed(self.dataset_path + 'milca_r%s_f%s_d%s_s%s_c%s_'%(region, self.freq, self.disk_radius, self.npix, int(cold_cores)) + self.dataset, milca)

    def fit_gaussian_up_to_mode(self, dataset, index, slice='train', plot=True):
        """Fits a gaussian function to a 1d-ndarray on data up to mode.

        Args:
            dataset (np.ndarray): 1D ndarray containing a 3D flattened ndarray with input data for one band.
            index ([type]): index in self.bands.
            slice (str, optional): Dataset type. Either 'train', 'val', or 'test'. Defaults to 'train'.
            plot (bool, optional): If True will plot dataset distribution with the gaussian fit on top. 
                                Defaults to True.

        Returns:
            float: sigma value in A*np.exp(-0.5*((x-mu)/sigma)**2) obtained with a leastsq fit.
        """

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

    def train_val_split(self, labels_train, dataset_type_train, label_only, inputs_train=None):
        """Splits input, output and milca datasets into training, validation and test.

        Args:
            inputs_train (np.ndarray): [description]
            labels_train (np.ndarray): [description]
            dataset_type_train (np.ndarray): [description]

        Returns:
            (np.ndarray): X_train; 
            (np.ndarray): X_val;
            (np.ndarray): output_train; 
            (np.ndarray): output_val;
        """

        type_count_train = Counter(dataset_type_train)
        print("[Inputs] training: {:.0f}, validation: {:.0f}, test: {:.0f}".format(type_count_train['train'], type_count_train['val'], int(type_count_train['test']/self.loops)))

        if not label_only:
            X_train = np.ndarray((type_count_train['train'], self.npix, self.npix, len(self.bands)))
            X_val = np.ndarray((type_count_train['val'], self.npix, self.npix, len(self.bands)))
        output_train = np.ndarray((type_count_train['train'], self.npix, self.npix, self.n_labels))
        output_val = np.ndarray((type_count_train['val'], self.npix, self.npix, self.n_labels))

        index_train, index_val = 0, 0
        for i in range(len(labels_train)):
            if dataset_type_train[i] == 'train':
                if not label_only:
                    X_train[index_train,:,:,:] = inputs_train[i,:,:,:]
                output_train[index_train,:,:,:] = labels_train[i,:,:,:]
                index_train += 1
            if dataset_type_train[i] == 'val':
                if not label_only:
                    X_val[index_val,:,:,:] = inputs_train[i,:,:,:]
                output_val[index_val,:,:,:] = labels_train[i,:,:,:]
                index_val += 1
        if label_only:
            return output_train, output_val
        if not label_only:
            return X_train, X_val, output_train, output_val

    def preprocess_region(self, region, cold_cores=False, label_only=False, leastsq=False, range_comp=True, plot=True):
        if not label_only:
            inputs_train = np.load(self.dataset_path + 'input_r%s_f%s_d%s_s%s_c%s_'%(region, self.freq, self.disk_radius, self.npix, int(cold_cores)) + self.dataset + '.npz')['arr_0']
        labels_train = np.load(self.dataset_path + 'label_r%s_f%s_d%s_s%s_c%s_'%(region, self.freq, self.disk_radius, self.npix, int(cold_cores)) + self.dataset + '.npz')['arr_0']
        dataset_type_train = np.load(self.dataset_path + 'type_r%s_f%s_d%s_s%s_c%s_'%(region, self.freq, self.disk_radius, self.npix, int(cold_cores)) + self.dataset + '.npz')['arr_0']

        # inputs_test = np.load(self.output_path + 'files/f%s_d%s/'%(self.freq, self.disk_radius) + 'input_n0_f%s_'%self.freq + self.dataset + '.npz')['arr_0']
        # labels_test = np.load(self.output_path + 'files/f%s_d%s/'%(self.freq, self.disk_radius) + 'label_n0_f%s_'%self.freq + self.dataset + '.npz')['arr_0']
        # dataset_type_test = np.load(self.output_path + 'files/f%s_d%s/'%(self.freq, self.disk_radius) + 'type_n0_f%s_'%self.freq + self.dataset + '.npz')['arr_0']
        # milca_test = np.load(self.output_path + 'files/f%s_d%s/'%(self.freq, self.disk_radius) + 'milca_n0_f%s_'%self.freq + self.dataset + '.npz')['arr_0']
        M_test = np.load(self.output_path + 'files/r%s_f%s_d%s_s%s_c%s/'%(region, self.freq, self.disk_radius, self.npix, int(cold_cores)) + 'milca_test_f%s_'%(self.freq) + self.dataset + '.npz')['arr_0']
        if not label_only:
            X_test = np.load(self.output_path + 'files/r%s_f%s_d%s_s%s_c%s/'%(region, self.freq, self.disk_radius, self.npix, int(cold_cores)) + 'input_test_f%s_'%(self.freq) + self.dataset + '.npz')['arr_0']
        output_test = np.load(self.output_path + 'files/r%s_f%s_d%s_s%s_c%s/'%(region, self.freq, self.disk_radius, self.npix, int(cold_cores)) + 'label_test_f%s_'%(self.freq) + self.dataset + '.npz')['arr_0']

        if label_only:
            output_train, output_val = self.train_val_split(labels_train, dataset_type_train, label_only)
        if not label_only:
            X_train, X_val, output_train, output_val = self.train_val_split(labels_train, dataset_type_train, label_only, inputs_train=inputs_train)
        

        np.savez_compressed(self.dataset_path + 'label_train_pre_r%s_f%s_d%s_s%s_c%s_'%(region, self.freq, self.disk_radius, self.npix, int(cold_cores)) + self.dataset, output_train)

        np.savez_compressed(self.dataset_path + 'label_val_pre_r%s_f%s_d%s_s%s_c%s_'%(region, self.freq, self.disk_radius, self.npix, int(cold_cores)) + self.dataset, output_val)

        np.savez_compressed(self.dataset_path + 'label_test_pre_r%s_f%s_d%s_s%s_c%s_'%(region, self.freq, self.disk_radius, self.npix, int(cold_cores)) + self.dataset, output_test)

        np.savez_compressed(self.dataset_path + 'milca_test_pre_r%s_f%s_s%s_c%s_'%(region, self.freq, self.npix, int(cold_cores)) + self.dataset, M_test)

        if not label_only:

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
                np.savez_compressed(self.dataset_path + 'input_train_pre_r%s_f%s_s%s_c%s_'%(region, self.freq, self.npix, int(cold_cores)) + self.dataset, input_train)

                input_val = np.ndarray((len(X_val),self.npix,self.npix,len(self.bands)))
                for i in range(len(X_val)):
                    for j in range(len(self.bands)):
                        input_val[i,:,:,j] = np.arcsinh(X_val[i,:,:,j]/ scaling_test[j])
                np.savez_compressed(self.dataset_path + 'input_val_pre_r%s_f%s_s%s_c%s_'%(region, self.freq, self.npix, int(cold_cores)) + self.dataset, input_val)

                input_test = np.ndarray((len(X_test),self.npix,self.npix,len(self.bands)))
                for i in range(len(X_test)):
                    for j in range(len(self.bands)):
                        input_test[i,:,:,j] = np.arcsinh(X_test[i,:,:,j]/ scaling_test[j])
                np.savez_compressed(self.dataset_path + 'input_test_pre_r%s_f%s_s%s_c%s_'%(region, self.freq, self.npix, int(cold_cores)) + self.dataset, input_test)

            else:
                np.savez_compressed(self.dataset_path + 'input_train_pre_r%s_f%s_r0_s%s_c%s_'%(region, self.freq, self.npix, int(cold_cores)) + self.dataset, X_train)
                np.savez_compressed(self.dataset_path + 'input_val_pre_r%s_f%s_r0_s%s_c%s_'%(region, self.freq, self.npix, int(cold_cores)) + self.dataset, X_val)
                np.savez_compressed(self.dataset_path + 'input_test_pre_r%s_f%s_r0_s%s_c%s_'%(region, self.freq, self.npix, int(cold_cores)) + self.dataset, X_test)


            if plot == True:
                density = True
                GenerateFiles.make_directory(self, path_to_file = self.output_path + 'figures/prepro_r%s_f%s_d%s_s%s_c%s/'%(region, self.freq, self.disk_radius, self.npix, int(cold_cores)))

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
                plt.savefig(self.output_path + 'figures/prepro_r%s_f%s_d%s_s%s_c%s/'%(region, self.freq, self.disk_radius, self.npix, int(cold_cores)) + 'preprocessing_normalization_train'  + '.png', bbox_inches='tight', transparent=False)
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
                plt.savefig(self.output_path + 'figures/prepro_r%s_f%s_d%s_s%s_c%s/'%(region, self.freq, self.disk_radius, self.npix, int(cold_cores)) + 'preprocessing_range_compression_train'  + '.png', bbox_inches='tight', transparent=False)
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
                plt.savefig(self.output_path + 'figures/prepro_r%s_f%s_d%s_s%s_c%s/'%(region, self.freq, self.disk_radius, self.npix, int(cold_cores)) + 'preprocessing_normalization_test'  + '.png', bbox_inches='tight', transparent=False)
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
                plt.savefig(self.output_path + 'figures/prepro_r%s_f%s_d%s_s%s_c%s/'%(region, self.freq, self.disk_radius, self.npix, int(cold_cores)) + 'preprocessing_range_compression_test'  + '.png', bbox_inches='tight', transparent=False)
                plt.show()
                plt.close()

        all_files = glob.glob(os.path.join(self.output_path + "files/r%s_f%s_d%s_s%s_c%s/"%(region, self.freq, self.disk_radius, self.npix, int(cold_cores)) + "*.npz"))
        for f in all_files:
            os.remove(f)
        os.rmdir(self.output_path + "files/r%s_f%s_d%s_s%s_c%s/"%(region, self.freq, self.disk_radius, self.npix, int(cold_cores)))
        if not label_only:
            os.remove(self.dataset_path + 'input_r%s_f%s_d%s_s%s_c%s_'%(region, self.freq, self.disk_radius, self.npix, int(cold_cores)) + self.dataset + '.npz')
        os.remove(self.dataset_path + 'label_r%s_f%s_d%s_s%s_c%s_'%(region, self.freq, self.disk_radius, self.npix, int(cold_cores)) + self.dataset + '.npz')
        os.remove(self.dataset_path + 'type_r%s_f%s_d%s_s%s_c%s_'%(region, self.freq, self.disk_radius, self.npix, int(cold_cores)) + self.dataset + '.npz')
        os.remove(self.dataset_path + 'milca_r%s_f%s_d%s_s%s_c%s_'%(region, self.freq, self.disk_radius, self.npix, int(cold_cores)) + self.dataset + '.npz')
        


    def preprocess(self, cold_cores=False, label_only=False, leastsq=False, range_comp=True, n_jobs=1, plot=True):
        """Takes input, output and milca data from train_data_generator and splits it into training, validation and test set.
        Data is then individually standardized and compressed for each dataset.

        Args:
            leastsq (bool, optional): If True, sigma will be computed by fitting a gaussian up to mode. If False, sigma will be 
                                    computed using MAD. Defaults to False.
            range_comp (bool, optional): If True, range compression will be applied. Defaults to True.
            plot (bool, optional): If True, will plot distributions for data after standadization and range compression. Defaults to True.
        """

        Parallel(n_jobs=n_jobs)(delayed(self.preprocess_region)(region, cold_cores=cold_cores, label_only=label_only, leastsq=leastsq, range_comp=range_comp, plot=plot) for region in tqdm(range(0,len(self.test_regions))))
            
        print('[preprocessing] Done!')

