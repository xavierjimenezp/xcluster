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

    def __init__(self, dataset, bands, planck_path, milca_path, disk_radius=None, output_path=None):
        """
        Args:
            dataset (str): file name for the cluster catalog that will used.
                        Options are 'planck_z', 'planck_z_no-z', 'MCXC', 'RM30', 'RM50'.
            bands (list): list of full sky-maps that will be used for the input file.
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
        self.bands = bands # '100GHz','143GHz','217GHz','353GHz','545GHz','857GHz'

        maps = []
        self.freq = 0
        self.planck_freq = 0
        if '100GHz' in  bands:
            maps.append((planck_path + "HFI_SkyMap_100-field-IQU_2048_R3.00_full.fits", {'legend': 'HFI 100', 'docontour': True}))
            self.freq += 2
            self.planck_freq += 2
        if '143GHz' in bands:
            maps.append((planck_path + "HFI_SkyMap_143-field-IQU_2048_R3.00_full.fits", {'legend': 'HFI 143', 'docontour': True}))
            self.freq += 4
            self.planck_freq += 4
        if '217GHz' in bands:
            maps.append((planck_path + "HFI_SkyMap_217-field-IQU_2048_R3.00_full.fits", {'legend': 'HFI 217', 'docontour': True}))
            self.freq += 8
            self.planck_freq += 8
        if '353GHz' in bands:
            maps.append((planck_path + "HFI_SkyMap_353-psb-field-IQU_2048_R3.00_full.fits", {'legend': 'HFI 353', 'docontour': True}))
            self.freq += 16
            self.planck_freq += 16
        if '545GHz' in bands:
            maps.append((planck_path + "HFI_SkyMap_545-field-Int_2048_R3.00_full.fits", {'legend': 'HFI 545', 'docontour': True}))
            self.freq += 32
            self.planck_freq += 32
        if '857GHz' in bands:
            maps.append((planck_path + "HFI_SkyMap_857-field-Int_2048_R3.00_full.fits", {'legend': 'HFI 857', 'docontour': True}))
            self.freq += 64
            self.planck_freq += 64
        if 'y-map' in bands:
            maps.append((milca_path + "milca_ymaps.fits", {'legend': 'MILCA y-map', 'docontour': True}))
            self.freq += 128
        if 'CO' in bands:
            maps.append((planck_path + "COM_CompMap_CO21-commander_2048_R2.00.fits", {'legend': 'CO', 'docontour': True}))
            self.freq += 254
        maps.append((milca_path + "milca_ymaps.fits", {'legend': 'MILCA y-map', 'docontour': True})) #used for plots only
        
        self.maps = maps

        self.temp_path = self.path + 'to_clean/'
        if disk_radius is not None:
            self.disk_radius = disk_radius
        else:
            self.disk_radius = 'istri'
        self.npix = 64 #in pixels
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
        df_MCXC = pd.DataFrame(data={'RA': MCXC[1].data['RA'].tolist(), 'DEC': MCXC[1].data['DEC'].tolist(), 'R500': MCXC[1].data['RADIUS_500'].tolist(), 'M500': MCXC[1].data['MASS_500'].tolist(),
            'Z': MCXC[1].data['REDSHIFT'].tolist()})

        REDMAPPER = fits.open(self.planck_path + 'redmapper_dr8_public_v6.3_catalog.fits')
        df_REDMAPPER = pd.DataFrame(data={'RA': REDMAPPER[1].data['RA'].tolist(), 'DEC': REDMAPPER[1].data['DEC'].tolist(), 'LAMBDA': REDMAPPER[1].data['LAMBDA'].tolist(),
        'Z': REDMAPPER[1].data['Z_SPEC'].tolist()})

        df_REDMAPPER_30 = df_REDMAPPER.query("LAMBDA > 30")
        df_REDMAPPER_50 = df_REDMAPPER.query("LAMBDA > 50")

        self.remove_duplicates_on_radec(df_MCXC, df_psz2, output_name='MCXC_no_planck', plot=plot)
        self.remove_duplicates_on_radec(df_REDMAPPER_30, df_psz2, output_name='RM30_no_planck', plot=plot)
        self.remove_duplicates_on_radec(df_REDMAPPER_50, df_psz2, output_name='RM50_no_planck', plot=plot)


    def create_fake_source_catalog(self):
        ACT = fits.open(self.planck_path + 'sptecs_catalog_oct919_forSZDB.fits')
        SPT = fits.open(self.planck_path + 'DR5_cluster-catalog_v1.1_forSZDB.fits')
        PGCC = fits.open(self.planck_path + 'HFI_PCCS_GCC_R2.02.fits')

        df_act = pd.DataFrame(data={'RA': list(ACT[1].data['RA']), 'DEC': list(ACT[1].data['DEC']), 'GLON': list(ACT[1].data['GLON']), 'GLAT': list(ACT[1].data['GLAT'])})
        df_spt = pd.DataFrame(data={'RA': list(SPT[1].data['RA']), 'DEC': list(SPT[1].data['DEC']), 'GLON': list(SPT[1].data['GLON']), 'GLAT': list(SPT[1].data['GLAT'])})
        df_pgcc = pd.DataFrame(data={'RA': list(PGCC[1].data['RA']), 'DEC': list(PGCC[1].data['DEC']), 'GLON': list(PGCC[1].data['GLON']), 'GLAT': list(PGCC[1].data['GLAT'])})

        ACT.close()
        SPT.close()
        PGCC.close()

        df_act.to_csv(self.path + 'catalogs/' + 'ACT' + '.csv', index=False)
        df_spt.to_csv(self.path + 'catalogs/' + 'SPT' + '.csv', index=False)
        df_pgcc.to_csv(self.path + 'catalogs/' + 'PGCC' + '.csv', index=False)

        df = pd.DataFrame(columns=['RA','DEC','GLON','GLAT'])

        bands = ['100GHz', '143GHz', '217GHz', '353GHz', '545GHz', '857GHz']
        cs_100 = fits.open(self.planck_path + 'COM_PCCS_100_R2.01.fits')
        cs_143 = fits.open(self.planck_path + 'COM_PCCS_143_R2.01.fits')
        cs_217 = fits.open(self.planck_path + 'COM_PCCS_217_R2.01.fits')
        cs_353 = fits.open(self.planck_path + 'COM_PCCS_353_R2.01.fits')
        cs_545 = fits.open(self.planck_path + 'COM_PCCS_545_R2.01.fits')
        cs_857 = fits.open(self.planck_path + 'COM_PCCS_857_R2.01.fits')


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
        
        df_all = pd.concat((df_act, df_spt, df_pgcc, df))
        df.to_csv(self.path + 'catalogs/' + 'CS_f%s'%freq + '.csv', index=False)
        df_all.to_csv(self.path + 'catalogs/' + 'False_SZ_catalog_f%s'%freq + '.csv', index=False)

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
                
                df_all = pd.concat((df_act, df_spt, df_pgcc, df))
                df.to_csv(self.path + 'catalogs/' + 'CS_f%s'%freq + '.csv', index=False)
                df_all.to_csv(self.path + 'catalogs/' + 'False_SZ_catalog_f%s'%freq + '.csv', index=False)
    
        cs_100.close()
        cs_143.close()
        cs_217.close()
        cs_353.close()
        cs_545.close()
        cs_857.close()


    def remove_duplicates_on_radec(self, df_main, df_with_dup, output_name, plot=False):
        """Takes two different dataframes with columns 'RA' & 'DEC' and performs a spatial
        coordinate match with a 7arcmin tolerance. Saves a .csv file containing df_main 
        without objects in common from df_with_dup.

        Args:
            df_main (pd.DataFrame): main dataframe.
            df_with_dup (pd.DataFrame): dataframe that contains objects from df_main.
            output_name (str): name that will be used in the plot file name.
            plot (bool, optional): If True, will save duplicates distance from each other distribution plots. Defaults to False.
        """
        ID = np.arange(0, len(df_with_dup))
        df_with_dup = df_with_dup[['RA', 'DEC']].copy()
        df_with_dup.insert(loc=0, value=ID, column='ID')

        scatalog_sub = SkyCoord(ra=df_main['RA'].values, dec=df_main['DEC'].values, unit='deg')
        pcatalog_sub = SkyCoord(ra=df_with_dup['RA'].values, dec=df_with_dup['DEC'].values, unit='deg')
        idx, d2d, _ = match_coordinates_sky(scatalog_sub, pcatalog_sub, nthneighbor=1)

        tol = 5
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
            dist_from_center = np.sqrt((X - c[0])**2 + (Y-c[1])**2)
            mask += (dist_from_center <= radius).astype(int)
            is_all_zero = np.all(((dist_from_center <= radius).astype(int) == 0))
            if is_all_zero == False:
                count += 1
                ra.append(ang_center[i][0])
                dec.append(ang_center[i][1])
        return np.where(mask > 1, 1, mask), count, ra, dec

    def return_coord_catalog(self):
        if self.dataset == 'planck_z':
            planck_z = pd.read_csv(self.path + 'catalogs/planck_z' + '.csv')
            coord_catalog = planck_z[['RA', 'DEC', 'GLON', 'GLAT']].copy()
        elif self.dataset == 'planck_no-z':
            planck_z = pd.read_csv(self.path + 'catalogs/planck_z' + '.csv')
            planck_no_z = pd.read_csv(self.path + 'catalogs/planck_no-z' + '.csv')
            coord_catalog = pd.concat([planck_z[['RA', 'DEC', 'GLON', 'GLAT']].copy(), planck_no_z[['RA', 'DEC']].copy()], ignore_index=True)
        elif self.dataset == 'MCXC':
            planck_z = pd.read_csv(self.path + 'catalogs/planck_z' + '.csv')
            planck_no_z = pd.read_csv(self.path + 'catalogs/planck_no-z' + '.csv')
            MCXC = pd.read_csv(self.path + 'catalogs/MCXC_no_planck' + '.csv')
            coord_catalog = pd.concat([planck_z[['RA', 'DEC', 'GLON', 'GLAT']].copy(), planck_no_z[['RA', 'DEC']].copy(), MCXC[['RA', 'DEC']].copy()],
                ignore_index=True)
        elif self.dataset == 'RM30':
            planck_z = pd.read_csv(self.path + 'catalogs/planck_z' + '.csv')
            planck_no_z = pd.read_csv(self.path + 'catalogs/planck_no-z' + '.csv')
            MCXC = pd.read_csv(self.path + 'catalogs/MCXC_no_planck' + '.csv')
            RM30 = pd.read_csv(self.path + 'catalogs/RM30_no_planck' + '.csv')
            coord_catalog = pd.concat([planck_z[['RA', 'DEC', 'GLON', 'GLAT']].copy(), planck_no_z[['RA', 'DEC']].copy(), MCXC[['RA', 'DEC']].copy(),
                RM30[['RA', 'DEC']].copy()], ignore_index=True)
        elif self.dataset == 'RM50':
            planck_z = pd.read_csv(self.path + 'catalogs/planck_z' + '.csv')
            planck_no_z = pd.read_csv(self.path + 'catalogs/planck_no-z' + '.csv')
            MCXC = pd.read_csv(self.path + 'catalogs/MCXC_no_planck' + '.csv')
            RM50 = pd.read_csv(self.path + 'catalogs/RM50_no_planck' + '.csv')
            coord_catalog = pd.concat([planck_z[['RA', 'DEC', 'GLON', 'GLAT']].copy(), planck_no_z[['RA', 'DEC']].copy(), MCXC[['RA', 'DEC']].copy(),
                RM50[['RA', 'DEC']].copy()], ignore_index=True)

        false_catalog = pd.read_csv(self.path + 'catalogs/False_SZ_catalog_f%s.csv'%self.planck_freq)
        return coord_catalog, false_catalog

    def rotate(self, origin, point, angle):
        """
        Rotate a point clockwise by a given angle around a given origin.

        The angle should be given in radians.
        """
        angle = -np.radians(angle) #transform in radians, - sign is there to ensure clockwise

        ox, oy = origin
        px, py = point

        qx = ox + np.cos(angle) * (px - ox) - np.sin(angle) * (py - oy)
        qy = oy + np.sin(angle) * (px - ox) + np.cos(angle) * (py - oy)
        return qx, qy

    def rotate_patch(self, p, i, band, patch_rot, coord_catalog, random_angle, n_rot):

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

        return HDU_rot_data[n_rot:-n_rot, n_rot:-n_rot]


    def make_input(self, p, plot=False, verbose=False):
        """Creates input/output datasets for all clusters in the selected cluster catalog. Patches contain at least 
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

        coord_catalog, false_catalog = self.return_coord_catalog()

        #------------------------------------------------------------------#
        # # # # # Create ramdon coordinate translations # # # # #
        #------------------------------------------------------------------#

        input_size = len(coord_catalog['RA'].values)
        coords_ns = SkyCoord(ra=coord_catalog['RA'].values, dec=coord_catalog['DEC'].values, unit='deg')
        np.random.seed(p)
        random_coord_x = 2*np.random.rand(1, input_size).flatten() - np.ones_like(np.random.rand(1, input_size).flatten())
        np.random.seed(p+100)
        random_coord_y = 2*np.random.rand(1, input_size).flatten() - np.ones_like(np.random.rand(1, input_size).flatten())
        coords = SkyCoord(ra=coord_catalog['RA'].values + 0.5*(self.ndeg-0.2)*random_coord_x,
                          dec=coord_catalog['DEC'].values + 0.5*(self.ndeg-0.2)*random_coord_y, unit='deg') 

        #------------------------------------------------------------------#
        # # # # # Check for potential neighbours # # # # #
        #------------------------------------------------------------------#

        false_coords = SkyCoord(ra=false_catalog['RA'].values, dec=false_catalog['DEC'].values, unit='deg')
        cluster_density = []
        false_cluster_density = []
        coord_neighbours = []
        false_coord_neighbours = []
        for i in range(input_size):
            ## Match between galaxy clusters and themselves 
            idx, _, _ = match_coordinates_sky(coords_ns, coords_ns, nthneighbor=2)
            ra_diff = np.abs(coord_catalog['RA'].values[i] - coord_catalog['RA'].values[idx[i]])
            dec_diff = np.abs(coord_catalog['DEC'].values[i] - coord_catalog['DEC'].values[idx[i]])            
            k = 3
            neighb = [[coord_catalog['RA'].values[idx[i]], coord_catalog['DEC'].values[idx[i]]]]
            while ra_diff < 1.5*self.ndeg and dec_diff < 1.5*self.ndeg:
                idx, _, _ = match_coordinates_sky(coords_ns, coords_ns, nthneighbor=k)
                ra_diff = np.abs(coord_catalog['RA'].values[i] - coord_catalog['RA'].values[idx[i]])
                dec_diff = np.abs(coord_catalog['DEC'].values[i] - coord_catalog['DEC'].values[idx[i]])
                neighb.append([coord_catalog['RA'].values[idx[i]], coord_catalog['DEC'].values[idx[i]]])
                k += 1
            coord_neighbours.append(neighb)
            cluster_density.append(k-2)

            ## Match between galaxy clusters and false clusters
            idx, _, _ = match_coordinates_sky(coords_ns, false_coords, nthneighbor=1)
            ra_diff = np.abs(coord_catalog['RA'].values[i] - false_catalog['RA'].values[idx[i]])
            dec_diff = np.abs(coord_catalog['DEC'].values[i] - false_catalog['DEC'].values[idx[i]])            
            k = 2
            neighb = [[false_catalog['RA'].values[idx[i]], false_catalog['DEC'].values[idx[i]]]]
            while ra_diff < 1.5*self.ndeg and dec_diff < 1.5*self.ndeg:
                idx, _, _ = match_coordinates_sky(coords_ns, false_coords, nthneighbor=k)
                ra_diff = np.abs(coord_catalog['RA'].values[i] - false_catalog['RA'].values[idx[i]])
                dec_diff = np.abs(coord_catalog['DEC'].values[i] - false_catalog['DEC'].values[idx[i]])
                neighb.append([false_catalog['RA'].values[idx[i]], false_catalog['DEC'].values[idx[i]]])
                k += 1
            false_coord_neighbours.append(neighb)
            false_cluster_density.append(k-2)


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

            fig = plt.figure(figsize=(7,7), tight_layout=False)
            ax = fig.add_subplot(111)
            ax.set_facecolor('white')
            ax.grid(True, color='grey', lw=0.5)
            ax.set_xlabel('Neighbours per patch', fontsize=20)
            ax.set_ylabel('False cluster number', fontsize=20)
            ax.hist(false_cluster_density, bins=14)
            ax.set_yscale('log')

            plt.savefig(self.output_path + 'figures/' + 'false_cluster_density' + '.png', bbox_inches='tight', transparent=False)
            plt.show()
            plt.close()

        #------------------------------------------------------------------#
        # # # # # Create patch & masks # # # # #
        #------------------------------------------------------------------#

        maps = self.maps
        cutsky = CutSky(maps, npix=self.npix, pixsize=self.pixsize, low_mem=False)
        cutsky_rot = CutSky(maps, npix=int(np.sqrt(2)*self.npix), pixsize=self.pixsize, low_mem=False)

        labels = np.ndarray((input_size,self.npix,self.npix,2))
        inputs = np.ndarray((input_size,self.npix,self.npix,len(self.bands)))
        milca = np.ndarray((input_size,self.npix,self.npix,1))
        dataset_type = []
        
        for i, coord in enumerate(coords):
            # if hp.ang2pix(self.nside, coord.galactic.l.degree, coord.galactic.b.degree, lonlat=True) == 6 or hp.ang2pix(self.nside, coord.galactic.l.degree, coord.galactic.b.degree, lonlat=True) == 7:
            if coord.galactic.b.degree > -40 and coord.galactic.b.degree < -20 and coord.galactic.l.degree > 200 and coord.galactic.l.degree < 300:
                dataset_type.append('test')
                labels[i,:,:,0] = np.zeros((self.npix, self.npix))
                labels[i,:,:,1] = np.zeros((self.npix, self.npix))
                milca[i,:,:,0] = np.zeros((self.npix, self.npix))
                for j in range(len(self.bands)):
                    inputs[i,:,:,j] = np.zeros((self.npix, self.npix))
                continue
            # elif hp.ang2pix(self.nside, coord.galactic.l.degree, coord.galactic.b.degree, lonlat=True) == 9 or hp.ang2pix(self.nside, coord.galactic.l.degree, coord.galactic.b.degree, lonlat=True) == 38 or hp.ang2pix(self.nside, coord.galactic.l.degree, coord.galactic.b.degree, lonlat=True) == 41 or hp.ang2pix(self.nside, coord.galactic.l.degree, coord.galactic.b.degree, lonlat=True) == 25:
            elif coord.galactic.b.degree > -60 and coord.galactic.b.degree < -40:
                dataset_type.append('val')
            elif coord.galactic.b.degree > -62 and coord.galactic.b.degree < -60:
                dataset_type.append('skip')
                labels[i,:,:,0] = np.zeros((self.npix, self.npix))
                labels[i,:,:,1] = np.zeros((self.npix, self.npix))
                milca[i,:,:,0] = np.zeros((self.npix, self.npix))
                for j in range(len(self.bands)):
                    inputs[i,:,:,j] = np.zeros((self.npix, self.npix))
                continue
            else:
                dataset_type.append('train')

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
                center = [(x,y)]
                ang_center = [(coord_catalog['RA'].values[i], coord_catalog['DEC'].values[i])]
                if cluster_density[i] == 1:
                    mask, _, _, _ = self.create_circular_mask(h, w, center=center, ang_center= ang_center, radius=self.disk_radius)
                else:
                    for j in range(cluster_density[i]-1):
                        center.append(wcs.world_to_pixel_values(coord_neighbours[i][j][0], coord_neighbours[i][j][1]))
                        ang_center.append((coord_neighbours[i][j][0], coord_neighbours[i][j][1]))
                    mask, _, _, _ = self.create_circular_mask(h, w, center=center, ang_center= ang_center, radius=self.disk_radius)

            ## CLUSTERS
            center = [(x_rot,y_rot)]
            ang_center = [(coord_catalog['RA'].values[i], coord_catalog['DEC'].values[i])]
            if cluster_density[i] == 1:
                mask_rot, _, _, _ = self.create_circular_mask(h, w, center=center, ang_center= ang_center, radius=self.disk_radius)
                labels[i,:,:,0] = mask_rot.astype(int)
            else:
                for j in range(cluster_density[i]-1):
                    x_rotn, y_rotn = wcs_rot.world_to_pixel_values(coord_neighbours[i][j][0], coord_neighbours[i][j][1])
                    x_rotn, y_rotn = self.rotate(origin=(0.5*self.npix*np.sqrt(2), 0.5*self.npix*np.sqrt(2)), point=(x_rotn,y_rotn), angle=random_angle)
                    x_rotn, y_rotn = x_rotn-int(0.5*self.npix*(np.sqrt(2)-1)), y_rotn-int(0.5*self.npix*(np.sqrt(2)-1))
                    center.append((x_rotn, y_rotn))
                    ang_center.append((coord_neighbours[i][j][0], coord_neighbours[i][j][1]))
                mask_rot, _, _, _ = self.create_circular_mask(h, w, center=center, ang_center= ang_center, radius=self.disk_radius)
                labels[i,:,:,0] = mask_rot.astype(int)

            ## FALSE CLUSTERS
            center = []
            ang_center = []
            if false_cluster_density[i] == 0:
                mask_rot_false = np.zeros((self.npix, self.npix))
                labels[i,:,:,1] = mask_rot_false
            else:
                for j in range(false_cluster_density[i]):
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
            milca[i,:,:,0] = HDU_rot_data[n_rot:-n_rot, n_rot:-n_rot]
            for j in range(len(self.bands)):
                # inputs[i,:,:,j] = patch[j]['fits'].data
                inputs[i,:,:,j] = self.rotate_patch(p, i, j, patch_rot, coord_catalog, random_angle, n_rot)

            #------------------------------------------------------------------#
            # # # # # Plots # # # # #
            #------------------------------------------------------------------#

            if plot == True:
                fig = plt.figure(figsize=(25,5), tight_layout=False)

                ax = fig.add_subplot(151)
                im = ax.imshow(HDU.data, origin='lower')
                ax.scatter(x,y)
                ax.set_title('x={:.2f}, y={:.2f}'.format(x,y))

                ax = fig.add_subplot(152)
                im = ax.imshow(mask, origin='lower')
                ax.set_title('x={:.2f}, y={:.2f}'.format(x,y))

                ax = fig.add_subplot(153)
                im = ax.imshow(milca[i,:,:,0], origin='lower') #[n_rot:-n_rot, n_rot:-n_rot]
                ax.scatter(x_rot,y_rot)
                ax.set_title('xrot={:.2f}, yrot={:.2f}'.format(x_rot,y_rot))

                ax = fig.add_subplot(154)
                im = ax.imshow(labels[i,:,:,0], origin='lower')
                ax.set_title('Potential clusters: {:.0f}'.format(cluster_density[i]))

                ax = fig.add_subplot(155)
                im = ax.imshow(labels[i,:,:,1], origin='lower')
                ax.set_title('Potential sources: {:.0f}'.format(false_cluster_density[i]))

                GenerateFiles.make_directory(self, path_to_file = self.temp_path + 'training_set/')
                plt.savefig(self.temp_path + 'training_set/training_%s'%i + '.png', bbox_inches='tight', transparent=False)
                plt.show()
                plt.close()

        #------------------------------------------------------------------#
        # # # # # Save files # # # # #
        #------------------------------------------------------------------#

        assert len(coords) == len(dataset_type)

        if plot == True:
            counts = Counter(dataset_type)
            df = pd.DataFrame.from_dict(counts, orient='index')
            ax = df.plot(kind='bar')
            ax.figure.savefig(self.output_path + 'figures/' + 'dataset_type_density' + '.png', bbox_inches='tight', transparent=False)

        GenerateFiles.make_directory(self, path_to_file = self.output_path + 'files/' + 'f%s_d%s'%(self.freq, self.disk_radius))
        np.savez_compressed(self.output_path + 'files/f%s_d%s/'%(self.freq, self.disk_radius) + 'milca_n%s_f%s_'%(p, self.freq) + self.dataset, milca)
        np.savez_compressed(self.output_path + 'files/f%s_d%s/'%(self.freq, self.disk_radius) + 'type_n%s_f%s_'%(p, self.freq) + self.dataset, np.array(dataset_type))
        if p == 0:
            np.savez_compressed(self.dataset_path + 'type_test_f%s_'%(self.freq) + self.dataset, np.array(dataset_type))
        np.savez_compressed(self.output_path + 'files/f%s_d%s/'%(self.freq, self.disk_radius) + 'input_n%s_f%s_'%(p, self.freq) + self.dataset, inputs)
        np.savez_compressed(self.output_path + 'files/f%s_d%s/'%(self.freq, self.disk_radius) + 'label_n%s_f%s_'%(p, self.freq) + self.dataset, labels)


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

    def test_data_generator(self, plot=False, verbose=False):
        coord_catalog, false_catalog = self.return_coord_catalog()

        false_catalog.query("GLAT > -40", inplace=True)
        false_catalog.query("GLAT < -20", inplace=True)
        false_catalog.query("GLON > 200", inplace=True)
        false_catalog.query("GLON < 300", inplace=True)
        # print(len(false_catalog))
        # print(false_catalog.head(10))
        # return


        test_coords, test_catalog = self.test_coords(x_left=200, x_right=300, y_up=-18, y_down=-40)
        input_size = len(test_coords)
        maps = self.maps
        cutsky = CutSky(maps, npix=self.npix, pixsize=self.pixsize, low_mem=False)

        labels = np.ndarray((len(test_coords),self.npix,self.npix,2))
        inputs = np.ndarray((len(test_coords),self.npix,self.npix,len(self.bands)))
        milca = np.ndarray((len(test_coords),self.npix,self.npix,1))

        coords_ns = SkyCoord(coord_catalog['GLON'].values, coord_catalog['GLAT'].values, unit='deg', frame='galactic')
        false_coords = SkyCoord(false_catalog['GLON'].values, false_catalog['GLAT'].values, unit='deg', frame='galactic')
        cluster_density = []
        false_cluster_density = []
        coord_neighbours = []
        false_coord_neighbours = []
        for i in range(input_size):
            ## Match between test patch center and galaxy clusters
            idx, _, _ = match_coordinates_sky(test_coords, coords_ns, nthneighbor=1)
            l_diff = np.abs(test_catalog['GLON'].values[i] - coord_catalog['GLON'].values[idx[i]])
            b_diff = np.abs(test_catalog['GLAT'].values[i] - coord_catalog['GLAT'].values[idx[i]])            
            k = 2
            neighb = [[coord_catalog['GLON'].values[idx[i]], coord_catalog['GLAT'].values[idx[i]]]]
            while l_diff <= 0.5*self.ndeg and b_diff <= 0.5*self.ndeg:
                idx, _, _ = match_coordinates_sky(test_coords, coords_ns, nthneighbor=k)
                l_diff = np.abs(test_catalog['GLON'].values[i] - coord_catalog['GLON'].values[idx[i]])
                b_diff = np.abs(test_catalog['GLAT'].values[i] - coord_catalog['GLAT'].values[idx[i]])
                neighb.append([coord_catalog['GLON'].values[idx[i]], coord_catalog['GLAT'].values[idx[i]]])
                k += 1
            coord_neighbours.append(neighb)
            cluster_density.append(k-2)

            ## Match between test patch center and false clusters
            idx, _, _ = match_coordinates_sky(test_coords, false_coords, nthneighbor=1)
            l_diff = np.abs(test_catalog['GLON'].values[i] - false_catalog['GLON'].values[idx[i]])
            b_diff = np.abs(test_catalog['GLAT'].values[i] - false_catalog['GLAT'].values[idx[i]])            
            k = 2
            neighb = [[false_catalog['GLON'].values[idx[i]], false_catalog['GLAT'].values[idx[i]]]]
            while l_diff <= 0.5*self.ndeg and b_diff <= 0.5*self.ndeg:
                idx, _, _ = match_coordinates_sky(coords_ns, false_coords, nthneighbor=k)
                l_diff = np.abs(test_catalog['GLON'].values[i] - false_catalog['GLON'].values[idx[i]])
                b_diff = np.abs(test_catalog['GLAT'].values[i] - false_catalog['GLAT'].values[idx[i]])
                neighb.append([false_catalog['GLON'].values[idx[i]], false_catalog['GLAT'].values[idx[i]]])
                k += 1
            false_coord_neighbours.append(neighb)
            false_cluster_density.append(k-2)


        sum_cluster, sum_false = 0, 0
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
                mask, _, _, _ = self.create_circular_mask(self.npix, self.npix, center=center, ang_center= ang_center, radius=self.disk_radius)
                labels[i,:,:,0] = mask.astype(int)

            center = []
            ang_center = []
            if false_cluster_density[i] == 0:
                mask_false = np.zeros((self.npix, self.npix))
                labels[i,:,:,1] = mask
            else:
                for j in range(false_cluster_density[i]):
                    center.append(wcs.world_to_pixel_values(false_coord_neighbours[i][j][0], false_coord_neighbours[i][j][1]))
                    ang_center.append((false_coord_neighbours[i][j][0], false_coord_neighbours[i][j][1]))
                mask_false, _, _, _ = self.create_circular_mask(self.npix, self.npix, center=center, ang_center= ang_center, radius=self.disk_radius)
                labels[i,:,:,1] = mask_false.astype(int)

            milca[i,:,:,0] = patch[-1]['fits'].data
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
                
                GenerateFiles.make_directory(self, path_to_file = self.temp_path + 'test_set/')
                plt.savefig(self.temp_path + 'test_set/test_%s'%i + '.png', bbox_inches='tight', transparent=False)
                plt.show()
                plt.close()

            sum_cluster += cluster_density[i]
            sum_false += false_cluster_density[i]

        print(sum_cluster, sum_false)

        np.savez_compressed(self.output_path + 'files/f%s_d%s/'%(self.freq, self.disk_radius) + 'milca_test_f%s_'%(self.freq) + self.dataset, milca)
        np.savez_compressed(self.output_path + 'files/f%s_d%s/'%(self.freq, self.disk_radius) + 'input_test_f%s_'%(self.freq) + self.dataset, inputs)
        np.savez_compressed(self.output_path + 'files/f%s_d%s/'%(self.freq, self.disk_radius) + 'label_test_f%s_'%(self.freq) + self.dataset, labels)



          

    def create_input(self, p, plot=False, verbose=False):
        """Creates input/output datasets for all clusters in the selected cluster catalog. Patches contain at least 
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
        coords = SkyCoord(ra=coord_catalog['RA'].values -30*self.pixsize/60 + (60*self.pixsize/60)*random_coord_x,
                          dec=coord_catalog['DEC'].values -30*self.pixsize/60 + (60*self.pixsize/60)*random_coord_y, unit='deg')

        if p == 0:
            test_positions = np.array([coord_catalog['RA'].values -30*self.pixsize/60 + (60*self.pixsize/60)*random_coord_x, coord_catalog['DEC'].values -30*self.pixsize/60 + (60*1.7/60)*random_coord_y])
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
        
        for i, coord in enumerate(coords):
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
                    print('coords shift: {:.2f}, {:.2f}'.format(coord_catalog['RA'].values[i] -30*self.pixsize/60 + (60*self.pixsize/60)*random_coord_x[i], coord_catalog['DEC'].values[i] -30*self.pixsize/60 + (60*self.pixsize/60)*random_coord_y[i]))
                    print(coord_neighbours[i])
                    print(center)
                    print('\n')
            
            milca[i,:,:,0] = patch[-1]['fits'].data
            for j in range(len(self.bands)):
                inputs[i,:,:,j] = patch[j]['fits'].data

                #------------------------------------------------------------------#
                # # # # # Plots # # # # #
                #------------------------------------------------------------------#

                if plot == True:
                    fig = plt.figure(figsize=(20,5), tight_layout=False)
                    ax = fig.add_subplot(131)
                    divider = make_axes_locatable(ax)
                    cax = divider.append_axes('right', size='5%', pad=0.05)
                    im = ax.imshow(HDU.data, origin='lower')
                    ax.scatter(x,y)
                    ax.set_title('x={:.2f}, y={:.2f}'.format(x,y))
                    fig.colorbar(im, cax=cax, orientation='vertical')

                    ax = fig.add_subplot(132)
                    divider = make_axes_locatable(ax)
                    cax = divider.append_axes('right', size='5%', pad=0.05)
                    im = ax.imshow(mask, origin='lower')
                    ax.set_title('x={:.2f}, y={:.2f}'.format(x,y))
                    fig.colorbar(im, cax=cax, orientation='vertical')

                    ax = fig.add_subplot(133)
                    divider = make_axes_locatable(ax)
                    cax = divider.append_axes('right', size='5%', pad=0.05)
                    patch_ns = cutsky.cut_fits(coords_ns[i])
                    HDU_ns = patch_ns[6]['fits']
                    im = ax.imshow(HDU_ns.data, origin='lower')
                    ax.scatter(32,32)
                    ax.set_title('x={:.0f}, y={:.0f}'.format(32,32))
                    fig.colorbar(im, cax=cax, orientation='vertical')

                    plt.savefig(self.temp_path + 'random_mask_milca-y_%s'%i + '.png', bbox_inches='tight', transparent=False)
                    plt.show()
                    plt.close()


        #------------------------------------------------------------------#
        # # # # # Save files # # # # #
        #------------------------------------------------------------------#

        assert len(coords) == len(dataset_type)

        GenerateFiles.make_directory(self, path_to_file = self.output_path + 'files/' + 'f%s_d%s'%(self.freq, self.disk_radius))
        np.savez_compressed(self.output_path + 'files/f%s_d%s/'%(self.freq, self.disk_radius) + 'milca_n%s_f%s_'%(p, self.freq) + self.dataset, milca)
        np.savez_compressed(self.output_path + 'files/f%s_d%s/'%(self.freq, self.disk_radius) + 'type_n%s_f%s_'%(p, self.freq) + self.dataset, np.array(dataset_type))
        if p == 0:
            np.savez_compressed(self.dataset_path + 'type_test_f%s_'%(self.freq) + self.dataset, np.array(dataset_type))
        np.savez_compressed(self.output_path + 'files/f%s_d%s/'%(self.freq, self.disk_radius) + 'input_n%s_f%s_'%(p, self.freq) + self.dataset, inputs)
        np.savez_compressed(self.output_path + 'files/f%s_d%s/'%(self.freq, self.disk_radius) + 'label_n%s_f%s_'%(p, self.freq) + self.dataset, labels)


    def train_data_generator(self, loops, n_jobs = 1, plot=True):
        """Calls create_input n=loops times to create input/output datasets for all clusters in the selected cluster
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

        all_files = glob.glob(os.path.join(self.output_path + 'files/f%s_d%s/*.npz'%(self.freq, self.disk_radius)))
        for f in all_files:
            os.remove(f)

        Parallel(n_jobs=n_jobs)(delayed(self.create_input)(p, plot=plot) for p in tqdm(range(loops)))

        all_type = glob.glob(os.path.join(self.output_path + 'files/f%s_d%s/'%(self.freq, self.disk_radius), "type_n*.npz"))
        X = []
        for f in all_type:
            X.append(np.load(f)['arr_0'])
        dataset_type = np.concatenate(X, axis=0)
        np.savez_compressed(self.dataset_path + 'type_f%s_d%s'%(self.freq, self.disk_radius) + self.dataset, dataset_type)

        if plot == True:
            counts = Counter(dataset_type)
            df = pd.DataFrame.from_dict(counts, orient='index')
            ax = df.plot(kind='bar')
            ax.figure.savefig(self.output_path + 'figures/' + 'dataset_type_density' + '.png', bbox_inches='tight', transparent=False)

        all_type = glob.glob(os.path.join(self.output_path + 'files/f%s_d%s/'%(self.freq, self.disk_radius), "input_n*.npz"))
        X = []
        for f in all_type:
            X.append(np.load(f)['arr_0'])
        inputs = np.concatenate(X, axis=0)
        np.savez_compressed(self.dataset_path + 'input_f%s_d%s'%(self.freq, self.disk_radius) + self.dataset, inputs)

        all_type = glob.glob(os.path.join(self.output_path + 'files/f%s_d%s/'%(self.freq, self.disk_radius), "label_n*.npz"))
        X = []
        for f in all_type:
            X.append(np.load(f)['arr_0'])
        labels = np.concatenate(X, axis=0)
        np.savez_compressed(self.dataset_path + 'label_f%s_d%s'%(self.freq, self.disk_radius) + self.dataset, labels)
        
        all_type = glob.glob(os.path.join(self.output_path + 'files/f%s_d%s/'%(self.freq, self.disk_radius), "milca_n*.npz"))
        X = []
        for f in all_type:
            X.append(np.load(f)['arr_0'])
        milca = np.concatenate(X, axis=0)
        np.savez_compressed(self.dataset_path + 'milca_f%s_d%s'%(self.freq, self.disk_radius) + self.dataset, milca)

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

    def train_val_test_split(self, inputs_train, labels_train, dataset_type_train, inputs_test, labels_test, dataset_type_test, milca_test):
        """Splits input, output and milca datasets into training, validation and test.

        Args:
            inputs_train (np.ndarray): [description]
            labels_train (np.ndarray): [description]
            dataset_type_train (np.ndarray): [description]
            inputs_test (np.ndarray): [description]
            labels_test (np.ndarray): [description]
            dataset_type_test (np.ndarray): [description]
            milca_test (np.ndarray): [description]

        Returns:
            (np.ndarray): X_train; 
            (np.ndarray): X_val;
            (np.ndarray): X_test; 
            (np.ndarray): output_train; 
            (np.ndarray): output_val;
            (np.ndarray): output_test;
            (np.ndarray): M_test;
        """

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
        """Takes input, output and milca data from train_data_generator and splits it into training, validation and test set.
        Data is then individually standardized and compressed for each dataset.

        Args:
            leastsq (bool, optional): If True, sigma will be computed by fitting a gaussian up to mode. If False, sigma will be 
                                    computed using MAD. Defaults to False.
            range_comp (bool, optional): If True, range compression will be applied. Defaults to True.
            plot (bool, optional): If True, will plot distributions for data after standadization and range compression. Defaults to True.
        """

        inputs_train = np.load(self.dataset_path + 'input_f%s_d%s'%(self.freq, self.disk_radius) + self.dataset + '.npz')['arr_0']
        labels_train = np.load(self.dataset_path + 'label_f%s_d%s'%(self.freq, self.disk_radius) + self.dataset + '.npz')['arr_0']
        dataset_type_train = np.load(self.dataset_path + 'type_f%s_d%s'%(self.freq, self.disk_radius) + self.dataset + '.npz')['arr_0']

        inputs_test = np.load(self.output_path + 'files/f%s_d%s/'%(self.freq, self.disk_radius) + 'input_n0_f%s_'%self.freq + self.dataset + '.npz')['arr_0']
        labels_test = np.load(self.output_path + 'files/f%s_d%s/'%(self.freq, self.disk_radius) + 'label_n0_f%s_'%self.freq + self.dataset + '.npz')['arr_0']
        dataset_type_test = np.load(self.output_path + 'files/f%s_d%s/'%(self.freq, self.disk_radius) + 'type_n0_f%s_'%self.freq + self.dataset + '.npz')['arr_0']
        milca_test = np.load(self.output_path + 'files/f%s_d%s/'%(self.freq, self.disk_radius) + 'milca_n0_f%s_'%self.freq + self.dataset + '.npz')['arr_0']

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
            np.savez_compressed(self.dataset_path + 'input_train_pre_f%s_r0_'%self.freq + self.dataset, X_train)
            np.savez_compressed(self.dataset_path + 'input_val_pre_f%s_r0_'%self.freq + self.dataset, X_val)
            np.savez_compressed(self.dataset_path + 'input_test_pre_f%s_r0_'%self.freq + self.dataset, X_test)


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

        all_files = glob.glob(os.path.join(self.output_path + "files/f%s_d%s/"%(self.freq, self.disk_radius) + "*.npz"))
        for f in all_files:
            os.remove(f)
        os.rmdir(self.output_path + "files/f%s_d%s/"%(self.freq, self.disk_radius))
        os.remove(self.dataset_path + 'input_f%s_d%s'%(self.freq, self.disk_radius) + self.dataset + '.npz')
        os.remove(self.dataset_path + 'label_f%s_d%s'%(self.freq, self.disk_radius) + self.dataset + '.npz')
        os.remove(self.dataset_path + 'type_f%s_d%s'%(self.freq, self.disk_radius) + self.dataset + '.npz')
        os.remove(self.dataset_path + 'milca_f%s_d%s'%(self.freq, self.disk_radius) + self.dataset + '.npz')
        

        print('[preprocessing] Done!')

