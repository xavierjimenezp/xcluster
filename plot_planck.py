#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 01 10:00:58 2021

@author: Xavier Jimenez
"""


#------------------------------------------------------------------#
# # # # # Imports # # # # #
#------------------------------------------------------------------#

import matplotlib.pyplot as plt
import numpy as np
import healpy as hp

#------------------------------------------------------------------#
# # # # # Functions # # # # #
#------------------------------------------------------------------#

planck_path = '/n17data/PLANCK/'
planck_217 = hp.read_map(planck_path + "HFI_SkyMap_217-field-IQU_2048_R3.00_full.fits")
hp.mollview(
    planck_217,
    coord=["G", "E"],
    title="Planck HFI 217 GHz frequency map",
    unit="mK",
    norm="hist",
    min=-1,
    max=1,
)
hp.graticule()
