#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 18:56:21 2019

@author: andrewpauling
"""

import numpy as np
import matplotlib.pyplot as plt

import constants as const
from salinity_prof import salinity_prof
from snownrg import snownrg
from energ import energ


class ColumnModel():
    """
    Class containing the Bitz and Lipscomb 1999 sea ice column model

    An instance of "ColumnModel" is initialized with the following parameters

    :param float LW_pert:      longwave perturbation W/m^2

    :param int nyrs:            number of years to run for

    :param int timeofyear:     flag for time of year to apply LW perturbation
                               0 = all year (default)
                               1 = winter only
                               2 = summer only

    """

    def __init__(self,
                 LW_pert=0,
                 nyrs=20,
                 timeofyear=0):

        self.n1 = 10           # number of vertical layers
        self.nday = 2          # number of timesteps per day
        self.dtau = 86400/self.nday

        self.hice = 2.53*const.centi
        self.hsnow = 0.2827*const.centi
        self.tw = const.frzpt
        self.ts = const.tfrez

        # variables for time series
        self.hiout = np.zeros(nyrs*365)
        self.hsout = np.zeros(nyrs*365)
        self.tsout = np.zeros(nyrs*365)
        self.errout = np.zeros(nyrs*365)

        self.saltz = salinity_prof(self.n1)

        self.tice = np.zeros(self.n1+1)

        if self.n1 == 10:
            self.ts = -29.2894
            self.tice[0] = -23.1128
            self.tice[1] = -15.7925
            self.tice[2] = -14.1042
            self.tice[3] = -12.4534
            self.tice[4] = -10.8423
            self.tice[5] = -9.2777
            self.tice[6] = -7.7687
            self.tice[7] = -6.3255
            self.tice[8] = -4.9605
            self.tice[9] = -3.6879
            self.tice[10] = -2.5230
        else:
            self.tice[0] = -23.16-(const.tfrez+23.16)/self.n1
            for layer in range(1, self.n1+1):
                self.tice[layer] = -23.16+(layer-1)*(const.tfrez+23.16)/self.n1

        self.tbot = np.minimum(self.tw, const.tmelt)

        self.esnow = snownrg(self.hsnow, self.tice)
        self.layers = np.arange(1, self.n1+1)
        self.eice = energ(self.tice[self.layers], self.saltz[self.layers])
        
        
        
    
