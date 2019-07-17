#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 18:56:21 2019

@author: andrewpauling
"""

import numpy as np
import matplotlib.pyplot as plt
import time
from attrdict import AttrDict

import pyBL99.utils.constants as const
from pyBL99.utils.salinity_prof import salinity_prof
from pyBL99.physics.snownrg import snownrg
from pyBL99.physics.energ import energ
from pyBL99.utils.sumall import sumall
from pyBL99.physics.thermo import thermo
from pyBL99.utils.snowfall import snowfall
from pyBL99.utils.state import initial_state, internal_state


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

        self.LW_pert = LW_pert
        self.nyrs = nyrs
        self.timeofyear = timeofyear

        self.nlayers = 10           # number of vertical layers

        # variables for time series
        self.hiout = np.zeros(nyrs*365)
        self.hsout = np.zeros(nyrs*365)
        self.tsout = np.zeros(nyrs*365)
        self.errout = np.zeros(nyrs*365)

        # initialize physical state
        self.state = initial_state(self.nlayers)

        # initialize internal state
        self.internal_state = internal_state()

        # compute the initial energy in the ice./snow and mixed layer;
        self.internal_state.e_init = sumall(self.state.hice, self.state.hsnow,
                                            self.state.eice, self.state.esnow,
                                            self.nlayers)

    def load_data(self):
        with open('data.mu71', 'r') as f:
            all_lines = [[float(num) for num in line.split()] for
                         line in f]

        data = np.array(all_lines)

        pertdays = np.ones(365)

        if self.timeofyear == 1:
            pertdays[:, 111:223] = 0
        elif self.timeofyear == 2:
            pertdays[:, np.concatenate(np.arange(111),
                                       np.arange(223, 365))] = 0

        return data, pertdays

    def compute(self, nperday=2):

        dtau = 86400/nperday  # timestep
        data, pertdays = self.load_data()

        start_time = time.time()

        for iyear in range(self.nyrs):
            if iyear+1 > self.nyrs-3:
                nday = 6
                dtau = 86400/nday    # time step

            for iday in range(365):

                # prepare to interpolate the forcing data
                # n1 = today, n = yesterday;
                self.internal_state.fsh_n = self.internal_state.fsh_n1
                self.internal_state.flo_n = self.internal_state.flo_n1
                self.internal_state.dnsens_n = self.internal_state.dnsens_n1
                self.internal_state.dnltnt_n = self.internal_state.dnltnt_n1
                self.internal_state.mualbedo_n = \
                    self.internal_state.mualbedo_n1
                self.internal_state.fsh_n1 = data[iday, 0]
                self.internal_state.flo_n1 = data[iday, 1]
                self.internal_state.dnsens_n1 = data[iday, 2]
                self.internal_state.dnltnt_n1 = data[iday, 3]
                self.internal_state.mualbedo_n1 = data[iday, 4]

                if self.internal_state.firststep:
                    self.internal_state.fsh_n = self.internal_state.fsh_n1
                    self.internal_state.flo_n = self.internal_state.flo_n1
                    self.internal_state.dnsens_n = self.internal_state.dnsens_n1
                    self.internal_state.dnltnt_n = self.internal_state.dnltnt_n1
                    self.internal_state.mualbedo_n = \
                        self.internal_state.mualbedo_n1
                    self.internal_state.firststep = False

#                for idter in range(nday):
#                    # linear spline daily forcing data forml timestep=day./nday
#                    fsh = fsh_n + (fsh_n1 - fsh_n)*(idter+1)/nday
#                    flo = self.LWpert*1000*pertdays[iday] + flo_n + \
#                        (flo_n1-flo_n)*(idter+1)/nday
#                    upsens = dnsens_n + (dnsens_n1-dnsens_n)*(idter+1)/nday
#                    upltnt = dnltnt_n + (dnltnt_n1-dnltnt_n)*(idter+1)/nday
#                    mualbedo = mualbedo_n + (mualbedo_n1-mualbedo_n) * \
#                        (idter+1)/nday
#                    mualbedo = mualbedo-0.0475
#
#                    io_surf = 0.3
#                    snofal = const.centi*snowfall(iday)/nday
#                    self.heat_added = thermo(dtau, self.heat_added, io_surf,
#                                             snofal)
#                    
#                print('finished year ' + iyear)
#
#        self.e_end = sumall(self.hice, self.hsnow, self.eice, self.esnow,
#                            self.n1)
#        end_time = time.time()
#
#        print('Energy Totals from the Run converted into  W/m^2')
#        print('energy change of system =' +
#              (self.e_end-self.e_init)*0.001/(self.nyrs*86400*365))
#        print('heat added to the ice/snow = ' +
#              self.heat_added*0.001/(self.nyrs*86400*365))
#        print('-->  difference: ' +
#              (self.e_end-self.e_init-self.heat_added) *
#              0.001/(self.nyrs*86400*365))
#        print('run time' + end_time-start_time)
#
#        print('Final Year Statistics')
#        htme = (self.nyrs-1)*365 + np.arange(1, 366)
#        tme = (self.nyrs-1)*365 + np.arange(32, 92)
#        print('Mean Thickness = ' + np.mean(self.hiout[htme-1]))
#        print('Mean Feb-Mar Temperature = ' + np.mean(self.tsout[tme-1]))
#
#        tme = np.arange(1, len(hiout)+1)/365

#    fig, (axtop, axbot) = plt.subplots(2, 2, figsize=(9, 6))
#    axtop[0].plot(tme, hiout)
#    axtop[0].set_xlabel('year')
#    axtop[0].set_ylabel('ice thickness - cm')
#
#    axtop[1].plot(tme, hsout)
#    axtop[1].set_xlabel('year')
#    axtop[1].set_ylabel('snow depth - cm')
#
#    axbot[0].plot(tme, tsout)
#    axbot.set_xlabel('year')
#    axbot[0].set_ylabel('surface temperature - C')
#
#    axbot[1].plot(tme, errout)
#    axbot[1].set_xlabel('year')
#    axbot[1].set_ylabel('error - W m^{-2}')
#
#    return hiout, hsout, tsout, errout
        
        
    
