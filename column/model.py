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
from copy import deepcopy

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
                 nyrs=1,
                 timeofyear=0):

        self.LW_pert = LW_pert
        self.nyrs = nyrs
        self.timeofyear = timeofyear

        self.nlayers = 10           # number of vertical layers
        
        self.out_state = AttrDict()

        # variables for time series
        self.out_state['hiout'] = np.zeros(nyrs*365)
        self.out_state['hsout'] = np.zeros(nyrs*365)
        self.out_state['tsout'] = np.zeros(nyrs*365)
        self.out_state['errout'] = np.zeros(nyrs*365)

        # initialize physical state
        self.state = initial_state(self.nlayers)

        # initialize internal state
        self.internal_state = internal_state()

        # compute the initial energy in the ice./snow and mixed layer;
        self.internal_state.e_init = sumall(self.state.hice, self.state.hsnow,
                                            self.state.eice, self.state.esnow,
                                            self.nlayers)

    def load_data(self):
        ddir = '/Users/andrewpauling/Documents/PhD/bl99/pyBL99/column/'
        dfile = 'data.mu71'
        with open(ddir + dfile, 'r') as f:
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
                nperday = 6
                dtau = 86400/nperday    # time step

            for iday in range(200):
                print('day = '+str(iday))
                # prepare to interpolate the forcing data
                # n1 = today, n = yesterday;
                self.internal_state.fsh_n = \
                    deepcopy(self.internal_state.fsh_n1)
                self.internal_state.flo_n = \
                    deepcopy(self.internal_state.flo_n1)
                self.internal_state.dnsens_n = \
                    deepcopy(self.internal_state.dnsens_n1)
                self.internal_state.dnltnt_n = \
                    deepcopy(self.internal_state.dnltnt_n1)
                self.internal_state.mualbedo_n = \
                    deepcopy(self.internal_state.mualbedo_n1)
                self.internal_state.fsh_n1 = data[iday, 0]
                self.internal_state.flo_n1 = data[iday, 1]
                self.internal_state.dnsens_n1 = data[iday, 2]
                self.internal_state.dnltnt_n1 = data[iday, 3]
                self.internal_state.mualbedo_n1 = data[iday, 4]

                if self.internal_state.firststep:
                    self.internal_state.fsh_n = \
                        deepcopy(self.internal_state.fsh_n1)
                    self.internal_state.flo_n = \
                        deepcopy(self.internal_state.flo_n1)
                    self.internal_state.dnsens_n = \
                        deepcopy(self.internal_state.dnsens_n1)
                    self.internal_state.dnltnt_n = \
                        deepcopy(self.internal_state.dnltnt_n1)
                    self.internal_state.mualbedo_n = \
                        deepcopy(self.internal_state.mualbedo_n1)
                    self.internal_state.firststep = False

                for idter in range(nperday):
                    # linear spline daily forcing data forml timestep=day./nday
                    self.internal_state['fsh'] = self.internal_state.fsh_n + \
                        (self.internal_state.fsh_n1 -
                         self.internal_state.fsh_n)*(idter+1)/nperday
                    self.internal_state['flo'] = \
                        self.LW_pert*1000*pertdays[iday] + \
                        self.internal_state.flo_n + \
                        (self.internal_state.flo_n1 -
                         self.internal_state.flo_n)*(idter+1)/nperday
                    self.internal_state['upsens'] = \
                        self.internal_state.dnsens_n + \
                        (self.internal_state.dnsens_n1 -
                         self.internal_state.dnsens_n)*(idter+1)/nperday
                    self.internal_state['upltnt'] = \
                        self.internal_state.dnltnt_n + \
                        (self.internal_state.dnltnt_n1 -
                         self.internal_state.dnltnt_n)*(idter+1)/nperday
                    self.internal_state['mualbedo'] = \
                        self.internal_state.mualbedo_n + \
                        (self.internal_state.mualbedo_n1 -
                         self.internal_state.mualbedo_n) * \
                        (idter+1)/nperday
                    self.internal_state['mualbedo'] -= 0.0475

                    snofal = const.centi*snowfall(iday)/nperday
                    self.state, self.internal_state, self.out_state = \
                        thermo(dtau,
                               self.state,
                               self.internal_state,
                               self.out_state,
                               snofal,
                               idter,
                               iyear,
                               iday)

            print('finished year ' + str(iyear))

        self.internal_state.e_end = sumall(self.state.hice, self.state.hsnow,
                                           self.state.eice, self.state.esnow,
                                           self.state.nlayers)
        end_time = time.time()

        print('Energy Totals from the Run converted into  W/m^2')
        print('energy change of system =' + str(
              (self.internal_state.e_end -
               self.internal_state.e_init)*0.001/(self.nyrs*86400*365)))
        print('heat added to the ice/snow = ' + str(
              self.internal_state.heat_added*0.001/(self.nyrs*86400*365)))
        print('-->  difference: ' + str(
              (self.internal_state.e_end-self.internal_state.e_init -
               self.internal_state.heat_added) *
              0.001/(self.nyrs*86400*365)))
        print('run time = ' + str(end_time-start_time))
        print(' ')
        print('Final Year Statistics')
        htme = (self.nyrs-1)*365 + np.arange(1, 366)
        tme = (self.nyrs-1)*365 + np.arange(32, 92)
        print('Mean Thickness = ' + str(np.mean(self.out_state.hiout[htme-1])))
        print('Mean Feb-Mar Temperature = ' +
              str(np.mean(self.out_state.tsout[tme-1])))

        self.plot()

    def plot(self):

        tme = np.arange(1, len(self.out_state.hiout)+1)/365

        fig, (axtop, axbot) = plt.subplots(2, 2, figsize=(9, 6))
        axtop[0].plot(tme, self.out_state.hiout)
        axtop[0].set_xlabel('year')
        axtop[0].set_ylabel('ice thickness - cm')

        axtop[1].plot(tme, self.out_state.hsout)
        axtop[1].set_xlabel('year')
        axtop[1].set_ylabel('snow depth - cm')

        axbot[0].plot(tme, self.out_state.tsout)
        axbot[0].set_xlabel('year')
        axbot[0].set_ylabel('surface temperature - C')

        axbot[1].plot(tme, self.out_state.errout)
        axbot[1].set_xlabel('year')
        axbot[1].set_ylabel('error - W m^{-2}')

    
        
        
    
