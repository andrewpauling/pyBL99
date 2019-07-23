#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 18:56:21 2019

@author: andrewpauling
"""

import time
from copy import deepcopy
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from attrdict import AttrDict

import pyBL99.utils.constants as const
from pyBL99.utils.sumall import sumall
from pyBL99.physics.thermo import thermo
from pyBL99.utils.snowfall import snowfall
from pyBL99.utils.state import initial_state, internal_state

mpl.rcParams['figure.figsize'] = 12.0, 8.0
mpl.rcParams['font.size'] = 18.0


class ColumnModel():
    """
    Class containing the Bitz and Lipscomb 1999 sea ice column model

    An instance of "ColumnModel" is initialized with the following parameters

    :param float LW_pert:      longwave perturbation W/m^2. Default is 0.

    :param int nyrs:           number of years to run for. Default is 20.

    :param int timeofyear:     flag for time of year to apply LW perturbation
                               0 = all year (default)
                               1 = winter only
                               2 = summer only

    """

    def __init__(self,
                 lw_pert=0,
                 nyrs=20,
                 timeofyear=0):

        self.lw_pert = lw_pert
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
        self.e_start = sumall(self.state.hice, self.state.hsnow,
                              self.state.eice, self.state.esnow,
                              self.nlayers)

        # Attribute that will contain the ice/snow/mixed layer at the end
        self.e_finish = 0

    def load_data(self):
        """
        Loads input data from Maykut and Untersteiner (1971)
        """
        ddir = '/Users/andrewpauling/Documents/PhD/bl99/pyBL99/column/'
        dfile = 'data.mu71'
        with open(ddir + dfile, 'r') as file:
            all_lines = [[float(num) for num in line.split()] for
                         line in file]

        data = np.array(all_lines)

        pertdays = np.ones(365)

        if self.timeofyear == 1:
            pertdays[:, 111:223] = 0
        elif self.timeofyear == 2:
            pertdays[:, np.concatenate(np.arange(111),
                                       np.arange(223, 365))] = 0

        return data, pertdays

    def compute(self, nperday=2):
        """
        Integrate the model forward for number of years = nyrs
        """

        dtau = 86400/nperday  # timestep
        data, pertdays = self.load_data()

        start_time = time.time()

        for iyear in range(self.nyrs):
            if iyear+1 > self.nyrs-3:
                nperday = 6
                dtau = 86400/nperday    # time step

            for iday in range(365):
                # prepare to interpolate the forcing data
                # n1 = today, n = yesterday;
                self.internal_state['fsh_n'] = \
                    deepcopy(self.internal_state['fsh_n1'])
                self.internal_state['flo_n'] = \
                    deepcopy(self.internal_state['flo_n1'])
                self.internal_state['dnsens_n'] = \
                    deepcopy(self.internal_state['dnsens_n1'])
                self.internal_state['dnltnt_n'] = \
                    deepcopy(self.internal_state['dnltnt_n1'])
                self.internal_state['mualbedo_n'] = \
                    deepcopy(self.internal_state['mualbedo_n1'])
                self.internal_state['fsh_n1'] = data[iday, 0]
                self.internal_state['flo_n1'] = data[iday, 1]
                self.internal_state['dnsens_n1'] = data[iday, 2]
                self.internal_state['dnltnt_n1'] = data[iday, 3]
                self.internal_state['mualbedo_n1'] = data[iday, 4]

                if self.internal_state['firststep']:
                    self.internal_state['fsh_n'] = \
                        deepcopy(self.internal_state['fsh_n1'])
                    self.internal_state['flo_n'] = \
                        deepcopy(self.internal_state['flo_n1'])
                    self.internal_state['dnsens_n'] = \
                        deepcopy(self.internal_state['dnsens_n1'])
                    self.internal_state['dnltnt_n'] = \
                        deepcopy(self.internal_state['dnltnt_n1'])
                    self.internal_state['mualbedo_n'] = \
                        deepcopy(self.internal_state['mualbedo_n1'])
                    self.internal_state['firststep'] = False

                for idter in range(nperday):
                    # linear spline daily forcing data forml timestep=day./nday
                    self.internal_state['fsh'] = \
                        self.internal_state['fsh_n'] + \
                        (self.internal_state['fsh_n1'] -
                         self.internal_state['fsh_n'])*(idter+1)/nperday
                    self.internal_state['flo'] = \
                        self.lw_pert*1000*pertdays[iday] + \
                        self.internal_state['flo_n'] + \
                        (self.internal_state['flo_n1'] -
                         self.internal_state['flo_n'])*(idter+1)/nperday
                    self.internal_state['upsens'] = \
                        self.internal_state['dnsens_n'] + \
                        (self.internal_state['dnsens_n1'] -
                         self.internal_state['dnsens_n'])*(idter+1)/nperday
                    self.internal_state['upltnt'] = \
                        self.internal_state['dnltnt_n'] + \
                        (self.internal_state['dnltnt_n1'] -
                         self.internal_state['dnltnt_n'])*(idter+1)/nperday
                    self.internal_state['mualbedo'] = \
                        self.internal_state['mualbedo_n'] + \
                        (self.internal_state['mualbedo_n1'] -
                         self.internal_state['mualbedo_n']) * \
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

        self.e_finish = sumall(self.state['hice'], self.state['hsnow'],
                               self.state['eice'], self.state['esnow'],
                               self.state['nlayers'])

        print('e_init = ' + str(self.e_start))
        print('e_end = ' + str(self.e_finish))
        end_time = time.time()

        print('Energy Totals from the Run converted into  W/m^2')
        print('energy change of system =' + str(
            (self.e_finish -
             self.e_start)*0.001/(self.nyrs*86400*365)))
        print('heat added to the ice/snow = ' + str(
            self.internal_state['heat_added']*0.001/(self.nyrs*86400*365)))
        print('-->  difference: ' + str(
            (self.e_finish-self.e_start -
             self.internal_state['heat_added']) *
            0.001/(self.nyrs*86400*365)))
        print('run time = ' + str(end_time-start_time))
        print(' ')
        print('Final Year Statistics')
        htme = (self.nyrs-1)*365 + np.arange(1, 366)
        tme = (self.nyrs-1)*365 + np.arange(32, 92)
        print('Mean Thickness = ' +
              str(np.mean(self.out_state['hiout'][htme-1])))
        print('Mean Feb-Mar Temperature = ' +
              str(np.mean(self.out_state['tsout'][tme-1])))

        self.plot()

    def plot(self):
        """
        Generates plot at the end of the integration
        """

        tme = np.arange(1, len(self.out_state.hiout)+1)/365

        fig, (axtop, axbot) = plt.subplots(2, 2, figsize=(9, 6))
        axtop[0].plot(tme, self.out_state['hiout'])
        axtop[0].set_xlabel('year')
        axtop[0].set_ylabel('ice thickness - cm')

        axtop[1].plot(tme, self.out_state['hsout'])
        axtop[1].set_xlabel('year')
        axtop[1].set_ylabel('snow depth - cm')

        axbot[0].plot(tme, self.out_state['tsout'])
        axbot[0].set_xlabel('year')
        axbot[0].set_ylabel('surface temperature - C')

        axbot[1].plot(tme, self.out_state['errout'])
        axbot[1].set_xlabel('year')
        axbot[1].set_ylabel('error - W m^{-2}')

        plt.show()
