#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 16:12:42 2019

@author: andrewpauling
"""
from init_column import init_column
import time
from sumall import sumall
import numpy as np
from thermo import thermo
import matplotlib.pyplot as plt

def driver(LWpert=0, Nyr=20, timeofyear=0):
    """ 
    [HIOUT, HSOUT, TSOUT, ERROUT]=COLUMN(LWPERT,NYR,TIMEOFYEAR)  
    Driver for Bitz and Lipscomb column model.
    If no options are given the model runs with default 
    values for inputs LWPERT, NYR, and TIMEOFYEAR.

    Be sure to type a semicolon after calling this routine.
   
    For example, type: 
    [hiout, hsout] = column(-2,50);    
    or
    [hiout, hsout] = column(2,20,2);
    or
    column(8);

    Intended values for the options are 
      LWPERT downward longwave perturbation:  -10 to 10 (default 
          is zero), units are W/m2
      NYRS   run length: 10 to 150 (default is 20), units are years
      TIMEOFYEAR portion of year when LWPERT is applied, 0, 1, or 
          2 = allyear, winter only, or summer only, respectively
          (default is 0).  Winter only is from autumnal to vernal 
          equinox.
    The user may specify LWPERT alone, LWPERT and NYR alone, or all 
    three options.

    The output variables are timeseries of daily values for
      HIOUT ice thickness in cm
      HSOUT snow thickness in cm
      TSOUT surface temperature in deg C
      ERROUT error in the energy conservation converted into W/m2

    A figure is plotted with the daily timeseries


    C.M. Bitz, June 24, 2007
    """
    global n1, nday, dtau, idter, iyear, iday
    global fsh, flo, upsens, upltnt, mualbedo, io_surf, snofal
    global hiout, hsout, tsout, errout
    global centi

    nyrs = Nyr    # run length

    n1 = 10   # number of layers
    nday = 2
    dtau = 86400/nday    # time step,
    # note that nday is increased in last 3 yrs of the run

    init_column(nyrs, n1)   # initialize a bunch of other global vars

    # declare vars for this routine
    start_time = time.time()
    firststep = True
    fsh_n1 = 0
    flo_n1 = 0
    dnsens_n1 = 0
    dnltnt_n1 = 0
    mualbedo_n1 = 0
    fsh_n = 0
    flo_n = 0
    dnsens_n = 0
    dnltnt_n = 0
    mualbedo_n = 0
    e_init = 0  # energy in the ice and snow
    e_end = 0
    heat_added = 0  # running total of heat added to the ice and snow

    # compute the initial energy in the ice./snow and mixed layer;
    e_init = sumall()

    # data from fletcher 1965, used by maykut and untersteiner;
    # interpolated from monthly to daily without preserving monthly means.
    # This is less than ideal
    with open('data.mu71', 'r') as f:
        all_lines = [[float(num) for num in line.split()] for line in f]

    data = np.array(all_lines)

    pertdays = np.ones(365)
    equin = 172
    if timeofyear == 1:
        pertdays[:, 111:223] = 0
    elif timeofyear == 2:
        pertdays[:, np.concatenate(np.arange(111), np.arange(223, 365))] = 0

    for iyear in range(nyrs):
        if iyear+1 > nyrs-3:
            nday = 6
            dtau = 86400/nday    # time step

        for iday in range(365):

            # prepare to interpolate the forcing data
            # n1 = today, n = yesterday;
            fsh_n = fsh_n1
            flo_n = flo_n1
            dnsens_n = dnsens_n1
            dnltnt_n = dnltnt_n1
            mualbedo_n = mualbedo_n1
            fsh_n1 = data(iday, 0)
            flo_n1 = data(iday, 1)
            dnsens_n1 = data(iday, 2)
            dnltnt_n1 = data(iday, 3)
            mualbedo_n1 = data(iday, 4)

            if firststep:
                fsh_n = fsh_n1
                flo_n = flo_n1
                dnsens_n = dnsens_n1
                dnltnt_n = dnltnt_n1
                mualbedo_n = mualbedo_n1
                firststep = False

            for idter in range(nday):
                # linear spline daily forcing data forml timestep=day./nday;
                fsh = fsh_n + (fsh_n1 - fsh_n)*(idter+1)/nday
                flo = LWpert*1000*pertdays[iday] + flo_n + \
                    (flo_n1-flo_n)*(idter+1)/nday
                upsens = dnsens_n + (dnsens_n1-dnsens_n)*(idter+1)/nday
                upltnt = dnltnt_n + (dnltnt_n1-dnltnt_n)*(idter+1)/nday
                mualbedo = mualbedo_n + (mualbedo_n1-mualbedo_n)*(idter+1)/nday
                mualbedo = mualbedo-0.0475

                io_surf = 0.3
                snofal = centi*snowfall[iday]/nday
                heat_added = thermo(heat_added)
    print('finished year ' + iyear)

    e_end = sumall()
    end_time = time.time()

    print('Energy Totals from the Run converted into  W/m^2')
    print('energy change of system =' + (e_end-e_init)*0.001/(nyrs*86400*365))
    print('heat added to the ice/snow = ' + heat_added*0.001/(nyrs*86400*365))
    print('-->  difference: ' +
          (e_end-e_init-heat_added)*0.001/(nyrs*86400*365))
    print('run time' + end_time-start_time)

    print('Final Year Statistics')
    htme = (nyrs-1)*365 + np.arange(1, 366)
    tme = (nyrs-1)*365 + np.arange(32, 92)
    print('Mean Thickness = ' + np.mean(hiout[htme-1]))
    print('Mean Feb-Mar Temperature = ' + np.mean(tsout[tme-1]))

    tme = np.arange(1, len(hiout)+1)/365

    fig, (axtop, axbot) = plt.subplots(2, 2, figsize=(9, 6))
    axtop[0].plot(tme, hiout)
    axtop[0].set_xlabel('year')
    axtop[0].set_ylabel('ice thickness - cm')

    axtop[1].plot(tme, hsout)
    axtop[1].set_xlabel('year')
    axtop[1].set_ylabel('snow depth - cm')

    axbot[0].plot(tme, tsout)
    axbot.set_xlabel('year')
    axbot[0].set_ylabel('surface temperature - C')

    axbot[1].plot(tme, errout)
    axbot[1].set_xlabel('year')
    axbot[1].set_ylabel('error - W m^{-2}')

    return hiout, hsout, tsout, errout


def snowfall(idx):
    """
    snowfall from maykut and untersteiner 1971
    """

    snow = 0

    if idx <= 119 or idx >= 302:
        snow = 2.79e-4
    elif idx >= 120 and idx <= 150:
        snow = 1.61e-3
    elif idx > 230:
        snow = 4.16e-3
    else:
        snow = 0.0

    return snow

if __name__ == '__main__':
    hiout, hsout, tsout, errout = driver()
