#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 13:56:14 2019

@author: andrewpauling
"""
import numpy as np
from numba import njit, prange
from pyBL99.utils.sumall import sumall
from pyBL99.physics.snownrg import snownrg
from pyBL99.utils.calc_albedo import calc_albedo
from pyBL99.physics.growb import growb
from pyBL99.utils.tstmnew import tstmnew

import pyBL99.utils.constants as const
from copy import deepcopy


def thermo(dtau, state, internal_state, out_state, snofal, idter, iyear, iday):
    """
    heat budget

    need to read forcing Tair for entire grid
    add back in gairx,y
    deal with long dep of solar

    flux is positive towards the atm/ice or atm/ocean surface even if the flux
    is below the surface

    compute fluxes, temp and thickness over open water and ice
    """
    
    n1 = state['nlayers']
    nday = 86400/dtau

    io = 0     # solar transmitted through top surface if ice
    io1 = 0    # solar absorbed by first layer
    ib = 0     # solar transmitted through bottom surface of ice
    fneti = 0  # net flux into top surface of ice from atmosphere and ice
    fx = 0     # adjusted Fw accounting for times when ice melts away
    fneg = 0   # net flux exchanged between ice and ocean
    condb = 0  # conductive flux in ice at bottom surface
    condt = 0  # conductive flux in ice at top surface
    dq1 = 0    # emthalpy change of first layer

    delhib = 0  # ice thickness change at bottom for each category
    delhit = 0  # ice thickness change at top for each category (Excl. subl)
    delhs = 0   # snow thickness change for each category (excl. subl.)
    subi = 0    # ice thickness change for each category from sublimation
    subs = 0    # isnow thickness change for each category from sublimation

    # heat and energy accounting variables
    heat_init = 0
    heat_end = 0

    # compute the initial enthalpy stored in ice/snow and mixed layer and the
    # heat to check each step
    internal_state['e_init'] = sumall(state['hice'],
                                      state['hsnow'],
                                      state['eice'],
                                      state['esnow'],
                                      state['nlayers'])
    heat_init = np.copy(internal_state['heat_added'])
    fneg = 0

    if state['hice'] < 5:
        print('model cannot run without ice')
    else:
#        print('tice before: '+str(state.tice))
#        print('eice before: '+str(state.eice))
#        print('saltz before: '+str(state.saltz))
        # initialize ice temperature
        state.tice[1:(n1+1)] = gettmp(state.eice, state.saltz, state.nlayers)
 #       print('tice after: '+str(state.tice))
        # wipe out small amount of snow
        if state.hsnow < const.hsstar or state.esnow > 0:
            fneg = snownrg(state.hsnow, state.tice)
            state.hsnow = 0
            state.tice[0] = deepcopy(const.tsmelt)

        fneg = fneg/dtau

        albedo = calc_albedo(state.hsnow, state.ts)

        fsh_net = internal_state.fsh*(1-albedo)

        fracsnow = state.hsnow/(state.hsnow + 0.1*const.centi)
        io = 0
        if state.hsnow < const.hsstar:
            io = fsh_net*state.io_surf
            
        #print(state)
        #print(internal_state)

        state, internal_state, fneti, condb, dq1, io1, ib, condt, ulwr = \
            tstmnew(state,
                    internal_state,
                    io,
                    fsh_net,
                    dtau)

        state, delhib, delhs, delhit, subi, subs, fx = growb(state, fneti,
                                                             0., condb,
                                                             n1, nday, dtau)

        state.hice += delhit + delhib + subi
        state.hsnow += delhs + subs
        fneg += fx

        if snofal > 0.:
            hs_init = np.copy(state.hsnow)
            state.hsnow = np.maximum(const.hsstar, state.hsnow+snofal)
            dhs = state.hsnow - hs_init
            state.tice[0] = (state.tice[0]*hs_init +
                             const.tsmelt*dhs)/state.hsnow
            internal_state.heat_added -= dhs*const.rflsno

        # Energy budget diagnostics
        internal_state.heat_added += const.fw*dtau

        state.esnow = snownrg(state.hsnow, state.tice)
        internal_state.e_end = sumall(state.hice, state.hsnow, state.eice,
                                      state.esnow, n1)
        heat_end = np.copy(internal_state.heat_added)
        state.difference = ((internal_state.e_end-internal_state.e_init) -
                      (heat_end-heat_init))*0.001/dtau

        if idter == nday-1:
            nout = (iyear)*365+iday+1
            out_state.hiout[nout-1] = state.hice
            out_state.hsout[nout-1] = state.hsnow
            out_state.tsout[nout-1] = state.ts
            out_state.errout[nout-1] = state.difference

    return state, internal_state, out_state


def gettmp(eice, saltz, n1):

    layers = np.arange(n1)

    q = eice[layers] + const.rflice-const.rcpice*const.alpha*saltz[layers+1]
    b = -q/const.rcpice
    c = -const.gamma*saltz[layers+1]/const.rcpice

    b_2 = b/2
    tmp = -b_2-np.sqrt(b_2*b_2-c)

    return tmp
