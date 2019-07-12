#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 13:56:14 2019

@author: andrewpauling
"""
import numpy as np
from sumall import sumall
from snownrg import snownrg
from calc_albedo import calc_albedo
from growb import growb


def thermo(heat_added):
    """
    heat budget

    need to read forcing Tair for entire grid
    add back in gairx,y
    deal with long dep of solar

    flux is positive towards the atm/ice or atm/ocean surface even if the flux
    is below the surface

    compute fluxes, temp and thickness over open water and ice
    """

    global centi
    global rflsno, rcpice, rcpsno
    global tsmelt, tfrez

    global n1, nday, dtau, idter, iyear, iday
    global hsstar
    global fw, hice, hsnow, ts, tice, eice, esnow, saltz
    global hiout, hsout, tsout, errout
    global fsh, flo, upsens, upltnt, mualbedo, io_surf, snofal

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

    layer = 0

    # heat and energy accounting variables
    e_init = 0
    e_end = 0
    heat_init = 0
    heat_end = 0
    difference = 0

    # compute the initial enthalpy stored in ice/snow and mixed layer and the
    # heat to check each step
    e_init = sumall()
    heat_init = heat_added
    fneg = 0

    if hice < 5:
        print('model cannot run without ice')
    else:
        # initialize ice temperature
        tice[1:(n1+1)] = gettmp()

        if hsnow < hsstar or esnow > 0:  # wipe out small amount of snow
            fneg = snownrg()
            hsnow = 0
            tice[0] = tsmelt

        fneg = fneg/dtau

        albedo = calc_albedo()

        fsh_net = fsh*(1-albedo)

        fracsnow = hsnow/(hsnow + 0.1*centi)
        io = 0
        if hsnow < hsstar:
            io = fsh_net*io_surf

        heat_added, fneti, condb, dq1, io1, ib, condt, ulwr = tstmnew(
                flo,
                io,
                fsh_net,
                upltnt,
                upsens,
                heat_added)

        delhib, delhs, delhit, subi, subs, fx = growb(fneti, 0., condb)

        hice += delhit + delhib + subi
        hsnow += delhs + subs
        fneg += fx

        if snofal > 0.:
            hs_init = hsnow
            hsnow = np.maximum(hsstar, hsnow+snofal)
            dhs = hsnow - hs_init
            tice[0] = (tice[0]*hs_init + tsmelt*dhs)/hsnow
            heat_added -= dhs*rflsno

        # Energy budget diagnostics
        heat_added += fw*dtau

        esnow = snownrg()
        e_end = sumall()
        heat_end = heat_added
        difference = ((e_end-e_init) - (heat_end-heat_init))*0.001/dtau

        if idter == nday:
            nout = (iyear-1)*365+iday
            hiout[nout-1] = hice
            hsout[nout-1] = hsnow
            tsout[nout-1] = ts
            errout[nout-1] = difference

    return heat_added


def gettmp():

    global alpha, gamma
    global rflice, rcpice
    global eice, saltz
    global n1

    layers = np.arange(n1)

    q = eice[layers] = rflice-rcpice*alpha*saltz[layers+1]
    b = -q/rcpice
    c = -gamma*saltz[layers+1]/rcpice

    b_2 = b/2
    tmp = -b_2-np.sqrt(b_2*b_2-c)

    return tmp
