#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 15:12:10 2019

@author: andrewpauling
"""
import numpy as np
from copy import deepcopy
from pyBL99.physics.energ import energ
from pyBL99.utils.sumall import sumall
from pyBL99.physics.snownrg import snownrg

import pyBL99.utils.constants as const


def growb(state, fneti, ultnt, condb, n1, nday, dtau):
    """
    Compute height change
    """

    alarm = False
    alarm2 = False

    state.esnow = snownrg(state.hsnow, state.tice)
    #print('eice before: '+str(state.eice))
    state.eice = energ(state.tice[1:n1+1], state.saltz[1:n1+1])
    #print('eice after: '+str(state.eice))
    enet = sumall(state.hice, state.hsnow, state.eice, state.esnow, n1)

    dhi = state.hice/n1*np.ones(n1)
    dhs = deepcopy(state.hsnow)
    delh = 0
    delhs = 0
    delb = 0
    subi = 0
    subs = 0

    if (fneti > 0 and not alarm):
        etop = fneti*dtau
        enet += etop

        if enet > 0:
            delh = -(state.hice + subi)
            # delhs = -(hsice + subs)
            fx = condb - enet/dtau
            alarm = True
            print('Melted through all layers from top')
        else:
            delh, delhs, alarm2 = surfmelt(state, etop, dhs, dhi, delh, delhs)
            dhs += delhs
            si = deepcopy(delh)
            for layer in np.arange(n1):
                s = np.maximum(-dhi[layer], si)
                dhi[layer] += s
                si -= s
            if alarm2:
                alarm = True
                print('surfmelt error')

    if not alarm:
        delht = delh + subi
        fx = deepcopy(const.fw)
        ebot = dtau * (const.fw-condb)

        if ebot < 0:
            egrow = energ(state.tbot, state.saltz[n1+1])
            delb = ebot/egrow
        else:
            # melt at bottom, melting temp = Tbot
            egrow = 0

            if (enet + ebot) > 0:
                delb = -(state.hice + delht)
                delhs = -(state.hsnow + delhs + subs)
                fx = condb - enet/dtau
                print('Melted through all layers from bottom')
                alarm = True
            else:
                delb, delh, delhs, alarm2 = botmelt(state, ebot, dhs, dhi, delh,
                                                    delhs, delb)
                if alarm2:
                    print('problem with botmelt')
                    alarm = True
        if not alarm:
            state.eice = adjust(state, egrow, delb, delht)

    return state, delb, delhs, delh, subi, subs, alarm


#def surfsub(state, enet, etop, dhs, dh, delh, delhs):
#    """
#    Compute subsurface melting
#    """
#
#    global rvlsno, rvlice, eice, esnow, n1, es, ei
#
#    u = esnow - rvlsno
#    finished = False
#    alarm = False
#
#    if u*dhs < 0:
#        # convert etop into equivalent snowmelt
#        delhs = etop/u
#        if (dhs + delhs) < 0:
#            # Melt only some of the snow
#            etop = 0.
#            enet = enet + esnow*delhs
#            finished = True
#        else:
#            # Melt all of the snow and some ice too
#            delhs = -dhs
#            etop = etop + u*dhs
#            enet = enet + es*delhs
#
#    if not finished:
#        for layer in np.arange(n1):
#            u = eice[layer] - rvlice
#            if (-u*dh[layer]) <= etop:
#                # melt partial layer
#                delh = delh + etop/u
#                enet = enet + etop/u*ei[layer]
#                etop = 0.
#                finished = True
#            else:
#                # melt out whole layer
#                delh = delh - dh[layer]
#                etop = etop + u*dh[layer]
#                enet = enet + dh[layer]*eice[layer]
#        if not finished:
#            print('ERROR in surfsub, etop')
#            alarm = True
#
#    return delh, delhs, alarm


def surfmelt(state, etop, dhs, dh, delh, delhs):
    """
    Compute surface melt
    """

    finished = False
    alarm = False

    u = deepcopy(state.esnow)

    if (u*dhs < 0):
        # convert etop into equilvalent snowmelt
        delhs = etop/u
        if (dhs + delhs) > 0:
            # Melt only some of the snow
            etop = 0
            finished = True
        else:
            # Melt all of the snow and some ie too
            delhs = -dhs
            etop += u*dhs

    if not finished:
        for layer in range(state.nlayers):
            u = deepcopy(state.eice[layer])
            if -u*dh[layer] >= etop:
                delh += etop/u
                etop = 0
                finished = True
            else:
                delh -= dh[layer]
                etop += u*dh[layer]

        if not finished:
            print('ERROR in surfmelt')
            alarm = True

    return delh, delhs, alarm


def botmelt(state, ebot, dhs, dh, delh, delhs, delb):
    """
    Compute bottom melt
    """

    finished = False
    alarm = False

    for layer in np.arange(state.nlayers-1, -1, -1):
        u = deepcopy(state.eice[layer])
        if -u*dh[layer] >= ebot:
            delb += ebot/u
            ebot = 0
            finished = True
        else:
            delb -= dh[layer]
            ebot += u*dh[layer]

    if not finished:
        # Finally, melt snow if nexessary
        u = deepcopy(state.esnow)
        if -u*dhs >= ebot:
            delhs += ebot/u
            ebot = 0.
            finished = True
        else:
            print('melted completely through all ice and snow')
            print('ERROR in botmelt')
            alarm = True

    return delb, delh, delhs, alarm


def adjust(state, egrow, delb, delh):
    """
    Adjusts temperature profile after melting/growing

    eice is the energy density after updating tice from the heat equation
    without regard to delh and delb

    hice is the thickness from previous time step!
    h_tw us the NEW thickness

    delb is negative if there is melt at the bottom
    delh is negstive if there is melt at the top

    generally _tw is a suffix to label the new layer spacing variables
    """

    n1 = state.nlayers
    e_tw = deepcopy(state.eice)

    if not ((np.abs(delb) < const.tiny) and (delh > -const.tiny)):
        h_tw = state.hice + delb + delh

        if h_tw <= 0.:
            e_tw = np.zeros(n1)
        else:
            # layer thickness
            delta = state.hice/n1
            delta_tw = h_tw/n1

            # z is positive down and zero is relative to the top of the ice
            # from the old time step
            z = np.zeros(n1+2)
            z_tw = np.zeros(n1+1)
            z_tw[0] = -delh

            layers = np.arange(2, n1+1)
            z[layers-1] = delta*(layers-1)
            z_tw[layers-1] = z_tw[0] + delta_tw*(layers-1)

            z[n1] = deepcopy(state.hice)
            z[n1+1] = state.hice + np.maximum(delb, 0.)
            z_tw[n1] = z_tw[0] + h_tw

            fract = np.zeros((n1, n1+1))

            for l_tw in range(n1):
                for l in range(n1+1):
                    fract[l_tw, l] = np.minimum(z_tw[l_tw+1], z[l+1]) - \
                         np.maximum(z_tw[l_tw], z[l])

            fract = fract/delta_tw
            fract = np.maximum(fract, 0)

            tmp = np.append(state.eice, egrow)
            e_tw = tmp @ fract.T

    return e_tw
