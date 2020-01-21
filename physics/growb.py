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
    

    Parameters
    ----------
    state : dict
        Current ice state dictionary
    fneti : float
    ultnt : float
    condb : float
    n1 : int
    nday : int
    dtau : float

    Returns
    -------
    state : dict
        Updated state dictionary
    delb : float
        change in thickness at bottom
    delhs : float
        change in snow depth
    delh : float
        total thickness change
    subi : float
        sublimation of ice
    subs : float
        sublimation of snow
    alarm : bool
       alarm flag

    """

    alarm = False
    alarm2 = False

    esnow = np.copy(state['esnow'])
    hsnow = np.copy(state['hsnow'])
    tice = np.copy(state['tice'])
    eice = np.copy(state['eice'])
    saltz = np.copy(state['saltz'])
    hice = np.copy(state['hice'])
    tbot = np.copy(state['tbot'])

    esnow = snownrg(hsnow, tice)
    eice = energ(tice[1:n1+1], saltz[1:n1+1])
    enet = sumall(hice, hsnow, eice, esnow, n1)

    dhi = hice/n1*np.ones(n1)
    dhs = np.copy(hsnow)
    delh = 0
    delhs = 0
    delb = 0
    subi = 0
    subs = 0

    if (fneti > 0 and not alarm):
        etop = fneti*dtau
        enet += etop

        if enet > 0:
            delh = -(hice + subi)
            # delhs = -(hsice + subs)
            fx = condb - enet/dtau
            alarm = True
            print('Melted through all layers from top')
        else:
            delh, delhs, alarm2 = surfmelt(esnow, eice, etop, dhs, dhi, delh,
                                           delhs, n1)
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
        fx = np.copy(const.fw)
        ebot = dtau * (const.fw-condb)

        if ebot < 0:
            egrow = energ(tbot, saltz[n1+1])
            delb = ebot/egrow
        else:
            # melt at bottom, melting temp = Tbot
            egrow = 0

            if (enet + ebot) > 0:
                delb = -(hice + delht)
                delhs = -(hsnow + delhs + subs)
                fx = condb - enet/dtau
                print('Melted through all layers from bottom')
                alarm = True
            else:
                delb, delh, delhs, alarm2 = botmelt(esnow, eice, ebot, dhs,
                                                    dhi, delh,
                                                    delhs, delb, n1)
                if alarm2:
                    print('problem with botmelt')
                    alarm = True
        if not alarm:
            eice = adjust(eice, hice, egrow, delb, delht, n1)

        state['hice'] = hice
        state['hsnow'] = hsnow
        state['eice'] = eice
        state['esnow'] = esnow
        state['tice'] = tice
        state['tbot'] = tbot
        state['saltz'] = saltz

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


def surfmelt(esnow, eice, etop, dhs, dh, delh, delhs, n1):
    """
    Compute surface melt
    """

    finished = False
    alarm = False

    u = np.copy(esnow)

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
        for layer in range(n1):
            u = np.copy(eice[layer])
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


def botmelt(esnow, eice, ebot, dhs, dh, delh, delhs, delb, n1):
    """
    Compute bottom melt
    """

    finished = False
    alarm = False

    for layer in np.arange(n1-1, -1, -1):
        u = np.copy(eice[layer])
        if -u*dh[layer] >= ebot:
            delb += ebot/u
            ebot = 0
            finished = True
        else:
            delb -= dh[layer]
            ebot += u*dh[layer]

    if not finished:
        # Finally, melt snow if nexessary
        u = np.copy(esnow)
        if -u*dhs >= ebot:
            delhs += ebot/u
            ebot = 0.
            finished = True
        else:
            print('melted completely through all ice and snow')
            print('ERROR in botmelt')
            alarm = True

    return delb, delh, delhs, alarm


def adjust(eice, hice, egrow, delb, delh, n1):
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

    e_tw = np.copy(eice)

    if not ((np.abs(delb) < const.tiny) and (delh > -const.tiny)):
        h_tw = hice + delb + delh

        if h_tw <= 0.:
            e_tw = np.zeros(n1)
        else:
            # layer thickness
            delta = hice/n1
            delta_tw = h_tw/n1

            # z is positive down and zero is relative to the top of the ice
            # from the old time step
            z = np.zeros(n1+2)
            z_tw = np.zeros(n1+1)
            z_tw[0] = -delh

            layers = np.arange(2, n1+1)
            z[layers-1] = delta*(layers-1)
            z_tw[layers-1] = z_tw[0] + delta_tw*(layers-1)

            z[n1] = np.copy(hice)
            z[n1+1] = hice + np.maximum(delb, 0.)
            z_tw[n1] = z_tw[0] + h_tw

            fract = np.zeros((n1, n1+1))

#            for l_tw in np.arange(n1):
#                for l in np.arange(n1+1):
#                    fract[l_tw, l] = np.minimum(z_tw[l_tw+1], z[l+1]) - \
#                         np.maximum(z_tw[l_tw], z[l])

            for l_tw in np.arange(n1):
                fract[l_tw, :] = np.minimum(z_tw[l_tw+1], z[1:]) - \
                    np.maximum(z_tw[l_tw], z[:-1])

            fract = fract/delta_tw
            fract = np.maximum(fract, 0)

            tmp = np.append(eice, egrow)
            e_tw = tmp @ fract.T

    return e_tw
