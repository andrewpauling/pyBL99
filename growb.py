#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 15:12:10 2019

@author: andrewpauling
"""
import numpy as np
from energ import energ
from sumall import sumall
from snownrg import snownrg


def dh(fneti, ultnt, condb):
    """
    Compute height change
    """

    global rflsno, rcpsno
    global tsmelt
    global n1, nday, dtau
    global tiny, hsmin, hsstar
    global frzpt, fw, hice, hsice, hsnow, tbot, ts, tice, eice
    global esnow, saltz, rflice, rcpice, alpha, gamma

    alarm = False
    alarm2 = False

    esnow = snownrg()
    eice = energ(tice[1:n1+1], saltz[1:n1+1])
    enet = sumall()

    dhi = hice/n1*np.ones(n1)
    dhs = hsnow.copy()
    delh = 0
    delhs = 0
    delb = 0
    subi = 0
    subs = 0

    if (fneti > 0 and not alarm):
        etop = fneti*dtau
        enet = enet + etop

        if enet > 0:
            delh = -(hice + subi)
            delhs = -(hsice + subs)
            fx = condb - enet/dtau
            alarm = True
            print('Melted through all layers from top')
        else:
            delh, delhs, alarm2 = surfmelt(etop, dhs, dhi, delh, delhs)
            dhs = dhs + delhs
            si = delh.copy()
            for layer in np.arange(n1):
                s = np.maximum(-dhi[layer], si)
                dhi[layer] = dhi[layer] + s
                si -= s
            if alarm2:
                alarm = True
                print('surfmelt error')

    if not alarm:
        delht = delh + subi
        fx = fw.copy()
        ebot = dtau * (fw-condb)

        if ebot < 0:
            egrow = energ(tbot, saltz[n1+1]z)
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
                delb, delh, delhs, alarm2 = botmelt(ebot, dhs, dhi, delh,
                                                    delhs, delb)
                if alarm2:
                    print('problem with botmelt')
                    alarm = True
        if not alarm:
            eice = adjust(egrow, delb, delht)

    return delb, delhs, delh, subi, subs, alarm


def surfsub(enet, etop, dhs, dh, delh, delhs):
    """
    Compute subsurface melting
    """

    global rvlsno, rvlice, eice, esnow, n1, es, ei

    u = esnow - rvlsno
    finished = False
    alarm = False

    if u*dhs < 0:
        # convert etop into equivalent snowmelt
        delhs = etop/u
        if (dhs + delhs) < 0:
            # Melt only some of the snow
            etop = 0.
            enet = enet + esnow*delhs
            finished = True
        else:
            # Melt all of the snow and some ice too
            delhs = -dhs
            etop = etop + u*dhs
            enet = enet + es*delhs

    if not finished:
        for layer in np.arange(n1):
            u = eice[layer] - rvlice
            if (-u*dh[layer]) <= etop:
                # melt partial layer
                delh = delh + etop/u
                enet = enet + etop/u*ei[layer]
                etop = 0.
                finished = True
            else:
                # melt out whole layer
                delh = delh - dh[layer]
                etop = etop + u*dh[layer]
                enet = enet + dh[layer]*eice[layer]
        if not finished:
            print('ERROR in surfsub, etop')
            alarm = True

    return delh, delhs, alarm


def surfmelt(etop, dhs, dh, delh, delhs):
    """
    Compute surface melt
    """

    global eice, esnow, n1, tice, tsmelt, rflsno, rcpsno

    finished = False
    alarm = False

    u = esnow.copy()

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
            u = eice[layer]
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


def botmelt(ebot, dhs, dh, delh, delhs, delb):
    """
    Compute bottom melt
    """

    global eice, esnow, n1, hsnow

    finished = False
    alarm = False

    for layer in np.arange(n1-1, -1, -1):
        u = eice[layer]
        if -u*dh[layer] >= ebot:
            delb += ebot/u
            ebot = 0
            finished = True
        else:
            delb -= dh[layer]
            ebot += u*dh[layer]

    if not finished:
        # Finally, melt snow if nexessary
        u = esnow.copy()
        if -u*dhs >= ebot:
            delhs += ebot/u
            ebot = 0.
            finished = True
        else:
            print('melted completely through all ice and snow')
            print('ERROR in botmelt')

    return delb, delh, delhs, alarm


def adjust(egrow, delb, delh):
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

    global tiny, hice, n1, eice

    e_tw = eice.copy()

    if not (np.abs(delb < tiny) and (delh > -tiny)):
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

            layers = np.arange(2, n1+1)
            z[layers-1] = delta*(layers-1)
            z_tw[layers-1] = z_tw[0] + delta_tw*(layers-1)

            z[n1] = hice
            z[n1+1] = hice + np.maximum(delb, 0.)
            z_tw[n1] = z_tw[0] + h_tw

            fract = np.zeros((n1, n1+1))

            for l_tw in range(n1):
                for l in range(n1+1):
                    fract[l_tw, l] = np.minimum(z_tw[l_tw], z[l]) - \
                         np.maximum(z_tw[l_tw-1], z[l-1])

            fract = fract/delta_tw
            fract = np.maximum(fract, 0)

            e_tw = np.concatenate((eice, egrow), axis=1) @ fract.T

    return e_tw
