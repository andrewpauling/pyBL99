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


def dh(fneti, ultnt, condb, rflsno, rcpsno, tsmelt, n1, nday, dtau, tiny,
       hsmin, hsstar, frzpt, fw, hice, hsice, hsnow, tbot, ts, tice, eice,
       esnow, saltz, rflice, rcpice, alpha, gamma):
    """
    Compute height change
    """

    alarm = False
    alarm2 = False

    esnow = snownrg(rflsno, rcpsno, hsnow, tice, tsmelt)
    eice = energ(tice[1:n1+1], saltz[1:n1+1])
    enet = sumall(hice, hsnow, eice, esnow, n1)

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
            egrow = energ(tbot, saltz[n1+1], rflice, rcpice, alpha, gamma)
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
                delb, delh, delhs, alarm2 = botmelt(ebot, dhs, dhi, delh, delhs, delb)
                if alarm2:
                    print('problem with botmelt')
                    alarm = True
        if not alarm:
            eice = adjust(egrow, delb, delht)

    return delb, delhs, delh, subi, subs, alarm


def surfsub(enet, etop, dhs, dh, delh, delhs, rvlsno, rvlice, eice, esnow, n1,
            es, ei):
    """
    Compute subsurface melting
    """

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
            
            