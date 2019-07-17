#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 12:07:43 2019

@author: andrewpauling
"""

import numpy as np
from attrdict import AttrDict

from pyBL99.utils import constants as const
from pyBL99.utils.salinity_prof import salinity_prof
from pyBL99.physics.energ import energ
from pyBL99.physics.snownrg import snownrg


def initial_state(nlayers):

    saltz = salinity_prof(nlayers)         # Generate initial salinity profile

    tw = const.frzpt                       # Water temperature

    tbot = np.minimum(tw, const.tmelt)     # Temperature at bottom of ice

    hsnow = 0.2827*const.centi             # Initial snow depth
    hice = 2.53*const.centi                # initial ice thickness

    # Initialize temperature, one more than nlayers due to snow
    tice = np.zeros(nlayers+1)

    if nlayers == 10:
        ts = -29.2894
        tice[0] = -23.1128
        tice[1] = -15.7925
        tice[2] = -14.1042
        tice[3] = -12.4534
        tice[4] = -10.8423
        tice[5] = -9.2777
        tice[6] = -7.7687
        tice[7] = -6.3255
        tice[8] = -4.9605
        tice[9] = -3.6879
        tice[10] = -2.5230
    else:  # Don't use this yet
        tice[0] = -23.16-(const.tfrez+23.16)/nlayers
        for layer in range(1, nlayers+1):
            tice[layer] = -23.16+(layer-1) * (const.tfrez+23.16)/nlayers

    esnow = snownrg(hsnow, tice)                # initial snow energy
    layers = np.arange(1, nlayers+1)
    eice = energ(tice[layers], saltz[layers])   # initial ice energy

    state = AttrDict()

    state['nlayers'] = nlayers
    state['hice'] = hice
    state['hsnow'] = hsnow
    state['tice'] = tice
    state['ts'] = ts
    state['tw'] = tw
    state['saltz'] = saltz
    state['tbot'] = tbot
    state['esnow'] = esnow
    state['eice'] = eice
    state['io_surf'] = 0.3

    return state


def internal_state():
    # declare vars for this routine
    firststep = True
    fsh_n1 = 0
    flo_n1 = 0
    dnsens_n1 = 0
    dnltnt_n1 = 0
    mualbedo_n1 = 0
    fsh_n = 0
    fsh = 0
    flo_n = 0
    flo = 0
    dnsens_n = 0
    upsens = 0
    dnltnt_n = 0
    upltnt = 0
    mualbedo_n = 0
    mualbedo = 0
    e_init = 0  # energy in the ice and snow
    e_end = 0
    heat_added = 0  # running total of heat added to the ice and snow

    state = AttrDict()

    state['firststep'] = firststep
    state['fsh_n1'] = fsh_n1
    state['fsh_n'] = fsh_n
    state['fsh'] = fsh
    state['flo_n1'] = flo_n1
    state['flo_n'] = flo_n
    state['flo'] = flo
    state['dnsens_n1'] = dnsens_n1
    state['dnsens_n'] = dnsens_n
    state['upsens'] = upsens
    state['dnltnt_n1'] = dnltnt_n1
    state['dnltnt'] = dnltnt_n
    state['upltnt'] = upltnt
    state['mualbedo_n1'] = mualbedo_n1
    state['mualbedo_n'] = mualbedo_n
    state['mualbedo'] = mualbedo
    state['e_init'] = e_init
    state['e_end'] = e_end
    state['heat_added'] = heat_added

    return state
