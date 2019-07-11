#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 17:33:08 2019

@author: andrewpauling
"""

import numpy as np


def salinity_prof(n):
    """
    Define ice salinity profile
    """

    saltmax = 3.2

    saltz = np.zeros(n+2)

    for layer in range(1, n+1):
        zrel = (layer-0.5)/n
        saltz[layer+1] = saltmax/2 * \
            (1+np.sin(np.pi*(zrel**(0.40706205/(zrel+0.57265966))-0.5)))

    saltz[0] = 0     # snow layer salinity, not used
    saltz[n+1] = saltmax  # base of the ice, probably not used

    return saltz
