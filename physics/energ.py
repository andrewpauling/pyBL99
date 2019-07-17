#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 15:08:40 2019

@author: andrewpauling
"""
import pyBL99.utils.constants as const


def energ(tmp, sal):
    """
    Compute the specific enthalpy for non-new ice relative to melting
    (negative) quantity) assuning tmp is in celsius
    """

    nrg = -const.rflice - const.rcpice*(-const.alpha*sal-tmp) - \
        const.gamma*sal/tmp

    return nrg
