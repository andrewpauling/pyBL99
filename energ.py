#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 15:08:40 2019

@author: andrewpauling
"""


def energ(tmp, sal):
    """
    Compute the specific enthalpy for non-new ice relative to melting
    (negative) quantity) assuning tmp is in celsius
    """

    global rflice, rcpice, alpha, gamma

    nrg = -rflice - rcpice*(-alpha*sal-tmp) - gamma*sal/tmp

    return nrg
