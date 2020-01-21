#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 15:00:52 2019

@author: andrewpauling
"""
import pyBL99.utils.constants as const


def snownrg(hsnow, tice):
    """
    Compute specific enthalpy of snow

    Parameters
    ----------
    hsnow : float
        snow depth
    tice : ndarray
        Ice temperature profile

    Returns
    -------
    nrg : float
        specific enthalpy of ice

    """

    nrg = -const.rflsno + const.rcpsno * (tice[0]-const.tsmelt)

    return nrg
