#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 15:00:52 2019

@author: andrewpauling
"""
import constants as const


def snownrg(hsnow, tice):

    nrg = -const.rflsno + const.rcpsno * (tice[0]-const.tsmelt)

    return nrg
