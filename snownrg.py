#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 15:00:52 2019

@author: andrewpauling
"""


def snownrg(rflsno, rcpsno, hsnow, tice, tsmelt):

    nrg = -rflsno + rcpsno * (tice[0]-tsmelt)

    return nrg
