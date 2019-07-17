#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 15:03:48 2019

@author: andrewpauling
"""
import numpy as np


def sumall(hice, hsnow, eice, esnow, n1):

    energall = np.sum(eice) * hice/n1 + esnow * hsnow

    return energall
