#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 14:45:10 2019

@author: andrewpauling
"""


def calc_albedo():

    global hsmin, hsnow, ts, tmelt

    pert = 0.0
    albedo = 0.63 + pert

    if hsnow > hsmin:
        albedo = 0.8 + pert
        if ts >= (tmelt-0.01):
            albedo = 0.75 + pert

    return albedo
