#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 10:53:15 2019

@author: andrewpauling
"""


def snowfall(idx):
    """
    snowfall from maykut and untersteiner 1971
    """

    snow = 0

    if idx <= 118 or idx >= 301:
        snow = 2.79e-4
    elif idx >= 119 and idx <= 149:
        snow = 1.61e-3
    elif idx > 229:
        snow = 4.16e-3
    else:
        snow = 0.0

    return snow
