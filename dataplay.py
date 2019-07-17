#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 16:28:58 2019

@author: andrewpauling
"""

import numpy as np

dfile = '/Users/andrewpauling/Documents/PhD/bl99/column/data.mu71'

with open(dfile, 'r') as f:
    x = f.readlines()

with open(dfile, 'r') as f:
    all_lines = [[float(num) for num in line.split()] for line in f]
    
test = np.array(all_lines)
