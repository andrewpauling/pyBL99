#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 11:07:40 2019

@author: andrewpauling
"""

import numpy as np
from pyBL99.fortran.iter_lib import iter_ts_mu as iter_ts_mu1
from pyBL99.utils.tstmnew import iter_ts_mu as iter_ts_mu2

k = 1
fofix = 2
condfix = 3

ts1 = iter_ts_mu1(k, fofix, condfix)
ts2 = iter_ts_mu2(k, fofix, condfix)