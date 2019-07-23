#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 18:03:54 2019

@author: andrewpauling
"""

import numpy as np
import time
from pyBL99.utils.tstmnew import getabc as getabc1
from pyBL99.fortran.getabc_lib import getabc as getabc2

n1 = 10
ti = np.random.rand(n1+1)
tbot = 0.
zeta = np.random.rand(n1+1)
delta = np.random.rand(n1)
ki = np.random.rand(n1+2)
eta = np.random.rand(n1+1)
lfirst = 2

start = time.time()
a1, b1, c1, r1 = getabc1(ti, tbot, zeta, delta, ki, eta, n1, lfirst)
end = time.time()
print('time 1 = '+str(end-start))

start = time.time()
a2, b2, c2, r2 = getabc2(ti, tbot, zeta, delta, ki, eta, n1, lfirst)
end = time.time()
print('time 2 = '+str(end-start))