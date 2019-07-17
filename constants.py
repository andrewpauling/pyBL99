#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 18:51:58 2019

@author: andrewpauling
"""

from numpy import exp

sigma = 5.67e-5          # Stefan constant
esice = sigma            # ice emissivity times Stefan constant

rhoice = 917e-3          # density of ice
rhosno = 330e-3          # density of snow

cpice = 2054e4           # heat capacity of fresh ice
cpsno = 2113e4           # heat capacity of snow

vlocn = 2.501e10         # latent heat of vaporization freshwater
sbice = 2.835e10         # latent heat of sublimation freshwater
flice = sbice - vlocn    # latent heat of fusion freshwater

rflice = flice*rhoice    # specific latent heat of fusion ice
rflsno = flice*rhosno    # specific latent heat of fusion snow
rslice = sbice*rhoice    # specific latent heat of sublim ice
rslsno = sbice*rhosno    # specific latent heat of sublim snow
rcpice = cpice*rhoice    # specific heat capacity of fresh ice
rcpsno = cpsno*rhosno    # specific heat capacity of snow
tffresh = 273.16         # Freezing temperature of fresh water
tmelt = 0.
tsmelt = 0.              # Melting temperature of snow
tfrez = 271.2 - tffresh  # Approx freezing temp of the ocean

# coefficients for computing saturation vapor pressure
ai = 21.8746             # over ice
bi = 7.66-273.16         # over ice
qs1 = 0.622*6.11/1013    # mol. weight of water:dry air/surf. press (mb)
qsmelt = qs1*exp(ai*tsmelt/(tsmelt-bi))
qmelt = qs1*exp(ai*tmelt/(tmelt-bi))

# parameter for computing melting temp as a function of salinity
# i.e., melting = Tffresh - alpha*salinity
alpha = 0.054

kappa = 1.5e-2           # Solar extinction coefficient in sea ice
kimin = 0.1e5            # Minimum conductivity in ice
beta = 0.1172e5          # Conductivity K=Kappai+Beta*Salinity/T
gamma = rflice*alpha     # Heat capacity C=Cpi+Gamma*Salinity/T**2

# for thin ice growth in the lead want to use predetermined values for
# kappai, based on estimate for -5 deg C surface temp
kappai = 2.034e5         # thermal conductivity of fresh ice
kappas = 0.31e5          # thermal conduictivity of snow

tiny = 1e-6

# if there is not at least a centimeter of snow do not bother with
# conductive effects of snow in the heat equation
# do not rub out though, give it a chance to accumulate
# also allow it to melt
# albedo depends on snow depth so small amounts of snow hanging around
# will not screw up the albedo
# WHEN THE SNOW IS < HSMIN ITS TEMPERATURE IS EQUAL TO THE SURFACE
hsmin = 1.
hsstar = 0.0005

# first guess at initial hice and area and set misc. variables to 0;
# ice is uniformly set to 2.8 m with no open water;

centi = 100.
kilo = 1000.
frzpt = tfrez
fw = 2.0*kilo
area = 1.

