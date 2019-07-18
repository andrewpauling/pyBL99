#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 14:22:46 2019

@author: andrewpauling
"""
import numpy as np
import pyBL99.utils.constants as const


def tstmnew(state, internal_state, io, dswr, dtau):
    """
    This function solves the evolution of the ice interior and surface
    temperature from the heat equation and surface energy balance

    It solves the heat equation which is non-linear for saline ice;
    using an iterative newt-raps multi-equation solution.;
    scheme is either backwards or crank-nicholson, giving a tridiagonal;
    set of equations implicit in ti;
    Default differencing is backwards (CN seems to work, but the model;
    does not conserve heat when melting);

    must have a sufficient amount of snow to solve heat equation in snow;
    hsmin is the minimum depth of snow in order to solve for tice(0+1);
    if snow thickness < hsmin then do not change tice(0+1);

    the number of equations that must be solved by the tridiagonal solver;
    depends on whether the surface is melting and whether there is snow.;
    four soln_types are possible(see variable soln_type below):;
    1 = freezing w./ snow, 2 = freezing w./ no snow,;
    3 = melting w./ snow, and 4 = melting w./ no snow;

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    The call list I/O

    vars sent
    tw;                  % water temp below ice
    flo;                 % long wave down at surface
    io; ib;              % solar penetrating top and bottom surfaces
    dswr;                % above surface net downward shortwave, indep of ts
    ultnt; usens;        % net upward latent, sensible
                         % in this implementation with MU forcing, these are
                         % independent of temperature

    vars sent and returned
    heat_added;          % diagnostic to keep track of work done on ice
                           from atm.
    """

    n1 = state.nlayers

    # vars returned
    f = 0                # net flux at top surface including conductive flux
    # in ice and snow
    cond = 0             # conductive flux at top surface
    condb = 0            # conductive flux at bottom surface
    dq1 = 0              # enthalpy change of top
    io1 = 0              # solar absorbed in top layer
    ulwr = 0             # net upward longwave flux

    # local variables
    ts_old = 0
    ti_old = np.zeros(n1+1)
    dti = np.zeros(n1+2)  # incremental changes to ts and ti from N-R
    ts_kelv = 0           # surface temp in Kelvin
    z = np.zeros(n1+1)    # vertical coordinate of layer faces
    qice = 0              # vapor at ice/snow surface
    ulwr_init = 0
    ulwr_melt = 0

    isol = np.zeros(n1+1)  # solar at layer interfaces
    iabs = np.zeros(n1)    # solar absorbed in each layer
    absorb = 0             # sum of iabs
    fo = 0                 # net flux at top surf excluding conductive flux
    f_init = 0
    fo_init = 0
    fo_melt = 0
    dfo_dt = 0             # derivative of Fo wrt temperature
    fofix = 0              # terms in Fo independent of temp
    iru = 0                # dummies used to compute F and Fo
    condfix = 0            # terms in cond that are independent of temp
    cond_init = 0
    cond_melt = 0
    specialk = 0           # modified conductivity for top layer of ice/snow
    melts = 0              # surface melting temp
    alph = 0               # parameters for maintaining 2nd order accurate diff
    bet = 0

    # a, b, c are vectors that describe the diagonal and off-diagonal elements
    # of the matrix [a] such that [a] ti = r
    a = np.zeros(n1+2)
    b = np.zeros(n1+2)
    c = np.zeros(n1+2)
    d = np.zeros(3)
    r = np.zeros(n1+2)
    ki = np.zeros(n1+2)     # layer confuctivity divided by layer thickness
    zeta = np.zeros(n1+1)   # terms in heat eqn independent of ti
    delta = np.zeros(n1)    # gamm*Salinity/fresh ice heat capacity
    eta = np.zeros(n1+1)    # time step/ice layer thickness/f. i. h. c.
    cpi = np.zeros(n1)      #
    dh = 0                  # ice layer thickness
    dt_dh = 0               # time tsep/ice thickness
    dt_hs = 0               # time step/snow thickness
    n = 0                   # number of eqns solved by tridiag solver
    n1p2 = n1+2
    layers = np.arange(n1)  # counters for ice layers
    layers0 = np.arange(-1, n1)      # counters for snoe and ice layers
    iloop = 0               # counter for iterations of numerical loop
    niter = 0               # counter for number of times try to find soln

    keepiterating = True    # flag indicating N-R has gone astray
    errit = 0               # abs value of maximum dti
    errmax = 1e-4           # max error allowed for dti in degrees

    theta = 1.              # 1 for backwards and 0.5 for CN
    soln_type = 0           # indicates the case of snow vs no snow and melting
    # vs freezinhg

    # Setup helpful parameters
    dh = state.hice/n1
    dt_dh = dtau/dh
    dt_hs = 0

    if state.hsnow > const.hsmin:
        dt_hs = dtau/state.hsnow

    ts_old = state['ts']
    ti_old = state['tice']
    state.tbot = np.minimum(state.tw, const.tmelt)
    ki = conductiv(state.tice, state.tbot, state.hsnow, state.saltz, dh, n1)

    # solar radiation absorbed internally
    z = np.cumsum(np.insert(dh*np.ones(n1), 0, 0))
    isol = np.exp(-const.kappa*z)
    iabs = io*(isol[layers] - isol[1+layers])

    if state.hsnow > const.hsmin:
        alph = 2*(2*state.hsnow + dh)/(state.hsnow + dh)
        bet = -2*state.hsnow*state.hsnow/(2*state.hsnow + dh) / \
            (state.hsnow + dh)
        condfix = ki[0]*(alph*state.tice[0]+bet*state.tice[1])
        specialk = ki[0]*(alph + bet)
        melts = const.tsmelt
        ulwr_melt = const.esice*(const.tsmelt+const.tffresh)**4 - \
            internal_state.flo
        qice = const.qsmelt
    else:
        alph = 3.
        bet = -1/3
        specialk = ki[1]*8/3
        condfix = ki[1]*(alph*state.tice[1]+bet*state.tice[2])
        melts = const.tmelt
        ulwr_melt = const.esice*(const.tsmelt+const.tffresh)**4 - \
            internal_state.flo
        qice = const.qmelt

    # -----------------------------------------------------------------------;
    # get a first guess for ts based on ti from previous step;
    # assume the surface temperature is at the melting point and;
    # see if it is balanced. note that this alters ts_old;
    # as far as the cn method goes;
    # this could be debated - though it doesn't affect energy conservation.;
    # -----------------------------------------------------------------------

    cond_melt = condfix - specialk*melts
    fofix = dswr - io + internal_state.flo - internal_state.upltnt - \
        internal_state.upsens
    fo_melt = dswr - io - ulwr_melt - internal_state.upltnt - \
        internal_state.upsens
    f = fo_melt + cond_melt

    if f < 0.:
        f = 0.
        state.ts = iter_ts_mu(specialk, fofix, condfix)
    else:
        ulwr_init = ulwr_melt

    condb_init = ki[n1+1]*(3*(state.tbot-state.tice[n1]) -
                           (state.tbot-state.tice[n1-1])/3)
    f_init = f
    ts_old = state.ts

    # BEGINNING OF ITERATIVE PROCEDURE

    niter = 1
    iloop = 1
    errit = 10
    keepiterating = True

    while keepiterating:
        while (errit > errmax) and iloop < 20:
            # setup terms that depend on the cpi and
            # initial temp(ti_old and ts_old)

            cpi[layers] = const.rcpice + \
                const.gamma*state.saltz[layers+1]/state.tice[layers+1] / \
                ti_old[layers+1]
            eta[0] = 0

            if state.hsnow > const.hsmin:
                eta[0] = dtau/(state.hsnow*const.rcpsno)

            eta[layers+1] = dt_dh/cpi[layers]

            if theta < 1:
                pass
            else:
                zeta = ti_old
                zeta[layers+1] = zeta[1+layers] + eta[1+layers]*iabs

            f = fo_melt + cond_melt
            if f <= 0:
                ts_kelv = state.ts + const.tffresh
                iru = const.esice*ts_kelv**4
                dfo_dt = -4*const.esice*ts_kelv**3
                fo = fofix - iru
                if state.hsnow > const.hsmin:
                    soln_type = 1

                    a, b, c, r = getabc(state.tice, state.tbot, zeta, delta,
                                        ki, eta, n1, 1)
                    cond = ki[0]*(alph*(state.tice[0]-state.ts) +
                                  bet*(state.tice[1]-state.ts))
                    a[1] = -eta[0]*ki[0]*(alph+bet)
                    c[1] = eta[0]*(bet*ki[0]-ki[1])
                    b[1] = 1+eta[0]*(alph*ki[0]+ki[1])
                    r[1] = -zeta[0] + a[1]*state.ts + b[1]*state.tice[0] + \
                        c[1]*state.tice[1]
                    a[0] = 0
                    c[0] = -ki[0]*alph
                    d[0] = -ki[0]*bet
                    b[0] = -dfo_dt-c[0]-d[0]
                    r[0] = -fo-cond

                    b[0] = c[1]*b[0]-d[1]*a[1]
                    c[0] = c[1]*c[0]-d[0]*b[1]
                    r[0] = c[1]*r[0]-d[0]*r[1]

                    n = np.floor(n1+2)
                    dti = tridag(a, b, c, r, n, n1)
                    state.ts -= dti[0]
                    state.tice[layers0] = state.tice[layers0] - dti[layers0+1]
                else:
                    soln_type = 2

                    a, b, c, r = getabc(state.tice, state.tbot, zeta, delta,
                                        ki, eta, n1, 2)
                    a[2] = -eta[1]*ki[1]*(alph+bet)
                    c[2] = -eta[1]*ki[2]-bet*ki[1]
                    b[2] = 1 + eta[1]*ki[2]+alph*ki[1]
                    r[2] = -zeta[1] + a[2]*state.ts + b[2]*state.tice[1] + \
                        c[2]*state.tice[2]
                    a[1] = 0
                    c[1] = -ki[1]*alph
                    d[1] = -ki[1]*bet
                    b[1] = -dfo_dt-c[1]-d[1]
                    r[1] = -fo - ki[1]*(alph*(state.tice[1]-state.ts) +
                                        bet*(state.tice[2]-state.ts))

                    b[1] = c[2]*b[1]-d[1]*a[2]
                    c[1] = c[2]*c[1]-d[1]*b[2]
                    r[1] = c[2]*r[1]-d[1]*r[2]
                    n = np.floor(n1+1)
                    dti[1:n1p2] = tridag(a[1:n1p2], b[1:n1p2], c[1:n1p2],
                                         r[1:n1p2], n, n1)
                    state.ts -= dti[1]
                    state.tice[layers+1] = state.tice[layers+1]-dti[layers+2]

            else:
                state.ts = melts

                if state.hsnow > const.hsmin:
                    soln_type = 3

                    a, b, c, r = getabc(state.tice, state.tbot, zeta, delta,
                                        ki, eta, n1, 1)
                    a[1] = -eta[0]*ki[0]*(alph+bet)
                    c[1] = eta[0]*(bet*ki[0]-ki[1])
                    b[1] = 1+eta[0]*(alph*ki[0]+ki[1])
                    r[1] = -zeta[0] + a[1]*state.ts + b[1]*state.tice[0] + \
                        c[1]*state.tice[1]
                    a[1] = 0

                    n = np.floor(n1+1)
                    dti[1:n1p2] - tridag(a[1:n1p2], b[1:n1p2], c[1:n1p2],
                                         r[1:n1p2], n, n1)
                    state.tice[layers0+1] = state.tice[layers0+1] - \
                        dti[layers0+2]

                else:
                    soln_type = 4

                    a, b, c, r = getabc(state.tice, state.tbot, zeta, delta,
                                        ki, eta, n1, 2)
                    a[2] = -eta[1]*ki[1]*(alph+bet)
                    c[2] = -eta[1]*(ki[2]-bet*ki[1])
                    b[2] = 1+eta[1]*(ki[2]+alph*ki[1])
                    r[2] = -zeta[1] + a[2]*state.ts + b[2]*state.tice[1] + \
                        c[2]*state.tice[2]
                    a[2] = 0
                    n = np.floor(n1)
                    dti[2:n1p2] = tridag(a[2:n1p2], b[2:n1p2], c[2:n1p2],
                                         r[2:n1p2], n, n1)
                    state.tice[layers+1] = state.tice[layers+1] - dti[layers+2]
            
            errit = 0.
            if state.hsnow > const.hsmin:
                errit = np.abs(dti[1])
            errit = np.max(np.append(np.abs(dti[layers+2]), errit))
            state.ts = np.minimum(state.ts, melts)

            if state.hsnow > const.hsmin:
                condfix = ki[0]*(alph*state.tice[0] + bet*state.tice[1])
                cond_melt = condfix - specialk*const.tsmelt
            else:
                condfix = ki[1]*(alph*state.tice[1] + bet*state.tice[2])
                cond_melt = condfix - specialk*const.tmelt

            iloop = np.floor(iloop + 1)

        keepiterating = False

        if np.logical_or(state.tice[0] > (const.tsmelt+const.tiny),
                         np.any(state.tice[layers+1] >
                         -const.alpha*state.saltz[layers+1])):
            keepiterating = True

            if niter == 1:
                # this worksa 99 times out of 100
                state.ts = ts_old - 20
                state.tice[layers+1] = ti_old[layers+1] - 20
            elif niter == 2:
                # this works the rest of the time
                state.ts = melts - 0.5
                state.tice[layers+1] = const.tmelt - 0.5
            else:
                # when this error occurs, the model does not conserve energy;
                # it should still run, but there probably is something wrong;
                # that is causing this sceme to fail;
                keepiterating = False  # give up on finding a solution
                print('WARNING converges to ti>TMELT')
                print(ts_old)
                for l in range(n1):
                    print('ti_old =' + ti_old[l+1])
                print('tbot =' + state.tbot)
                print('dswr =' + dswr)
                print('flo =' + internal_state.flo)
                print('io =' + io)
                state.ts = melts
                state.tice[layers0+1] = np.minimum(
                    -const.alpha*state.saltz[layers0+1],
                    state.tice[layers0+1])

            if state.hsnow > const.hsmin:
                condfix = ki[0]*(alph*state.tice[0] + bet*state.tice[1])
                cond_melt = condfix - specialk*const.tsmelt
            else:
                condfix = ki[1]*(alph*state.tice[1]+bet*state.tice[2])
                cond_melt = condfix - specialk*const.tmelt

        niter = np.floor(niter+1)
        iloop = 0  # reset numerical loop counter

    # ------------------------------------------------------------------------
    # continue from here when done iterative
    # finish up by updating the surface fluxes, etc.;
    # ------------------------------------------------------------------------
    if state.ts < melts:
        cond = condfix-specialk*state.ts
        fo = -cond
        f = 0.0
        ulwr = const.esice*(state.ts+const.tffresh)**4 - internal_state.flo

        if theta < 1:
            ulwr = theta*ulwr + (1.-theta)*ulwr_init
        else:
            # gaurantees the surface is in balance;
            ulwr = cond+dswr-io-internal_state.upltnt-internal_state.upsens

    else:
        fo = fo_melt
        f = fo_melt+cond_melt
        cond = cond_melt

        if theta < 1:
            ulwr = theta*ulwr_melt + (1.-theta)*ulwr_init
        else:
            ulwr = ulwr_melt

    if errit > errmax:
        # when this error occurs, the model does not conserve energy;
        # it should still run, but there probably is something wrong;
        # that is causing this sceme to fail;
        print('WARNING No CONVERGENCE')
        print('errit = ' + errit)
        print('ts = ' + state.ts + 'tice = ' + state.tice)
        print('ts_old = ' + ts_old + 'ti_old = ' + ti_old)
        print('tbot = ' + state.tbot)
        print('dswr = ' + dswr)
        print('flo = ' + internal_state.flo)
        print('io = ' + io)
        state.ts = const.tmelt
        state.tice[layers0+1] = const.tmelt
        f = 0.0
        fo = 0.0
        internal_state.upltnt = 0.0

    # condb is positive if heat flows towards ice;
    condb = ki[n1+1]*(3*(state.tbot-state.tice[n1]) -
                      (state.tbot-state.tice[n1-1])/3.0)
    absorb = np.sum(iabs[:n1])
    ib = io-absorb
    # solar passing through the bottom of the ice
    condb = theta*condb + (1.-theta)*condb_init
    f = theta*f + (1.-theta)*f_init
    fo = theta*fo + (1.-theta)*fo_init
    internal_state.heat_added += (fo+absorb)*dtau
    dq1 = (const.gamma*state.saltz[1]*(1/state.tice[1] - 1/ti_old[1]) +
           const.rcpice*(state.tice[1] - ti_old[1]))/dt_dh
    io1 = iabs[0]

    if state.hsnow > const.hsmin:
        cond = (ki[2]*(state.tice[2]-state.tice[1]) -
                ki[1]*(state.tice[1]-state.tice[0]))
    else:
        cond = ki[2]*(state.tice[2]-state.tice[1]) - cond

    return state, internal_state, f, condb, dq1, io1, ib, cond, ulwr

# -------------------------------------------------------------------------;


def getabc(ti, tbot, zeta, delta, k, eta, ni, lfirst):

    alph = 3.0
    bet = -1.0/3.0
    # if there is snow lfirst=1 otherwise it is 2;
    layers = np.arange(lfirst, ni)

    a = np.zeros(ni+2)
    b = np.zeros(ni+2)
    c = np.zeros(ni+2)
    r = np.zeros(ni+2)

    a[layers+1] = -eta[layers]*k[layers]
    c[layers+1] = -eta[layers]*k[layers+1]
    b[layers+1] = 1 - c[layers+1] - a[layers+1]
    r[layers+1] = -zeta[layers] + a[layers+1]*ti[layers-1] + \
        b[layers+1]*ti[layers] + c[layers+1]*ti[layers+1]

    a[ni+1] = -eta[ni]*(k[ni] - bet*k[ni+1])
    c[ni+1] = 0.0
    b[ni+1] = 1 + eta[ni]*(k[ni] + alph*k[ni+1])
    r[ni+1] = -zeta[ni] + a[ni+1]*ti[ni-1] + b[ni+1]*ti[ni] - \
        eta[ni]*(alph+bet)*k[ni+1]*tbot

    return a, b, c, r

# -------------------------------------------------------------------------;


def tridag(a, b, c, r, n, n1):

    u = np.zeros(int(n))

    bet = 0
    gam = np.zeros(n1+2)
    # if(b(1)==0.) pause 'tridag: rewrite equations';
    bet = b[0]
    u[0] = r[0]/bet

    for layer in range(1, int(n)):
        gam[layer] = c[layer-1]/bet
        bet = b[layer] - a[layer]*gam[layer]
        # if(bet==0.0) pause 'tridag failed';
        u[layer] = (r[layer]-a[layer]*u[layer-1])/bet

    for layer in np.arange(int(n)-2, -1, -1):
        u[layer] = u[layer] - gam[layer+1]*u[layer+1]

    return u

# -------------------------------------------------------------------------;


def conductiv(tice, tbot, hsnow, saltz, dh, ni):

    ki = np.zeros(ni+2)

    tmax = -0.1

    layer = np.arange(1, ni)
    ki[layer+1] = const.kappai + \
        const.beta*(saltz[layer+1] + saltz[layer+2]) / \
        np.minimum(tmax, (tice[layer] + tice[layer+1]))

    ki[ni+1] = const.kappai + const.beta*saltz[ni+1]/tbot
    ki[1] = const.kappai + const.beta*saltz[1]/np.minimum(tmax, tice[1])

    ki = np.maximum(ki, const.kimin)
    ki = ki/dh

    ki[0] = 0.0
    if hsnow > const.hsmin:
        ki[0] = const.kappas/hsnow
        ki[1] = 2.0*ki[1]*ki[0]/(ki[1] + ki[0])

    return ki

# -------------------------------------------------------------------------;


def iter_ts_mu(k, fofix, condfix):

    # first guess
    ts = -20
    niter = 0
    keepiterating = True

    while keepiterating:
        ts_kelv = ts + const.tffresh
        iru = const.esice*ts_kelv**4
        cond = condfix - k*ts
        df_dt = -k-4*const.esice*ts_kelv**3
        f = fofix - iru + cond

        if np.abs(df_dt) < const.tiny:
            keepiterating = False
        else:
            dt = -f/df_dt
            ts += dt
            niter = np.floor(niter+1)
            if niter > 20 or np.abs(dt) > 0.001:
                keepiterating = False

    return ts
