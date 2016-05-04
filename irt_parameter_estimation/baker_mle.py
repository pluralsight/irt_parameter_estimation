#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2016 Pluralsight, LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.


'''Implementations of various Maximum Likelihood Estimation methods

The various methods below fit a logistic functions to the binned
response percentiles for each questions.
The input data is for a single question and consists of theta, r, and f;
  theta: a regularly spaced set of ability levels
  r:     the number of correct responses for each ability group in theta
  f:     the total number of responses for each ability group in theta

Percentages (p) are just r / f

a is the discriminatory power (slope):
  larger lambda means more discrimination (closer for a step-function)
b is the difficulty, the location parameter (offset along the x-axis)
c is the pseudo-guesing parameter (lower asymptotote)
d is the upper asymptote (not used here)

lamdba (or lam) is the same as a
zeta is the adjusted difficulty, -a * b

Code is based on work by Frank B. Baker and Seock-Ho Kim:
Item Response Theory: Parameter Estimation Techniques
http://www.crcpress.com/product/isbn/9780824758257

The original BASIC code that this was ported from can be downloaded here:
http://www.crcpress.com/downloads/DK2939/IRTPET.zip

Some more documentation:
Main equation for partial derivative of log-likelihood dL / dx
\frac{\partial L}{\partial x} =
\sum \frac{1}{P}\frac{\partial P}{\partial x} -
\sum (f - r)}\frac{1}{1 - P}\frac{\partial P}{\partial x}
All these algorithms minimize L and look for zeros in dL / dx

The differences in the methods have to do with how P is defined

This version is for comparative purposes only, use zlc_mle.py instead.
'''

from __future__ import division
import numpy as np
import matplotlib.pyplot as plt

from util import (ConvergenceError, dev_zeta_lam, dev_ab, logistic, scale_guessing,
                  logistic3PLabc as logistic3PL,
                  chi_squared as chiSquared)

#MAX_ITERATIONS = 10000
W_CUTOFF = 0.0000009
DM_CUTOFF = 0.000099
MAX_ZETA = 30
MAX_LAM = 20
MAX_B = 500

def mle_1_parameter(theta, r, f, a, b, MAX_ITERATIONS=10000):
    '''One parameter (Rausch) logistic ICC model parameter estimation using MLE
       From "Item Response Theory: Parameter Estimation Techniques"
       Chapter 2.3, pages 40-45 (esp Eqn 2.20) and
       Appendix A, pages 289-297 (port of A.3 on p.294)'''
    theta, r, f = map(np.asanyarray, [theta, r, f]) # ensure these are arrays
    p = r / f

    for i in range(MAX_ITERATIONS):
        print "iteration", i
        P = np.squeeze(logistic(dev_ab(a, b, theta)))
        W = P * (1 - P)
        W[np.where(W<W_CUTOFF)] = np.nan # Delete any dud values

        dLdb = np.nansum(r - f * P)
        d2Ldb2 = -np.nansum(f * W)

        print 'dLdb', dLdb, 'd2L / db2', d2Ldb2

        # Plot the Log Likelihood & 1st&2nd derivatives
        '''
        import plt
        bs = np.arange(-10, 10, 0.1)
        Ps = [ logistic(dev_ab(a, b, theta))
              for b in bs ]
        L  = [ log_likelihood(theta, r, f, P)
               for P in Ps ]
        dLdb = [ np.nansum(r - f * P)
                for P in Ps ]
        d2Ldb2 = [ np.nansum(f * P * (1 - P))
                  for P in Ps ]
        plt.ioff()
        plt.plot(bs, L)
        plt.plot(bs, dLdb)
        plt.plot(bs, d2Ldb2)
        plt.show()
        '''

        # Db = rhs * np.linalg.inv(mat) # rhs = np.nansum(f * W * V), mat = np.nansum(f * W) V = (p - P) / W
        Db = dLdb / d2Ldb2

        b += Db

        print 'b', b, 'Db', Db

        if abs(b)>MAX_ZETA:
            raise ConvergenceError("OUT OF BOUNDS ERROR ITERATION %i" % i)

        if abs(Db) <= .05:
            break

    if i == MAX_ITERATIONS:
        print "REACHED MAXIMUM NUMBER OF ITERATIONS"

    P = logistic(dev_ab(a, b, theta))
    chi2 = chiSquared(f, p, P)
    return a, b, chi2

def mle_2_parameter(theta, r, f, zeta, lam, MAX_ITERATIONS=10000):
    '''Two parameter logistic ICC model parameter estimation using MLE
       From "Item Response Theory: Parameter Estimation Techniques"
       Chapter 2.3, pages 40-45 (esp Eqn 2.20) and
       Appendix A, pages 289-297 (port of A.3 on p.294)'''
    theta, r, f = map(np.asanyarray, [theta, r, f]) # ensure these are arrays
    p = r / f

    for i in range(MAX_ITERATIONS):
        print "iteration", i
        P = np.squeeze(logistic(dev_zeta_lam(zeta, lam, theta)))
        W = P * (1 - P)
        W[np.where(W<W_CUTOFF)] = np.nan # Delete any dud values
        V = (p - P) / W
        fW = f * W; fWV = fW * V; fWV2 = fWV * V; fWth = fW * theta; fWth2 = fWth * theta; fWthV = fWth * V

        if np.nansum(fW) <= 0:
            raise ConvergenceError("OUT OF BOUNDS ERROR ITERATION %i" % i)

        mat = np.nansum([[fW, fWth],
                         [fWth, fWth2]], axis=-1)
        rhs = np.nansum([[fWV],
                         [fWthV]], axis=-1)

        mat_det = np.linalg.det(mat)

        if mat_det == 0:
            raise ConvergenceError("Non-invertible matrix encountered %i" % i)

        Dzeta, Dlam = np.dot( np.linalg.inv(mat), rhs ).flatten()

        if mat_det <= DM_CUTOFF:
            break

        zeta += Dzeta
        lam += Dlam

        if abs(zeta)>MAX_ZETA or abs(lam)>MAX_LAM:
            raise ConvergenceError("OUT OF BOUNDS ERROR ITERATION %i" % i)

        if abs(Dzeta) <= .05 and abs(Dlam) <= .05:
            break

    if i == MAX_ITERATIONS:
        print "REACHED MAXIMUM NUMBER OF ITERATIONS"

    #chi2 = np.nansum(fWV2) # sum of f[i] * W[i] * v[i]^2
    P = logistic(dev_zeta_lam(zeta, lam, theta))
    chi2 = chiSquared(f, p, P)
    return zeta, lam, chi2

def mle_3_parameter(theta, r, f, a, b, c, MAX_ITERATIONS=10000):
    '''Three parameter logistic ICC model parameter estimation using MLE
       From "Item Response Theory: Parameter Estimation Techniques"
       Chapter 2.6, pages 48-57 (esp final eqn on p.55)

       For comparison,
       a = lambda
       -a * b = zeta

       (structure based on the 2PL function above)


       Proof that Qstar = 1 - Pstar:
       Q / Qstar = (1 - c) ()
       Q = (1 - c) * Qstar
       P = c + (1 - c) * Pstar
       Q = 1 - P = 1 - c - (1 - c) * Pstar
       So, 1 - c - (1 - c) * Pstar = (1 - c) * Qstar
       Qstar = ((1 - c) - (1 - c) * Pstar) / (1 - c) = 1 - Pstar'''
    theta, r, f = map(np.asanyarray, [theta, r, f]) # ensure these are arrays
    p = r / f

    for i in range(MAX_ITERATIONS):
        print "iteration", i
        aa = a * np.ones(f.shape) # array version of a
        Pstar = logistic(dev_ab(a, b, theta))
        P = np.squeeze(scale_guessing(Pstar, c))
        pmP = p - P
        iPc = 1 / (P - c)
        Q = (1 - P)
        Qic = Q / (1 - c)
        rat = Pstar / P
        thmb = theta - b
        #W = Pstar * (1 - Pstar) / (P * (1 - P))
        #W[np.where(W<W_CUTOFF)] = np.nan # Delete any dud values
        #if np.any(np.nansum(f * W)<0):
        #    raise ConvergenceError("OUT OF BOUNDS ERROR ITERATION %i" % i)

        # LL = np.sum(r * np.log(P)) + np.sum((f - r) * np.log(1 - P)) # Compute the log-likelihood
        L1, L2, L3 = np.array([thmb, -aa, iPc]) * f * pmP * rat # This is right
        JLL = np.nansum([L1, L2, L3], axis = -1)
        rhs = np.nansum([[L1], [L2], [L3]], axis = -1)

        EL11, EL22, EL33 = np.array([-P * Q * thmb**2 * rat,
                                     -a**2 * P * Q * rat,
                                     Qic * iPc]) * f * rat
        EL12, EL13, EL23 = np.array([a * thmb * P * Q * rat,
                                     -thmb * Qic,
                                     a * Qic]) * f * rat

        # This was wrong, but somehow seemed to work just as well??
        #EL11, EL22, EL33 = np.array([-P * Q * thmb ** 2, -a**2 * P * Q, Qic * iPc]) * f * rat**2
        #EL12, EL13, EL23 = np.array([a * thmb * P * Q * rat, -thmb * Qic, a * Qic]) * f * rat

        mat = JJLL = np.nansum([[EL11, EL12, EL13],
                                [EL12, EL22, EL23],
                                [EL13, EL23, EL33]], axis = -1)

        Da, Db, Dc = np.dot( np.linalg.inv(mat), rhs ).flatten()

        if np.linalg.det(mat) <= DM_CUTOFF:
            break

        a += Da
        b += Db
        c += Dc

        if abs(a)>MAX_LAM or abs(b)>MAX_B or abs(c)>1:
            raise ConvergenceError("OUT OF BOUNDS ERROR ITERATION %i" % i)

        if abs(Da) <= .05 and abs(Db) <= .05 and abs(Dc) <= .05:
            break

    if i == MAX_ITERATIONS:
        print "REACHED MAXIMUM NUMBER OF ITERATIONS"

    P = logistic3PL(a, b, c, theta)
    chi2 = chiSquared(f, p, P)
    return a, b, c, chi2
