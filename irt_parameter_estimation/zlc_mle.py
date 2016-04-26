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

Percentages (p) are just r/f
(However, I have now refactored the code to eliminate this division, which
 may help some /0 headaches and should make it easier to extend this to
 possibly more generic methods in the future (perhaps a KDE-based one?))

a is the discriminatory power (slope):
  larger lambda means more discrimination (closer for a step-function)
b is the difficulty, the location parameter (offset along the x-axis)
c is the pseudo-guesing parameter (lower asymptotote)
d is the upper asymptote (not used here)

lamdba (or lam) is the same as a
zeta is the adjusted difficulty, -a * b   (conversely, b = -zeta / lam)

The original equality is:
    zeta + lam * theta == a * (theta - b)

Code is based on work by Frank B. Baker and Seock-Ho Kim:
Item Response Theory: Parameter Estimation Techniques
http://www.crcpress.com/product/isbn/9780824758257

The original BASIC code that this was ported from can be downloaded here:
http://www.crcpress.com/downloads/DK2939/IRTPET.zip

Some more documentation:
Main equation for partial derivative of log-likelihood dL / dx
\\frac{\\partial L}{\\partial x} =
\\sum \\frac{1}{P}\\frac{\\partial P}{\\partial x} -
\\sum (f - r)}\\frac{1}{1 - P}\\frac{\\partial P}{\\partial x}
All these algorithms minimize L and look for zeros in dL / dx

The differences in the methods have to do with how P is defined.

This version uses a hybrid (zeta, lambda, c) formulation which makes the
3PL system much more stable.
Various sub-folders implement other formulations:
abc_mle.py implements the a,b,c formulation
baker_mle.py implements the 2PL model directly from Baker
(has 1-1 matching with published values)

If you know nothing, use this and ignore the other two :)
'''

from __future__ import division
import numpy as np
from scipy.optimize import root #, minimize

from util import (ConvergenceError, dev_zeta_lam, logistic,
                  scale_guessing, logistic3PLzlc, chi_squared, reduced_chi_squared,
                  expand_dims, pack_zlc, unpack_zlc, get_L_zlc)

########################################################################
## Functions to compute 1st and 2nd derivatives of the likelihood
## for the 1, 2, & 3 parameter logistic models
## For consistency, all functions now use: np.nansum( [<>] , axis=1 or 2)
########################################################################

def J_1PL(theta, r, f, P, zeta, lam, c=0, Pstar=None):
    """Get the Jacobian of the log likelihood for the 1PL model
    (1st derivative wrt zeta)

    a here is an ARRAY, not a scalar
    """

    theta, r, f = expand_dims(P, theta, r, f)
    rmfP = r - f * P
    Prat = 1 if Pstar is None else Pstar / P
    L1 = Prat * rmfP
    return np.nansum([L1], axis=1)

def H_1PL(theta, r, f, P, zeta, lam, c=0, Pstar=None):
    """Get the Hessian Matrix of the log likelihood for the 1PL model
    (2nd derivative wrt zeta)
    """

    theta, r, f = expand_dims(P, theta, r, f)
    Pstar = P if Pstar is None else Pstar
    W = Pstar * (1 - Pstar)
    rP2 = r / P**2
    rP2cmf = rP2 * c - f
    EL11 = rP2cmf * W
    return np.nansum([[EL11]], axis=2)


def J_2PL(theta, r, f, P, zeta, lam, c=0, Pstar=None):
    """Get the Jacobian of the log likelihood for the 2PL model
    (1st derivatives wrt zeta, lam)

    a here is an ARRAY, not a scalar
    """

    theta, r, f = expand_dims(P, theta, r, f)
    rmfP = r - f * P
    Prat = 1 if Pstar is None else Pstar / P
    L1, L2 = Prat * rmfP, Prat * rmfP * theta
    return np.nansum([L1, L2], axis=1)

def H_2PL(theta, r, f, P, zeta, lam, c=0, Pstar=None):
    """Get the Hessian Matrix of the log likelihood for the 2PL model
    (2nd derivative wrt zeta, lam)
    """

    theta, r, f = expand_dims(P, theta, r, f)
    Pstar = P if Pstar is None else Pstar
    W = Pstar * (1 - Pstar)
    rP2 = r / P**2
    rP2cmf = rP2 * c - f
    EL11, EL22, EL12 = [rP2cmf * W,
                        rP2cmf * W * theta**2,
                        rP2cmf * W * theta]
    return np.nansum([[EL11, EL12],
                      [EL12, EL22]], axis=2)

def J_3PL(theta, r, f, P, zeta, lam, c, Pstar):
    """Get the Jacobian of the log likelihood for the 3PL model
    (1st derivatives wrt zeta, lam, c)

    a here is an ARRAY, not a scalar
    """

    theta, r, f = expand_dims(P, theta, r, f)
    rmfP = r - f * P
    Prat = 1 if Pstar is None else Pstar / P
    iP1mc = 1 / (P * (1 - c))
    L1, L2, L3 = np.array([Prat, Prat * theta, iP1mc]) * rmfP
    return np.nansum([L1, L2, L3], axis=1)

def H_3PL(theta, r, f, P, zeta, lam, c, Pstar):
    """Get the Hessian Matrix of the log likelihood for the 3PL model
    (2nd derivative wrt zeta, lam, c)
    """

    theta, r, f = expand_dims(P, theta, r, f)
    Pstar = P if Pstar is None else Pstar
    W = Pstar * (1 - Pstar)
    rP2 = r / P**2
    rP2cmf = rP2 * c - f
    EL11, EL22, EL33 = [rP2cmf * W,
                        rP2cmf * W * theta**2,
                        (rP2 * (2 * P - 1) - f) / (1 - c)**2]
    EL12, EL13, EL23 = [rP2cmf * W * theta,
                        -rP2 * W,
                        -rP2 * W * theta]
    return np.nansum([[EL11, EL12, EL13],
                      [EL12, EL22, EL23],
                      [EL13, EL23, EL33]], axis=2)

########################################################################
## Compute optimal values for the fit parameters using
## maximum likelihood estimation in the 1PL, 2PL, and 3PL cases:
########################################################################

JH = {1: (J_1PL, H_1PL),
      2: (J_2PL, H_2PL),
      3: (J_3PL, H_3PL)}

def get_derivative_L(num_params, theta, r, f, zeta=None, lam=None, c=None,
                     do2nd=False):
    DL = JH[num_params][do2nd]
    def derivative_L(zlc):
        _zeta, _lam, _c = unpack_zlc(zeta, lam, c, zlc) # unpack the parameters
        Pstar = logistic(dev_zeta_lam(_zeta, _lam, theta))
        P = scale_guessing(Pstar, _c)
        return DL(theta, r, f, P, _zeta, _lam, _c, Pstar)
    return derivative_L

def get_JL(num_params, theta, r, f, zeta=None, lam=None, c=None):
    return get_derivative_L(num_params, theta, r, f, zeta, lam, c,
                            do2nd=False)

def get_HL(num_params, theta, r, f, zeta=None, lam=None, c=None):
    return get_derivative_L(num_params, theta, r, f, zeta, lam, c,
                            do2nd=True)

def mle_zlc(num_params, theta, r, f, zeta, lam, c, use_2nd=False,
            force_convergence=True, method=None, return_history=False):
    """Perform logistic ICC model parameter estimation using MLE
    Based on theoretical foundations for the 3PL model in
    "Item Response Theory: Parameter Estimation Techniques"

    This function is capable of performing 1PL, 2PL or 3PL depending on
    the value passed as "num_params"

    If return_history is True, this additionally stores and returns the
    history of zlc values.
    """

    theta, r, f = map(np.asanyarray, [theta, r, f]) # ensure these are arrays

    count = [0]

    # Get the Jacobian (1st derivatives) of the log likelihood function
    # based on current values of the zeta, lam, c parameters.
    # if use_2nd=True, also return the Hessian (2nd derivatives)

    J, H = JH[num_params]

    if return_history:
        zlc_hist = []

    ZETA, LAM, C = zeta, lam, c
    def JL(params):
        count[0] += 1

        # unpack the parameters
        zeta, lam, c = unpack_zlc(ZETA, LAM, C, params)

        Pstar = logistic(dev_zeta_lam(zeta, lam, theta))
        P = scale_guessing(Pstar, c)

        if c == 0 and num_params < 3:
            Pstar = None # optimize :)

        JLL = J(theta, r, f, P, zeta, lam, c, Pstar)

        if return_history:
            zlc_hist.append((zeta, lam, c))

        if not use_2nd:
            return JLL

        HLL = H(theta, r, f, P, zeta, lam, c, Pstar)

        return JLL, HLL

    kwds = (dict(jac=use_2nd) if method is None else
            dict(jac=use_2nd, method=method))
    results = root(JL, pack_zlc(zeta, lam, c, num_params), **kwds)
    if force_convergence and not results.success:
        raise ConvergenceError('scipy.optimize.root failed to converge')

    zeta, lam, c = unpack_zlc(zeta, lam, c, results.x)
    print count[0], 'iterations in root'

    P = logistic3PLzlc(zeta, lam, c, theta)
    chi2 = chi_squared(r, f, P)
    return (zeta, lam, c, chi2, results.success) + ((zlc_hist,) if return_history else ())

def mle_abc(num_params, theta, r, f, a, b, c, use_2nd=False,
            force_convergence=True, method=None, return_history=False):
    """This version transforms (a, b, c) into (zeta, lambda, c) and back
    when performing the actual fits"""
    zeta, lam = -a * b, a
    result = mle_zlc(num_params, theta, r, f, zeta, lam, c, use_2nd, force_convergence,
                     method=method, return_history=return_history)
    zeta, lam = result[:2]
    a, b = lam, -zeta / lam
    return (a, b) + result[2:]

mle_abc.__doc__ = '\n\n'.join([mle_zlc.__doc__, mle_abc.__doc__])


def mle_1_parameter(theta, r, f, a, b, c=0, use_2nd=False,
                    force_convergence=True, return_history=False):
    """One parameter logistic ICC model parameter estimation using MLE
    From "Item Response Theory: Parameter Estimation Techniques"
    This is a 1PL model using a reformulation of Baker's 2PL math"""
    return mle_abc(1, theta, r, f, a, b, c,
                   use_2nd=use_2nd, force_convergence=force_convergence,
                   return_history=return_history)

def mle_2_parameter(theta, r, f, a, b, c=0, use_2nd=False,
                    force_convergence=True, return_history=False):
    """Two parameter logistic ICC model parameter estimation using MLE
    From "Item Response Theory: Parameter Estimation Techniques"
    This uses a reformulation of Baker's 2PL math"""
    return mle_abc(2, theta, r, f, a, b, c,
                   use_2nd=use_2nd, force_convergence=force_convergence,
                   return_history=return_history)

def mle_3_parameter(theta, r, f, a, b, c, use_2nd=False,
                    force_convergence=True, method='hybr', return_history=False):
    """Three parameter logistic ICC model parameter estimation using MLE
    From "Item Response Theory: Parameter Estimation Techniques"
    This is a 3PL model using a reformulation of Baker's 2PL math"""
    return mle_abc(3, theta, r, f, a, b, c,
                   use_2nd=use_2nd, force_convergence=force_convergence,
                   method=method, return_history=return_history)

