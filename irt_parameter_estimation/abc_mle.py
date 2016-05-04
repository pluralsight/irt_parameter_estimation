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

This version uses the 3PL (a, b, c) formulation for everything, which
has different properties than the (zeta, lambda, c) formulation
(in general the former is less stable).

If you know nothing, use the main zlc formulation and ignore this one :)
'''

from __future__ import division
import numpy as np
from scipy.optimize import root #, minimize

from util import (ConvergenceError, dev_ab, logistic,
                  scale_guessing, logistic3PLabc, chi_squared, reduced_chi_squared,
                  expand_dims, pack_abc, unpack_abc, get_L_abc)

########################################################################
## Functions to compute 1st and 2nd derivatives of the likelihood
## for the 1, 2, & 3 parameter logistic models
## For consistency, all functions now use: np.nansum( [<>] , axis=1 or 2)
########################################################################

def J_1PL(theta, r, f, P, a, b, c=0, Pstar=None):
    """Get the Jacobian of the log likelihood for the 1PL model
    (1st derivative wrt b)

    a here is an ARRAY, not a scalar
    """

    a = a * np.ones(P.shape) # force a to be an array
    theta, r, f = expand_dims(P, theta, r, f)
    rmfP = r - f * P
    Prat = 1 if Pstar is None else Pstar / P
    L2 = -a * rmfP * Prat
    return np.nansum([L2], axis=1)

def H_1PL(theta, r, f, P, a, b, c=0, Pstar=None):
    """Get the Hessian Matrix of the log likelihood for the 1PL model
    (2nd derivative wrt b)
    """

    theta, r, f = expand_dims(P, theta, r, f)
    rmfP = r - f * P
    Q = (1 - P)
    Pstar, Prat = ((P, 1) if Pstar is None else
                  (Pstar, Pstar / P))
    EL22 = -a**2 * f * Prat * Pstar * Q
    return np.nansum([[EL22]], axis=2)


def J_2PL(theta, r, f, P, a, b, c=0, Pstar=None):
    """Get the Jacobian of the log likelihood for the 2PL model
    (1st derivatives wrt a,b)

    a here is an ARRAY, not a scalar
    """

    a = a * np.ones(P.shape) # force a to be an array
    theta, r, f = expand_dims(P, theta, r, f)
    rmfP = r - f * P
    Prat = 1 if Pstar is None else Pstar / P
    thmb = theta - b
    L1, L2 = np.array([thmb, -a]) * rmfP * Prat
    return np.nansum([L1, L2], axis=1)

def H_2PL(theta, r, f, P, a, b, c=0, Pstar=None):
    """Get the Hessian Matrix of the log likelihood for the 2PL model
    (2nd derivative wrt a,b)
    """

    theta, r, f = expand_dims(P, theta, r, f)
    rmfP = r - f * P
    Q = (1 - P)
    Pstar, Prat = ((P, 1) if Pstar is None else
                  (Pstar, Pstar / P))
    thmb = theta - b
    EL11, EL22, EL12 = np.array([thmb**2,
                                 -a**2,
                                 a * thmb]) * f * Prat * Pstar * Q
    return np.nansum([[EL11, EL12],
                      [EL12, EL22]], axis=2)

def J_3PL(theta, r, f, P, a, b, c, Pstar):
    """Get the Jacobian of the log likelihood for the 3PL model
    (1st derivatives wrt a,b,c)

    a here is an ARRAY, not a scalar
    """

    a = a * np.ones(P.shape) # force a to be an array
    theta, r, f = expand_dims(P, theta, r, f)
    rmfP = r - f * P
    iPc = 1 / (P - c)
    Prat = Pstar / P
    thmb = theta - b
    L1, L2, L3 = np.array([thmb, -a, iPc]) * rmfP * Prat
    return np.nansum([L1, L2, L3], axis=1)

def H_3PL(theta, r, f, P, a, b, c, Pstar):
    """Get the Hessian Matrix of the log likelihood for the 3PL model
    (2nd derivative wrt a,b,c)
    """

    theta, r, f = expand_dims(P, theta, r, f)
    rmfP = r - f * P
    iPc = 1 / (P - c)
    Q = (1 - P)
    Qic = Q / (1 - c)
    Prat = Pstar / P
    thmb = theta - b
    EL11, EL22, EL33 = np.array([-P * Q * thmb**2 * Prat,
                                 -a**2 * P * Q * Prat,
                                 Qic * iPc]) * f * Prat
    EL12, EL13, EL23 = np.array([a * thmb * P * Q * Prat,
                                 -thmb * Qic,
                                 a * Qic]) * f * Prat
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

def get_derivative_L(num_params, theta, r, f, a=None, b=None, c=None,
                     do2nd=False):
    DL = JH[num_params][do2nd]
    def derivative_L(abc):
        _a, _b, _c = unpack_abc(a, b, c, abc) # unpack the parameters
        Pstar = logistic(dev_ab(_a, _b, theta))
        P = scale_guessing(Pstar, _c)
        return DL(theta, r, f, P, _a, _b, _c, Pstar)
    return derivative_L

def get_JL(num_params, theta, r, f, a=None, b=None, c=None):
    return get_derivative_L(num_params, theta, r, f, a, b, c,
                            do2nd=False)

def get_HL(num_params, theta, r, f, a=None, b=None, c=None):
    return get_derivative_L(num_params, theta, r, f, a, b, c,
                            do2nd=True)

def mle_abc(num_params, theta, r, f, a, b, c, use_2nd=False,
            force_convergence=True, method=None, return_history=False):
    """Perform logistic ICC model parameter estimation using MLE
    Based on theoretical foundations for the 3PL model in
    "Item Response Theory: Parameter Estimation Techniques"

    This function is capable of performing 1PL, 2PL or 3PL depending on
    the value passed as "num_params"

    If return_history is True, this additionally stores and returns the
    history of abc values.
    """

    theta, r, f = map(np.asanyarray, [theta, r, f]) # ensure these are arrays

    count = [0]

    # Get the Jacobian (1st derivatives) of the log likelihood function
    # based on current values of the a, b, c parameters.
    # if use_2nd=True, also return the Hessian (2nd derivatives)

    J, H = JH[num_params]

    if return_history:
        abc_hist = []

    A, B, C = a, b, c
    def JL(params):
        count[0] += 1

        # unpack the parameters
        a, b, c = unpack_abc(A, B, C, params)

        Pstar = logistic(dev_ab(a, b, theta))
        P = scale_guessing(Pstar, c)

        if c == 0 and num_params < 3:
            Pstar = None # optimize :)

        JLL = J(theta, r, f, P, a, b, c, Pstar)

        if return_history:
            abc_hist.append((a, b, c))

        if not use_2nd:
            return JLL

        HLL = H(theta, r, f, P, a, b, c, Pstar)

        return JLL, HLL

    kwds = (dict(jac=use_2nd) if method is None else
            dict(jac=use_2nd, method=method))
    results = root(JL, pack_abc(a, b, c, num_params), **kwds)
    if force_convergence and not results.success:
        raise ConvergenceError('scipy.optimize.root failed to converge')

    a, b, c = unpack_abc(a, b, c, results.x)
    print count[0], 'iterations in root'

    P = logistic3PLabc(a, b, c, theta)
    chi2 = chi_squared(r, f, P)
    return (a, b, c, chi2, results.success) + ((abc_hist,) if return_history else ())

def mle_zlc(num_params, theta, r, f, zeta, lam, c, use_2nd=False,
            force_convergence=True, method=None, return_history=False):
    """This version transforms (zeta, lambda, c) into (a, b, c) and back
    when performing the actual fits"""
    a, b = lam, -zeta / lam
    
    result = mle_abc(num_params, theta, r, f, a, b, c, use_2nd, force_convergence,
                     method=method, return_history=return_history)
    a, b = result[:2]
    zeta, lam = -a * b, a
    return (zeta, lam) + result[2:]

mle_zlc.__doc__ = '\n\n'.join([mle_abc.__doc__, mle_zlc.__doc__])

def mle_1_parameter(theta, r, f, a, b, c=0, use_2nd=False,
                    force_convergence=True, return_history=False):
    """One parameter logistic ICC model parameter estimation using MLE
    From "Item Response Theory: Parameter Estimation Techniques"
    This version of the 1PL is actually based on the same math as the 3PL model"""
    return mle_abc(1, theta, r, f, a, b, c,
                   use_2nd=use_2nd, force_convergence=force_convergence,
                   return_history=return_history)

def mle_2_parameter(theta, r, f, a, b, c=0, use_2nd=False,
                    force_convergence=True, return_history=False):
    """Two parameter logistic ICC model parameter estimation using MLE
    From "Item Response Theory: Parameter Estimation Techniques"
    This version of the 2PL is actually based on the same math as the 3PL model"""
    return mle_abc(2, theta, r, f, a, b, c,
                   use_2nd=use_2nd, force_convergence=force_convergence,
                   return_history=return_history)

def mle_3_parameter(theta, r, f, a, b, c, use_2nd=False,
                    force_convergence=True, method='hybr', return_history=False):
    """Three parameter logistic ICC model parameter estimation using MLE
    From "Item Response Theory: Parameter Estimation Techniques"
    This version is based directly on Baker's 3PL math"""
    return mle_abc(3, theta, r, f, a, b, c,
                   use_2nd=use_2nd, force_convergence=force_convergence,
                   method=method, return_history=return_history)

