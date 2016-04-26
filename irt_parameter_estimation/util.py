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


from __future__ import division
import numpy as np

class ConvergenceError(Exception):
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return repr(self.value)

def expand_dims(P, *x):
    '''Add singleton dimensions to each x so that they have the
       same shape as P.
       This make broadcasting work over the first dimension in
       each array instead of the last (which is default)'''
    return [np.reshape(i, np.shape(i) + (1,) * (np.ndim(P)-np.ndim(i)))
            for i in x]

def log_likelihood(theta, r, f, P):
    '''Compute the log likelihood using the standard formula.
       theta is the x-axis of the fit data
       P is the probability'''
    r, f = expand_dims(P, r, f)
    return np.sum(r * np.log(P) + (f - r) * np.log(1 - P), axis=0)

def dev_zeta_lam(zeta, lam, theta):
    '''Compute the logistic deviate (Z, logit, etc...)'''
    theta = np.asanyarray(theta)
    return zeta + np.multiply.outer(theta, lam)

def dev_ab(a, b, theta):
    theta = np.asanyarray(theta)
    return a * np.add.outer(theta, -b)

def logistic(x):
    '''Compute values from the logistic function'''
    return 1. / (1 + np.exp(-x))

def scale_guessing(Pstar, c, d=1):
    return c + (d - c) * Pstar

def logistic3PLzlc(zeta, lam, c, theta):
    """P for the 3-parameter logistic model
    a = lambda
    -a * b = zeta

    so, dev_zeta_lam(zeta, lam, theta) = zeta + lam * theta = a * (theta - b)
    """
    Pstar = logistic(dev_zeta_lam(zeta, lam, theta))
    return scale_guessing(Pstar, c)

def logistic3PLabc(a, b, c, theta):
    """P for the 3-parameter logistic model
    a = lambda
    -a * b = zeta

    so, dev_zeta_lam(zeta, lam, theta) = zeta + lam * theta = a * (theta - b)
    """
    Pstar = logistic(dev_ab(a, b, theta))
    return scale_guessing(Pstar, c)

def logistic4PLabcd(a, b, c, d, theta):
    """P for the 3-parameter logistic model
    a = lambda
    -a * b = zeta

    so, dev_zeta_lam(zeta, lam, theta) = zeta + lam * theta = a * (theta - b)
    """
    Pstar = logistic(dev_ab(a, b, theta))
    return scale_guessing(Pstar, c, d)

def chi_squared(r, f, P, reduced=False): # was (f, p, P):
    """Compute Pearson's Chi-Squared Statistic
    Uses the formula derived in Baker Chapter 2, section 2.7
    """
    r, f = expand_dims(P, r, f)
    denom = f * P * (1 - P)
    if reduced:
        denom *= f
    return np.nansum((r - f * P)**2 / denom) # was f / PQ * (p - P)**2

def reduced_chi_squared(r, f, P):
    return chi_squared(r, f, P, reduced=True)

def pack_zlc(zeta, lam, c, num_params):
    assert 0<num_params<4, 'num_params must be 1, 2 or 3'
    return (zeta, lam, c)[:num_params]

def unpack_zlc(zeta, lam, c, params):
    num_params = len(params)
    assert 0<num_params<4, 'num_params must be 1, 2 or 3'
    return tuple(params[:num_params]) + (zeta, lam, c)[num_params:]

def pack_abc(a, b, c, num_params):
    if   num_params == 1: return b
    elif num_params == 2: return a, b
    elif num_params == 3: return a, b, c
    else:
        raise AttributeError('num_params must be 1, 2 or 3!')

def unpack_abc(a, b, c, params):
    num_params = len(params)
    if   num_params == 1: return a, params[0], c
    elif num_params == 2: return params[0], params[1], c
    elif num_params == 3: return params
    else:
        raise AttributeError('len(params) must be 1, 2 or 3!')

def get_L_zlc(theta, r, f, zeta=None, lam=None, c=None):
    """Get the Log Likelihood function
    based on current values of the zeta, lam, c parameters
    If zeta/lam/c are specified, then they can be excluded from the
    zlc resulting of the resulting function
    """
    def L(zlc):
        _zeta, _lam, _c = unpack_zlc(zeta, lam, c, zlc) # unpack the parameters
        Pstar = logistic(dev_zeta_lam(_zeta, _lam, theta))
        P = scale_guessing(Pstar, _c)
        return log_likelihood(theta, r, f, P)
    return L

def get_L_abc(theta, r, f, a=None, b=None, c=None):
    """Get the Log Likelihood function
    based on current values of the a, b, c parameters
    If any of a/b/c are specified, then they can be excluded from the
    abc argument of the resulting function
    """
    def L(abc):
        _a, _b, _c = unpack_abc(a, b, c, abc) # unpack the parameters
        Pstar = logistic(dev_ab(_a, _b, theta))
        P = scale_guessing(Pstar, _c)
        return log_likelihood(theta, r, f, P)
    return L
