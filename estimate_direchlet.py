#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
from __future__ import division
import scipy as sp
from scipy.special import gammaln, psi, polygamma
import numpy as np
from numpy.linalg import norm
from numpy.random import multinomial, dirichlet
import numpy.random
import pandas as pd
from scipy.optimize import fmin_l_bfgs_b as optim

N = 100
C = 50

def invers_digamma(X):
    M = np.where(X >= -2.22, 1.0, 0.0)
    Y = M * (np.exp(X) + 0.5) + (1-M) * -1/(X-psi(1))

    # make 5 Newton iterations:
    Y = Y - (psi(Y)-X)/polygamma(1, Y)
    Y = Y - (psi(Y)-X)/polygamma(1, Y)
    Y = Y - (psi(Y)-X)/polygamma(1, Y)
    Y = Y - (psi(Y)-X)/polygamma(1, Y)
    Y = Y - (psi(Y)-X)/polygamma(1, Y)
    return Y

def station_point(alpha, suffcient_p, t=20):
    while t>0:
        t -= 1
        alpha = invers_digamma(psi(np.sum(alpha))+suffcient_p)
    return alpha

def desc_p(P):
    c = P.shape[0]
    return np.sum(np.log(P), axis=0)/c

def generate_dirichlet_data(c=C, n=N):
    alpha = np.random.rand(c)
    return alpha, dirichlet(alpha, n)

def estimate_alpha(P, alpha_guess=None, precision=3):
    if alpha_guess is None:
        alpha_guess = P.mean(axis=0)[:]
    suffcient_p = desc_p(P)
    return station_point(alpha_guess, suffcient_p, precision)

def main():
    truth, observation = generate_dirichlet_data()
    suffcient_p = desc_p(observation)
    alpha = np.random.rand(C)
    alpha = station_point(alpha, suffcient_p)
    print norm(alpha-truth)
    print alpha
    print truth 


if __name__ == "__main__":
    main()
