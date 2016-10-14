#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
import scipy as sp
from scipy.special import gammaln, psi
import numpy as np
from numpy.random import multinomial, dirichlet
import numpy.random
import pandas as pd
import pickle
from scipy.optimize import fmin_l_bfgs_b as optim


def load_data(file_name='sample.csv'):
    return pd.read_csv(file_name, index_col=0)

def cal_p(X, Alpha):
    Alpha_prime = X.add(Alpha, axis=1)
    P = Alpha_prime.div(Alpha_prime.sum(axis=1), axis=0)
    return P

def _log_likelihood(Alpha, X):
    P = cal_p(X, Alpha)
    res = 0.0
    c, k = X.shape
    res += sum(gammaln(X.sum(axis=1)+1))
    res -= sum(gammaln(X+1).sum())
    res += c*gammaln(sum(Alpha)+1)
    res -= c*sum(gammaln(Alpha))
    res += np.sum((X.add(Alpha, axis=1)-1).mul(np.log(P)).sum())
    return -res

def _jcobian(Alpha, X):
    P = cal_p(X, Alpha)
    res = 0.0
    c, k = X.shape
    S = sum(Alpha)
    constant = c*psi(S)
    return -(np.log(P).sum(axis=0)+constant-c*psi(Alpha)).as_matrix()

def papers(X, truth, t=10):
    infinitesimal = np.finfo(np.float).eps
    Alpha = np.mean(X, 0) + infinitesimal
    bounds = [(infinitesimal, None)] * X.shape[1]
    while t>0:
        t -= 1
        print t
        print map(lambda x: '%.3f'%x, truth)
        print map(lambda x: '%.3f'%x, Alpha)
        optimres = optim(_log_likelihood,
                x0=Alpha,
                approx_grad=1,
                bounds=bounds,
                #fprime=_jcobian,
                args=(X,),)
        Alpha = optimres[0]
    return Alpha

def log_likelihood(X, P, Alpha):
    assert X.shape == P.shape, 'the shape of observations should be same as prior probabilty for multinominal'
    res = 0.0
    c, k = X.shape
    res += sum(gammaln(X.sum(axis=1)+1))
    res -= sum(gammaln(X+1).sum())
    res += c*gammaln(sum(Alpha)+1)
    res -= c*sum(gammaln(Alpha))
    res += np.sum((X.add(Alpha, axis=1)-1).mul(np.log(P)).sum())
    return -res

def fit_mvpolya(X, initial_params=None):
    infinitesimal = np.finfo(np.float).eps

    def log_likelihood(params, *args):
        alpha = params
        X = args[0]

        res = np.sum([np.sum(gammaln(row+alpha)) \
                - np.sum(gammaln(alpha)) \
                + gammaln(np.sum(alpha)) \
                - gammaln(np.sum(row + alpha)) \
                + gammaln(np.sum(row)+1) \
                - np.sum(gammaln(row+1)) for row in X])

        return -res

    if initial_params is None:
        #initial_params = np.zeros(X.shape[1]) + 1.0
        initial_params = np.mean(X, 0) + infinitesimal

    bounds = [(infinitesimal, None)] * X.shape[1]
    optimres = optim(log_likelihood,
                     x0=initial_params,
                     args=(X,),
                     approx_grad=1,
                     bounds=bounds)

    params = optimres[0]
    return params

def main():
    X = load_data()
    truth = np.loadtxt('alphas.npy')
    Alpha = np.array([0.01 for i in xrange(X.shape[1])])
    P = cal_p(X, Alpha)
    paper_Alpha = papers(X, truth, t=50)
    borrow_Alpha = fit_mvpolya(X.as_matrix(), initial_params=None)
    print 'truth'
    print map(lambda x: '%.3f'%x, truth)
    print 'papers'
    print map(lambda x: '%.3f'%x, paper_Alpha)
    print 'borrowed model'
    print map(lambda x: '%.3f'%x, borrow_Alpha)

if __name__ == "__main__":
    main()
