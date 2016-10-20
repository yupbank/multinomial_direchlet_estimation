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
from scipy.optimize import fmin_l_bfgs_b
from estimate_direchlet import N, C, generate_dirichlet_data, estimate_alpha

N = 3200
C = 50
M = Maximun_obeservation_per_user = 5000

def generate_multinomial_data(priors, m=M):
    observations = np.zeros_like(priors)
    for n, prior in enumerate(priors):
        observations[n] = multinomial(np.random.randint(M), prior)
    return observations

def posterior_P(X, alpha):
    p_prime = X+alpha
    return (p_prime.T/np.sum(p_prime, axis=1)).T

def em(X, alpha, t=10):
    for i in xrange(t):
        P = posterior_P(X, alpha)
        alpha = estimate_alpha(P, alpha)
    return alpha, P

def mle(X, initial_params=None):
    infinitesimal = np.finfo(np.float).eps

    def _log_likelihood(params, *args):
        alpha = params
        X = args[0]
        c = X.shape[0]
        res = np.sum(gammaln(np.add(X, alpha))) \
                - c*np.sum(gammaln(alpha)) \
                + c*gammaln(np.sum(alpha)) \
                - np.sum(gammaln(np.sum(np.add(X, alpha), 1))) \
                + np.sum(gammaln(np.sum(X, 1)+1)) \
                - np.sum(gammaln(X+1))
        return -res

    if initial_params is None:
        initial_params = np.mean(X, 0) + infinitesimal

    bounds = [(infinitesimal, None)] * X.shape[1]
    optimres = fmin_l_bfgs_b(_log_likelihood,
                     x0=initial_params,
                     args=(X,),
                     approx_grad=1,
                     bounds=bounds)

    alpha = optimres[0]

    P = posterior_P(X, alpha)
    return alpha, P

def rank_match(Truth_X, Pred_P, plan, tops=10):
    Truth_rank = np.argsort(Truth_X, axis=0)[:tops, :]
    Pred_rank = np.argsort(Pred_P, axis=0)[:tops, :]
    total_hit = 0.0
    print '%s matches within top %s customer for each category'%(plan, tops)
    for i in xrange(Truth_rank.shape[1]):
        hit_num = len(set(Truth_rank[:, i]).intersection(set(Pred_rank[:, i])))
        total_hit += hit_num
        print 'cate_%s: %.3f,'%(i, hit_num/Truth_rank.shape[0]),
    print
    print '%s, total_hit rate:'%plan, total_hit/(Truth_rank.shape[0]*Truth_rank.shape[1])

def main():
    alpha_truth, P_truth = generate_dirichlet_data(C, N)
    X_train = generate_multinomial_data(P_truth)
    X_test = generate_multinomial_data(P_truth)
    alpha = np.random.rand(C)
    em_pred_alpha, em_pred_p = em(X_train, alpha)
    mle_pred_alpha, mle_pred_p = mle(X_test, alpha)
    print '%s category, %s customer, %s maximum purchase per customer'%(C, N, M)
    print 'difference in em_p:', norm(P_truth-em_pred_p), 'difference in em_alpha:', norm(alpha_truth-em_pred_alpha)
    print 'difference in mle_p:', norm(P_truth-mle_pred_p), 'difference in mle_alpha:', norm(alpha_truth-mle_pred_alpha)
    rank_match(X_test, em_pred_p, 'em')
    rank_match(X_test, mle_pred_p, 'mle')
    rank_match(X_test, X_train, 'history')
     
    



if __name__ == "__main__":
    main()
