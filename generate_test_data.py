#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
import scipy as sp
import numpy as np
from numpy.random import multinomial, dirichlet
import numpy.random
import pandas as pd
import pickle

NUM_people = 20
NUM_category = 10
NUM_obeservation_per_user = 5000
def main():
    alphas = np.random.rand(NUM_category)
    user_priors = numpy.random.dirichlet(alphas, NUM_people)
    obeservations = []
    for prior in user_priors:
        user_obeservation = multinomial(NUM_obeservation_per_user, prior)
        obeservations.append(user_obeservation)
    sample_data = pd.DataFrame(obeservations, columns=['cate_%s'%i for i in xrange(NUM_category)], index=['uid_%s'%i for i in xrange(NUM_people)])
    sample_data.to_csv('sample.csv')
    np.savetxt('alphas.npy', alphas)
    




if __name__ == "__main__":
    main()
