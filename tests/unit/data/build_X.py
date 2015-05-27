import pandas as pd
import numpy.random as npr 
import matplotlib.pyplot as pp





""" 
generate some toy data where a -> b, a-> c, b -> d, and c doesn't effect d.
"""

n = 100
a = npr.beta(2.5, 2.5, n)
b = npr.binomial( 1000, a)
c = npr.binomial( 1000, a)
d = 5. * b 
X = pd.DataFrame( { 'a' : a, 'b' : b, 'c' : c, 'd' : d})   
X.to_csv('./tests/unit/data/X.csv')



"""
generate a toy discrete dataset with the same dependence structure
as above
"""

n = 2000
a = npr.binomial(1, 0.25, n)
b = (a + npr.binomial(1, 0.75, n)) % 2
c = (a + npr.binomial(1,0.25, n)) % 2
d = (b + npr.binomial(1,0.75, n)) % 2
X = pd.DataFrame( { 'a' : a, 'b' : b, 'c' : c, 'd' : d})
X.to_csv('./tests/unit/data/discrete.csv')
