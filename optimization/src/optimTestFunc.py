"""
===========================================
Optimization Test Functions
Author: Gordon Woo
Email:  wuguoning@gmail.com
Date:   2021-01-05
China University of Petroleum at Beijing

===========================================
Reference:
    1. https://www.sfu.ca/~ssurjano/ackley.html
    2. Adorio, E. P., & Diliman, U. P. MVF -
       Multivariate Test Functions Library in C
       for Unconstrained Global Optimization (2005).
       Retrieved June 2013, from
       http://http://www.geocities.ws/eadorio/mvf.pdf.
    3. Molga, M., & Smutnicki, C. Test functions for
       optimization needs (2005). Retrieved June 2013,
       from http://www.zsd.ict.pwr.wroc.pl/files/docs/functions.pdf.
===========================================
"""

import numpy as np

class OptimTestFunc(object):

    def AckleyFun(self, x):
        """
        f(minimum) = 0, minimum = (0,0,...,0)
        """
        N = len(x)
        a = 20.
        b = 0.2
        c = 2.0*np.pi
        total1 = 0
        total2 = 0
        for i in range(N):
            total1 += x[i]**2
            total2 += np.cos(c*x[i])
        total1 = np.exp(-b*np.sqrt(total1 / N))
        total2 = np.exp(total2 / N)

        return -a*total1 - total2 + a + np.exp(1.0)

