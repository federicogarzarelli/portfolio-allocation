import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

def calculate_portfolio_var(w, cov):
    # function that calculates portfolio risk
    return (w.T @ cov @ w)

def risk_contribution(w, cov):
    """
    Compute the contributions to risk of the constituents of a portfolio, given a set of portfolio weights and a covariance matrix
    """
    # Marginal Risk Contribution
    MRC = cov @ w.T
    # Risk Contribution
    RC = np.multiply(MRC, w.T) / calculate_portfolio_var(w, cov)
    return RC

def target_risk_contribution(target_risk, cov):
    """
    Returns the weights of the portfolio such that the contributions to portfolio risk are as close as possible
    to the target_risk, given the covariance matrix
    """
    n = cov.shape[0]
    init_guess = np.repeat(1/n, n)
    bounds = ((0.0, 1.0),)*n
    # construct the constants
    weights_sum_to_1 = {'type': 'eq',
                        'fun': lambda w: np.sum(w) - 1
                        }

    def msd_risk(w, target_risk, cov):
        """
        Returns the Mean Squared Difference in risk contributions between weights and target_risk
        """
        w_contribs = risk_contribution(w, cov)
        return ((w_contribs - target_risk) ** 2).sum()

    w = minimize(msd_risk, init_guess,
                 args=(target_risk, cov), method='SLSQP',
                 options={'disp' : False},
                 constraints=weights_sum_to_1,
                 bounds=bounds)
    return w.x

"""
Test the risk parity class on random samples from a multivariate normal distribution.

The distribution has the same mean and covariance of the asset below:
------------------------------------------------------
Name 			        | Mean 		    | Stdev	     |
------------------------------------------------------
S&P 500			        | 0.055889303	| 0.19226612 |
Gold			        | 0.047811791	| 0.19771345 |
Commodities		        | 0.055117486	| 0.25125224 |
Long term bonds		    | 0.044180940	| 0.32972692 |
Intermediate term bonds	| 0.001598437	| 0.06113804 |
------------------------------------------------------

Covariance matrix
-----------------------------------------------------------------------------------------
            SP500	        GLD	            COM	            LTB	            ITB         |
-----------------------------------------------------------------------------------------
SP500	    0.00012102 	    0.00000733 	    0.00005970 	    -0.00007114 	-0.00000960 |
GLD	        0.00000733 	    0.00009797 	    0.00002028 	    0.00001718 	    0.00000313  |
COM	        0.00005970 	    0.00002028 	    0.00016614 	    -0.00005381 	-0.00000657 |
LTB	        -0.00007114 	 0.00001718 	-0.00005381 	 0.00022532 	 0.00002681 |
ITB	        -0.00000960 	 0.00000313 	-0.00000657 	 0.00002681 	 0.00000529 |
----------------------------------------------------------------------------------------
"""

"""
# Draw random samples from a multivariate normal distribution.

mean = np.array([0.055889303, 0.04418094, 0.001598437, 0.047811791, 0.055117486])
cov = np.array([ [0.0001210158,0.000007332669,0.00005969538,-0.00007114415,-0.000009601985],
                [0.000007332669,0.0000979746,0.00002027613,0.00001718156,0.000003128327],
                [0.00005969538,0.00002027613,0.0001661385,-0.00005381,-0.00000657358],
                [-0.00007114415,0.00001718156,-0.00005381,0.0002253209,0.00002681437],
                [-0.000009601985,0.000003128327,-0.00000657358,0.00002681437,0.000005293436]
                ])

x = np.random.multivariate_normal(mean, cov, 10000)
print(np.shape(x))

w = np.array([0.2, 0.2, 0.2, 0.2, 0.2])

w_pos = np.arange(len(w))

plt.bar(w_pos, risk_contribution(w, cov), align='center', alpha=0.5)
plt.title('Risk contribution of an EW portfolio')
plt.show()

target_risk = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
w_riskparity = target_risk_contribution(target_risk, cov)

plt.bar(w_pos, risk_contribution(w_riskparity, cov), align='center', alpha=0.5)
plt.title('Risk contribution of a Risk Parity portfolio')
plt.show()

"""