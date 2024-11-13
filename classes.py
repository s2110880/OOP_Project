from scipy.optimize import minimize
from scipy.stats import norm, bernoulli, poisson
import numpy as np

class GLM:
    def __init__(self, X, y):
        #instanse variables
        self._X = X
        self._y=y
        self._parameters = None
    
    def llik(self, parameters):
        #this method is for log_liklihood calculation for specific function defined in subclass
        raise NotImplementedError

    def negllik(self, parameters):
        return -self.llik(parameters)
    
    def prediction(self, new_X):
        etha = np.dot(new_X, self._parameters)
        return self.lf(etha)
    
    def lf(self, etha):
        #link function method, which will be individual for each subclass
        raise NotImplementedError

    def fit(self):
        #here I find MLE by minimizing negative log likelihood
        #creating initial guess of zeros
        init_params = np.zeros(self._X.shape[1])

        #resulted minimization
        res = minimize(self.negllik, init_params)
        #saving results in instanse variable
        self._parameters = res.x
        print(self._parameters)


class GLM_Normal(GLM):
    #method overrides method created in superclass
    def lf(self, etha):
        #returning link function for Normal distribution
        #for normal distribution it is just etha
        return etha
    
    #llik method also overrids method from superclass specifically for Normal Distribution
    def llik(self, parameters):
        #this is the formula for linear predictor etha 
        etha = np.dot(self._X, parameters)
        #assigning Normal Distribution link function to mu 
        mu = self.lf(etha)

        return np.sum(norm.logpdf(self._y, mu))


class GLM_Bernoulli(GLM):
    #method overrides method created in superclass
    def lf(self, etha):
        #this here is derived from table 1
        return 1/(1+np.exp(-etha))
    
    def llik(self, parameters):
        #same formula for linear predictor (XB)
        etha = np.dot(self._X, parameters)
        #assigning variable to a link fucntion
        mu = self.lf(etha)

        #calculating log-likelihood
        return np.sum(bernoulli.logpmf(self._y, mu))
    

#is this going to appear on github(test)?


