import argparse
from scipy.optimize import minimize
from scipy.stats import norm
import numpy as np
import statsmodels.api as sm
import pandas as pd
from classes import GLM_Normal, GLM_Bernoulli, GLM_Poisson

parser = argparse.ArgumentParser(
    description="""This script helps run the following commands:
    1. python parsing.py --model poisson --dset warpbreaks --predictors wool tension --add_intercept
    2. python parsing.py --model normal --dset duncan --predictors education prestige --add_intercept
    3. python parsing.py --model bernoulli --dset spector --add_intercept
    """)
parser.add_argument("--model", choices=["normal", "bernoulli", "poisson"], 
            help="You can only choose one of the following distributions: normal, bernoulli, poisson")
parser.add_argument("--dset", choices=["duncan", "spector", "warpbreaks"], 
            help="Each dataset is for specific distribution: duncan for normal, spector for bernoulli, warpbreaks for poisson")
parser.add_argument("--predictors", nargs="+", help="Enter your predictor variables spacing (like this: --predictors x1 x2)")
parser.add_argument("--add_intercept", action="store_true", help='adding intercept to a model')

args = parser.parse_args()
#logic block for dataset
if args.dset == 'duncan':
    df = sm.datasets.get_rdataset("Duncan", "carData")
    df = pd.DataFrame(data = df.data)
    y = df['income']
elif args.dset == 'spector':
    df = sm.datasets.spector.load_pandas()
    y = df.endog
elif args.dset == 'warpbreaks':
    df = pd.read_csv("warpbreaks.csv")
    y = df['breaks']
else:
    raise ValueError
#special logic block for spector dataset also adding constant
if args.dset == 'spector':
    X = df.exog
    X = sm.add_constant(X)
else:
    if args.predictors:
        X = df[args.predictors]
        X = sm.add_constant(X)
#logic block to assign models
if args.model == 'normal':
    mdl = GLM_Normal(X, y)
elif args.model == 'bernoulli':
    mdl = GLM_Bernoulli(X, y)
elif args.model == 'poisson':
    mdl = GLM_Poisson(X, y)

mdl.fit()
predict = mdl.prediction(X)

print(predict)

#type python parsing.py --help to see more about the script

