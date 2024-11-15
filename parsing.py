import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--model", choices=["normal", "bernoulli", "poisson"], 
            help="You can only choose one of the following distributions: normal, bernoulli, poisson")
parser.add_argument("--dset", choices=["duncan", "spector", "warpbreaks"], 
            help="Each dataset is for specific distribution: duncan for normal, spector for bernoulli, warpbreaks for poisson")
parser.add_argument("--predictors", nargs="+", help="Enter your predictor variables spacing (like this: --predictors x1 x2)")
parser.add_argument("--add_intercept", action="store_true", help='adding intercept to a model')

args = parser.parse_args()
print(args)
