from src.cla_mcr_back import main
from hyperopt import hp, tpe
from hyperopt import fmin

def objective(args):
    mcr_len, std_mult = args
    return 1./main(dimension_of_mcr_vectors=mcr_len, threshold_std_multiplier=std_mult, sparseness=0.0002, dimension_of_cla_vectors=65536)

def objective1(args):
    mcr_len, std_mult = args
    print(mcr_len, std_mult)
    return mcr_len*std_mult

space = hp.choice('set', [(2048, hp.uniform('std_mult', 1, 10)),
                          (1024, hp.uniform('std_mult1', 1, 10))
                         ]
                 )

best = fmin(objective, space=space, algo=tpe.suggest, max_evals=100)

print(best)
