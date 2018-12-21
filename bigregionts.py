import fwdpy11 as fp11
import fwdpy11.model_params
import fwdpy11.genetic_values
import fwdpy11.ts
import fwdpy11.tsrecorders
import fwdpy11.wright_fisher_ts
import pandas as pd
from collections import namedtuple
import numpy as np
import pickle
import lzma
import argparse
import sys
import os

Data = namedtuple('Data', ['generation', 'zbar', 'vg'])


def make_parser():
    parser = argparse.ArgumentParser(
        description="Simulate large genomic region")
    parser.add_argument("popsize", type=int, help="Population size")
    parser.add_argument("mu", type=float, help="Mutation rate")
    parser.add_argument("opt", type=float, help="Value of new optimum")
    parser.add_argument("sample_size", type=int, default=None,
                        help="Sample size (# diploids). Default is None")
    parser.add_argument("output_file", type=str,
                        help="Output file name for pickled populations")
    parser.add_argument(
        "db_file", type=str,
        help="Database name for summaries of genetic variance")

    parser.add_argument("--rho", type=float, default=1e4,
                        help="Scaled recombination rate rho. Default = 1e4")
    parser.add_argument("--VS", type=float, default=1.0,
                        help="Strength of stabilizing selection, default = 1.0")
    parser.add_argument("--gc", type=int, default=100,
                        help="Simplification interval (default = 100)")
    parser.add_argument("--nreps", type=int, default=1,
                        help="Number of replicates. Default = 1")
    parser.add_argument("--ngenes", type=int, default=100,
                        help="Number of regions where mutations affect the trait. Default = 100")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random number seed. Default is 42")
    parser.add_argument("--post", type=int, default=100,
                        help="Number of generations to simulate, post optimum shift. Default is 100")

    return parser


class Recorder(object):
    def __init__(self, nsam, post):
        self.data = []
        self.nsam = nsam
        self.post = post

    def __call__(self, pop, recorder):
        t = np.array(pop.diploid_metadata, copy=False)
        self.data.append(Data(pop.generation, t['g'].mean(), t['g'].var()))

        if pop.generation >= 9*pop.N and pop.generation < 10*pop.N:
            if pop.generation % 100 == 0.0:
                # Sample 100 random individuals
                s = np.random.choice(pop.N, self.nsam, replace=False)
                recorder.assign(s)
        elif pop.generation >= 10*pop.N and \
                pop.generation < 10*pop.N + self.post:
            s = np.random.choice(pop.N, self.nsam, replace=False)
            recorder.assign(s)


def runsim(args):
    N = args.popsize
    GSSmo = fp11.genetic_values.GSSmo([(0, 0, 1),
                                       (10*N, 1, 1)])

    genome_len = 10*args.ngenes
    sregion_starts, steps = np.linspace(
        0, genome_len, args.ngenes, retstep=True)
    while steps < 1.0:
        genome_len *= 2.0
        sregion_starts, steps = np.linspace(
            0, genome_len, args.ngenes, retstep=True)
    sregions = []
    for i in sregion_starts:
        sregions.append(fwdpy11.GaussianS(i, i+1, 1, 0.25))

    LEN = N*10 + args.post  # Simulate for 100 generations past the optimum shift
    p = {'nregions': [],  # No neutral mutations -- add them later!
         # The genetic value of a diploid will be according
         # to the model in the 2013 paper, and GSSmo will
         # map genetic value -> fitness via gaussian stabilizing selection
         'gvalue': fwdpy11.genetic_values.SlocusAdditive(2.0, GSSmo),
         # For the GBR model, effect sizes must be non-negative,
         # so we use an exponential distribution here
         'sregions': sregions,
         'recregions': [fp11.Region(0, genome_len, 1)],
         # Mutation rate to neutral, selected mutations, and recombination rate
         'rates': (0.0, args.mu, args.rho/float(4*N)),
         # Keep mutations at frequency 1 in the pop if they affect fitness.
         'prune_selected': False,
         'demography': np.array([N]*LEN, dtype=np.uint32)
         }
    params = fp11.model_params.ModelParams(**p)
    pop = fp11.SlocusPop(N, genome_len)
    r = Recorder(args.sample_size, args.post)
    rng = fp11.GSLrng(args.seed)
    fwdpy11.wright_fisher_ts.evolve(
        rng, pop, params, args.gc, r, suppress_table_indexing=True)
    return pop, r.data


if __name__ == "__main__":
    parser = make_parser()
    args = parser.parse_args(sys.argv[1:])
    if args.sample_size is None:
        raise ValueError("sample_size cannot be None")
    for i in [args.db_file, args.output_file]:
        if os.path.exists(i):
            os.remove(i)
    pop, data = runsim(args)
    df = pd.DataFrame(data, columns=Data._fields)
    df.to_csv(args.db_file, sep=" ", index=False)
    with lzma.open(args.output_file, 'wb') as f:
        pickle.dump(pop, f)
