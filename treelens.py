import fwdpy11
import fwdpy11.ts
import argparse
import sys
import numpy as np
import pandas as pd
from collections import namedtuple

DataRecord = namedtuple(
    "DataRecord", ['sampletime', 'left', 'right',
                   'ttime', 'tiptime', 'nmuts',
                   'vgproxy'])


def make_parser():
    parser = argparse.ArgumentParser(
        "Analyze tree sequences from simulations stored in binary files")
    parser.add_argument("infile", type=str, help="Input file name")
    parser.add_argument("outfile", type=str,
                        help="Output file name. Will be gzipped")
    return parser


def process_data(params):
    pop = fwdpy11.SlocusPop.load_from_file(params.infile)
    nodes = np.array(pop.tables.nodes, copy=False)
    ancient_node_ids = np.array(pop.tables.preserved_nodes)
    ancient_node_times = nodes['time'][ancient_node_ids]

    data = []
    for ant in np.unique(ancient_node_times):
        samples = ancient_node_ids[np.where(ancient_node_times == ant)[0]]
        sample_tables, idmap = fwdpy11.ts.simplify(pop, samples)
        sample_nodes = np.array(sample_tables.nodes, copy=False)
        sample_mutations = np.array(sample_tables.mutations)
        mpos = np.array(
            [pop.mutations[i].pos for i in sample_mutations['key']])
        remapped_samples = np.array(idmap, dtype=np.int32)[samples]
        tv = fwdpy11.ts.TreeVisitor(sample_tables, remapped_samples)

        while tv(False) is True:
            m = tv.tree()
            p = m.parents
            lc = m.leaf_counts
            stimes = sample_nodes['time'][remapped_samples]
            ptimes = sample_nodes['time'][p[remapped_samples]]

            ttime = m.total_time(sample_tables.nodes)
            tiptime = (stimes-ptimes).sum()

            # Ask if there are mutations on this tree
            muts_on_tree = np.where((mpos >= m.left) & (mpos < m.right))[0]
            # Get sum 2pqa^2 for all variants on the tree
            mnodes = sample_mutations['node'][muts_on_tree]
            mkeys = sample_mutations['key'][muts_on_tree]
            variance_proxy = 0.0
            for n, k in zip(mnodes, mkeys):
                q = lc[n]/len(samples)
                assert q > 0, "Mutation frequency error"
                p = 1.0 - q
                variance_proxy += 2.0*p*q*pop.mutations[k].s*pop.mutations[k].s

            data.append(DataRecord(ant, m.left, m.right,
                                   ttime, tiptime, len(muts_on_tree),
                                   variance_proxy))

    df = pd.DataFrame(data, columns=DataRecord._fields)
    df.to_csv(params.outfile, index=False, delim="\t", compression="gzip")


if __name__ == "__main__":
    parser = make_parser()
    params = parser.parse_args(sys.argv[1:])
    process_data(params)
