import pandas as pd
import argparse
import numpy as np

def create_txt(subset_path, sims_path):
    test_df = pd.read_csv(subset_path)
    sims_df = pd.read_csv(sims_path, header=None, sep=',').values
    ids = np.unique(test_df['ID'].values).astype(int)

    test_sims_df = sims_df[ids]
    df = open('test_sims_params.csv', 'w')
    for i in range(len(test_sims_df)):
        df.write(','.join([str(x) for x in test_sims_df[i]]))
        df.write('\n')
    df.close()

