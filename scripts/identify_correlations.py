import itertools
import arviz as az
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sys import argv
from astropy.table import Table

def get_var_corr(df, col_i, col_j):
    x = df[col_i].values
    y = df[col_j].values
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    x_std = np.std(x)
    y_std = np.std(y)

    cv = 0
    for i, xi in enumerate(x):
        cv += (xi - x_mean) * (y[i] - y_mean)

    cv /= (len(x)-1)

    corr = cv / (x_std * y_std)

    return corr


if __name__ == '__main__':

    cols = np.loadtxt('../data/min_cols_to_load.txt',dtype=str).T
    # Read in the data
    data = Table.read(argv[1])

    # Create a pandas DataFrame
    df = data.to_pandas()[cols]
    df.dropna(subset=['PC-MZPD', 'PSF-NOBJ','RADECOFF'], inplace=True)
    # df.dropna(subset=['PC-MZPD','RADECOFF'], inplace=True)
    df['A-NAST'] = df['A-NAST'].astype(float)
    df['NOBJECTS'] = df['NOBJECTS'].astype(float)
    df['NCOSMICS'] = df['NCOSMICS'].astype(float)

    df['FILTER'] = df['FILTER'].str.decode('utf-8')
    df['QC-FLAG'] = df['QC-FLAG'].str.decode('utf-8')

    dfu = df.loc[df['FILTER'] == 'u']
    dfi = df.loc[df['FILTER'] == 'i']
    dfq = df.loc[df['FILTER'] == 'q']

    print('Total u-band observations: ', len(dfu))
    print('Total i-band observations: ', len(dfi))
    print('Total q-band observations: ', len(dfq))

    combinations = itertools.combinations(dfq.columns, 2)
    for col_i, col_j in combinations:
        if ( (col_i not in ['QC-FLAG','FILTER']) and (col_j not in ['QC-FLAG','FILTER']) ):
            print('Processing columns: ', col_i, col_j)
            corr = get_var_corr(dfq, col_i, col_j)

            if abs(corr) >= 0.5:
                print('Strong correlation between {} and {}: {:.2f}'.format(col_i, col_j, corr))
                fig, ax = plt.subplots(1, 1, figsize=(8, 6))
                ax.scatter(dfq[col_i], dfq[col_j], c=dfq['QC-FLAG'], s=2, alpha=0.5)
                # sns.scatterplot(x=col_i, y=col_j, data=dfq, hue='QC-FLAG', ax=ax, label='Correlation: {:.2f}'.format(corr))
                plt.title('{} vs {} | corr = {:.2f}'.format(col_i, col_j,corr))
                ax.set_xlabel(col_i)
                ax.set_ylabel(col_j)
                fig.tight_layout()
                plt.savefig('../data/figures/{}_vs_{}.png'.format(col_i, col_j))
                plt.close()
                plt.clf()