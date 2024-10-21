import umap
import arviz as az
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sys import argv
from astropy.table import Table




if __name__ == '__main__':

    cols = np.loadtxt('../data/min_cols_to_load.txt',dtype=str).T
    cols = np.loadtxt('../data/cols_from_paul.txt',dtype=str).T
    # Read in the data
    data = Table.read(argv[1])

    # Create a pandas DataFrame
    df = data.to_pandas()[cols]
    # cols = np.hstack([cols,['FRAC_OBJS']])

    # df.dropna(subset=['PC-MZPD', 'PSF-NOBJ','RADECOFF'], inplace=True)
    df.dropna(subset=['PC-MZPD','RADECOFF'], inplace=True)
    df['A-NAST'] = df['A-NAST'].astype(float)
    df['NGAIA'] = df['NGAIA'].astype(float)
    df['NOBJECTS'] = df['NOBJECTS'].astype(float)
    df['NCOSMICS'] = df['NCOSMICS'].astype(float)
    df['FRAC_OBJS'] = df['NOBJECTS'] / df['NGAIA']

    df['FILTER'] = df['FILTER'].str.decode('utf-8')
    df['QC-FLAG'] = df['QC-FLAG'].str.decode('utf-8')

    dfu = df.loc[df['FILTER'] == 'u']
    dfg = df.loc[df['FILTER'] == 'g']
    dfr = df.loc[df['FILTER'] == 'r']
    dfi = df.loc[df['FILTER'] == 'i']
    dfz = df.loc[df['FILTER'] == 'z']
    dfq = df.loc[df['FILTER'] == 'q']

    print('Total u-band observations: ', len(dfu))
    print('Total g-band observations: ', len(dfg))
    print('Total r-band observations: ', len(dfr))
    print('Total i-band observations: ', len(dfi))
    print('Total z-band observations: ', len(dfz))
    print('Total q-band observations: ', len(dfq))

    uflags = dfu['QC-FLAG'].tolist()
    gflags = dfg['QC-FLAG'].tolist()
    rflags = dfr['QC-FLAG'].tolist()
    iflags = dfi['QC-FLAG'].tolist()
    zflags = dfz['QC-FLAG'].tolist()
    qflags = dfq['QC-FLAG'].tolist()

    Nq = len(dfq)
    Nq_green = len(dfq.loc[dfq['QC-FLAG'] == 'green'])
    Nq_yellow = len(dfq.loc[dfq['QC-FLAG'] == 'yellow'])
    Nq_orange = len(dfq.loc[dfq['QC-FLAG'] == 'orange'])
    Nq_red = len(dfq.loc[dfq['QC-FLAG'] == 'red'])

    # For each column, plot a kernel density estimate of the distribution of values
    # with respect to the QC-FLAG

    for col in dfq.columns:
        if col not in ['QC-FLAG','FILTER']:
            print('Prpcessing column: ', col)
            fig, ax = plt.subplots(1, 1, figsize=(8, 6))
            for flag in np.unique(qflags):
                grid, kde = az.kde(dfq.loc[dfq['QC-FLAG'] == flag, col].values)
                kde /= np.max(kde)
                kde *= len(dfq.loc[dfq['QC-FLAG'] == flag, col].values) / Nq
                ax.plot(grid, kde, color=flag, label=flag)
                # sns.kdeplot(dfq.loc[dfq['QC-FLAG'] == flag, col], color=flag, label=flag)
            ax.set_title('q-band: ' + col)
            ax.legend()
            fig.tight_layout()
            plt.savefig('../data/figures/q_band_' + col + '_kde.png')
            plt.show()
            plt.close()