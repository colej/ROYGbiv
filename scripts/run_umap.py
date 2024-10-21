import umap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sys import argv
from astropy.table import Table
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler



if __name__ == '__main__':

    cols = np.loadtxt('../data/min_cols_to_load.txt',dtype=str).T
    cols = np.loadtxt('../data/cols_from_paul.txt',dtype=str).T
    # Read in the data
    data = Table.read(argv[1])

    # Create a pandas DataFrame
    df = data.to_pandas()[cols]
    df.dropna(subset=['PC-MZPD','RADECOFF'], inplace=True)
    df['A-NAST'] = df['A-NAST'].astype(float)
    df['NGAIA'] = df['NGAIA'].astype(float)
    df['NOBJECTS'] = df['NOBJECTS'].astype(float)
    df['NCOSMICS'] = df['NCOSMICS'].astype(float)
    df['FRAC_OBJS'] = df['NOBJECTS'] / df['NGAIA']
    df.drop('NGAIA', axis=1, inplace=True)
    df.drop('NOBJECTS', axis=1, inplace=True)
    df['S-BKG'] = np.log(df['S-BKG'])


    df['FILTER'] = df['FILTER'].str.decode('utf-8')
    df['QC-FLAG'] = df['QC-FLAG'].str.decode('utf-8')

    dfu = df.loc[df['FILTER'] == 'u']
    dfi = df.loc[df['FILTER'] == 'i']
    dfq = df.loc[df['FILTER'] == 'q']

    dfu.drop('FILTER', axis=1, inplace=True)
    dfi.drop('FILTER', axis=1, inplace=True)
    dfq.drop('FILTER', axis=1, inplace=True)

    print('Total u-band observations: ', len(dfu))
    print('Total i-band observations: ', len(dfi))
    print('Total q-band observations: ', len(dfq))

    uflags = dfu['QC-FLAG'].tolist()
    iflags = dfi['QC-FLAG'].tolist()
    qflags = dfq['QC-FLAG'].tolist()

    dfu.drop('QC-FLAG', axis=1, inplace=True)
    dfi.drop('QC-FLAG', axis=1, inplace=True)
    dfq.drop('QC-FLAG', axis=1, inplace=True)

    print('Performing U-MAP dimensionality reduction...')

    ## Standardizing data
    dfq_standardized = StandardScaler().fit_transform(dfq)
\
    # Perform UMAP dimensionality reduction
    reducer = umap.UMAP( n_neighbors=10, n_components=2, random_state=42)
    # embedding = reducer.fit_transform(dfq)

    # # Create a DataFrame for the embedding
    # embedding_df = pd.DataFrame(embedding, columns=['UMAP1', 'UMAP2'])
    # embedding_df['QC-FLAG'] = qflags

    # # Plot the UMAP results
    # plt.figure(figsize=(10, 8))
    # scatter = plt.scatter(embedding_df['UMAP1'], embedding_df['UMAP2'], c=embedding_df['QC-FLAG'], alpha=0.7, s=2)
    # plt.xlabel('UMAP1')
    # plt.ylabel('UMAP2')
    # plt.title('UMAP Dimensionality Reduction of q-band Data | 10 Neighbors')
    # plt.savefig('../data/figures/umap_q_band_RawFeatures_10neighbors.png')
    # plt.show()



    # Run umap on standardized data
    umap_standardized_embedding = reducer.fit_transform(dfq_standardized)
    umap_standardized_embedding_df = pd.DataFrame(umap_standardized_embedding, columns=['UMAP1', 'UMAP2'])
    umap_standardized_embedding_df['QC-FLAG'] = qflags

    # plot umap
    fig, ax = plt.subplots(1, 1, figsize=(10, 8), num=3)
    scatter = plt.scatter(umap_standardized_embedding_df['UMAP1'], umap_standardized_embedding_df['UMAP2'], c=umap_standardized_embedding_df['QC-FLAG'], alpha=0.7, s=2)
    plt.xlabel('UMAP1')
    plt.ylabel('UMAP2')
    plt.title('UMAP Dimensionality Reduction of standardized q-band Data | 10 Neighbors')
    plt.savefig('../data/figures/umap_q_band_StandardizedFeatures_10neighbors.png')
    plt.show()


    # Run TSNE on the q-band data
    print('Performing TSNE dimensionality reduction...')
    tsne = TSNE(n_components=2, perplexity=50, random_state=42)
    # tsne_embedding = tsne.fit_transform(dfq)
    # tsne_embedding_df = pd.DataFrame(tsne_embedding, columns=['TSNE1', 'TSNE2'])
    # tsne_embedding_df['QC-FLAG'] = qflags
    # fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    # scatter = ax.scatter(tsne_embedding_df['TSNE1'], tsne_embedding_df['TSNE2'], c=tsne_embedding_df['QC-FLAG'], alpha=0.7, s=2)
    # ax.set_xlabel('TSNE1')
    # ax.set_ylabel('TSNE2')
    # ax.set_title('TSNE Dimensionality Reduction of q-band Data')
    # fig.tight_layout()
    # plt.savefig('../data/figures/tsne_q_band_RawFeatures.png')
    # plt.show()


    tsne_standardized_embedding = tsne.fit_transform(dfq_standardized)

    tsne_standardized_embedding_df = pd.DataFrame(tsne_standardized_embedding, columns=['TSNE1', 'TSNE2'])
    tsne_standardized_embedding_df['QC-FLAG'] = qflags
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    scatter = ax.scatter(tsne_standardized_embedding_df['TSNE1'], tsne_standardized_embedding_df['TSNE2'], c=tsne_standardized_embedding_df['QC-FLAG'], alpha=0.7, s=2)
    ax.set_xlabel('TSNE1')
    ax.set_ylabel('TSNE2')
    ax.set_title('TSNE Dimensionality Reduction of standardized q-band Data')
    fig.tight_layout()
    plt.savefig('../data/figures/tsne_q_band_StandardizedFeatures.png')
    plt.show()

