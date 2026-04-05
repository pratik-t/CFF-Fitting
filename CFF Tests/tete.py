import numpy as np
import pandas as pd

import plotly.graph_objects as go
df = pd.read_parquet('./data/2CFF_MSE_set4.parquet')

def plot(cff1, cff2, zrange):

    names = ['Re(H)','Re(Ht)','Re(E)','Re(Et)','Im(H)','Im(Ht)','Im(E)','Im(Et)']

    if (cff1<cff2):
        i1 = cff1
        i2 = cff2
    else:
        i1 = cff2
        i2 = cff1
        

    df_sub = df[(df['i1']==i1) & ((df['i2']==i2))].copy()

    pivot = df_sub.pivot_table(
        index='cff2',
        columns='cff1',
        values='chi2',
        aggfunc='min'
    )

    X = pivot.columns.to_numpy()
    Y = pivot.index.to_numpy()
    Z = np.log10(pivot.values + 1e-15)

    x_true = df_sub['true_cff1'].values[0]
    y_true = df_sub['true_cff2'].values[0]

    mask = (Z >= zrange[0]) & (Z <= zrange[1])
    x_vals = X[np.any(mask, axis=0)]
    y_vals = Y[np.any(mask, axis=1)]
    xmin, xmax = x_vals.min(), x_vals.max()
    ymin, ymax = y_vals.min(), y_vals.max()

    fig = go.Figure(
    data=go.Heatmap(
        x=X,
        y=Y,
        z=Z,
        colorbar=dict(title='log(MSE)'), 
        zmin=zrange[0], zmax=zrange[1]))

    fig.add_trace(go.Scatter(
    x=[x_true],
    y=[y_true],
    mode='markers',
    marker=dict(
        color='red',
        size=10,
        symbol='x'
    )))
    
    fig.add_trace(go.Contour(
    x=X,
    y=Y,
    z=Z,
    zmin=zrange[0], zmax=zrange[1],
    ncontours=50,
    contours=dict(showlines=True),
    showscale=False,
    line=dict(color='white', width=1)
    ))

    fig.update_layout(width= 800, height=800, 
                    xaxis_title=names[i1],yaxis_title=names[i2],
                    xaxis=dict(range=[xmin, xmax]),
                    yaxis=dict(range=[ymin, ymax]),
                    font=dict(family="Serif", size=16))
    
    fig.write_image(f'./figs/2CFF_{cff1}{cff2}.pdf')

    
for i in range(7):

    cffs_other = np.arange(i+1, 8)

    for j in cffs_other:

        print(i,j)
        plot(i, j, [-6,-1])