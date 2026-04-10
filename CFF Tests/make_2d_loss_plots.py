import numpy as np
import pandas as pd
from scipy.ndimage import label

import plotly.graph_objects as go
df = pd.read_parquet('./data/2CFF_MSE_set4.parquet')

def plot(cff1, cff2, zrange):

    names = ['Re(H)','Re(Ht)','Re(E)','Re(Et)','Im(H)','Im(Ht)','Im(E)','Im(Et)']
    true_cffs = [-2.51484,1.3474,2.1822,126.28265,3.20275, 1.49975,0.0,0.0,]
    
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

    mask_ = (Z >= -8) & (Z <= -6)
    labeled, n_groups = label(mask_, structure=np.ones((3, 3), dtype=int))
    texts = []
    for group_idx in range(1, n_groups + 1):
        group_mask = labeled == group_idx
        x_vals_ = X[np.any(group_mask, axis=0)]
        y_vals_ = Y[np.any(group_mask, axis=1)]
        text_1 = f"{group_idx}: {names[i1]}=[{x_vals_.min():.3f}, {x_vals_.max():.3f}], Δ={abs(x_vals_.min()-x_vals_.max()):.3f}; {names[i2]}=[{y_vals_.min():.3f}, {y_vals_.max():.3f}], Δ={abs(y_vals_.min()-y_vals_.max()):.3f}"
        texts.append(text_1)    
    texts = "<br>".join(texts)
    
    fig = go.Figure(data= go.Contour(
    x=X,
    y=Y,
    z=Z,
    zmin=zrange[0], zmax=zrange[1],
    ncontours=50,
    showscale= False, 
    colorscale = 'Inferno',
    contours=dict(showlines=True),
    line=dict(color='white', width=1)
    ))

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

    fig.add_annotation(
    text=texts,
    xref="paper", yref="paper",
    x=0, y=1,                 
    xanchor="left",
    yanchor="top",
    showarrow=False,
    align="left",
    font=dict(color="black", size=18)  
    )

    fig.update_layout(width= 800, height=800, 
                    xaxis_title=names[i1],yaxis_title=names[i2],
                    xaxis=dict(range=[xmin, xmax]),
                    yaxis=dict(range=[ymin, ymax]),
                    font=dict(family="Serif", size=20), 
                    margin=dict(l=10, r=10, t=10, b=10))
    
    fig.write_image(f'./figs/2CFF_{cff1}{cff2}.pdf')

    
for i in range(7):

    cffs_other = np.arange(i+1, 8)

    for j in cffs_other:

        print(i,j)
        plot(i, j, [-8,-1])

# colorbar

zrange = [-8,-1]
fig = go.Figure()

fig.add_trace(go.Contour(
    z=[[zrange[0], zrange[1]]],
    colorscale='Inferno',
    showscale=True,
    contours=dict(showlines=False),
    line=dict(width=0),
    hoverinfo='skip',
    colorbar=dict(
        orientation='h',
        x=0.5,
        y=0.5,
        xanchor='center',
        yanchor='middle',
        len=0.7, 
        title='log(MSE)',
        title_side='top'
    )
))

fig.update_layout(
    width=1000,
    height=200,
    paper_bgcolor='white',
    plot_bgcolor='white',
    xaxis=dict(visible=False),
    yaxis=dict(visible=False),
    font=dict(family="Serif", size=20),
    margin=dict(l=0, r=0, t=0, b=0)
)

fig.write_image(f'./figs/colorbar.pdf', scale=2)