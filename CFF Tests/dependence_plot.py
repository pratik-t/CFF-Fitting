import numpy as np
import panel as pn
import holoviews as hv
from functools import lru_cache
import matplotlib.pyplot as plt
import pandas as pd

pn.extension()
hv.extension('bokeh')

names = ['Re(H)','Re(Ht)','Re(E)','Re(Et)','Im(H)','Im(Ht)','Im(E)','Im(Et)']
ranges = [(-20,20),(-20,20),(-100,100),(-400,400),(-20,20),(-30,30),(-100,100),(-400,400)]

df = pd.read_csv('grid.csv')

k_vals = df['k'].unique().tolist()
q2_vals = df['Q2'].unique().tolist()
keys = [f"{i}_{j}" for i in range(8) for j in range(i+1, 8)]

k_sel  = pn.widgets.Select(name='k', options=sorted(df['k'].unique()))
q2_sel = pn.widgets.DiscreteSlider(name='Q2', options=[])
xb_sel = pn.widgets.DiscreteSlider(name='xB', options=[])
t_sel  = pn.widgets.DiscreteSlider(name='t', options=[])
key_sel  = pn.widgets.DiscreteSlider(name='key',  options=keys)
info = pn.pane.Str('', styles={'font-family': 'monospace', 'font-size': '13px'})

@pn.depends(k_sel, watch=True)
def update_q2(k):
    vals = sorted(df[df['k']==k]['Q2'].unique().tolist())
    q2_sel.options = vals
    if q2_sel.value not in vals:
        q2_sel.value = vals[0]

@pn.depends(k_sel, q2_sel, watch=True)
def update_xb(k, q2):
    sub = df[(df['k']==k) & (df['Q2']==q2)]
    vals = sorted(sub['xB'].unique().tolist())
    xb_sel.options = vals
    if xb_sel.value not in vals:
        xb_sel.value = vals[0]

@pn.depends(k_sel, q2_sel, xb_sel, watch=True)
def update_t(k, q2, xb):
    sub = df[(df['k']==k) & (df['Q2']==q2) & (df['xB']==xb)]
    vals = sorted(sub['t'].unique().tolist())
    t_sel.options = vals
    if t_sel.value not in vals:
        t_sel.value = vals[0]

update_q2(k_sel.value)
update_xb(k_sel.value, q2_sel.value)
update_t(k_sel.value, q2_sel.value, xb_sel.value)
        
@lru_cache(maxsize=550)
def get_file(k, q2, xb, t):
    row = df[
        (df['k']==k) &
        (df['Q2']==q2) &
        (df['xB']==xb) &
        (df['t']==t)
    ].iloc[0]
    idx = row.name
    cffs = [row['ReH'],row['ReHt'],row['ReE'],row['ReEt'],row['ImH'],row['ImHt'],row['ImE'],row['ImEt']]
    return cffs, np.load(f'./data grid/{idx}.npz', mmap_mode='r', allow_pickle=True)

@pn.depends(k_sel, q2_sel, xb_sel, t_sel, key_sel)
def make_plot(k, q2, xb, t, key):
    idx = int(key.split('_')[0])
    idy = int(key.split('_')[1])

    cffs, d = get_file(k, q2, xb, t)
    info.object = f'Q2={q2}  xB={xb}  t={t} cffs={cffs[idx]},{cffs[idy]}'
    d = d[key].item()
    x, y, Z = d['x'], d['y'], d['z']
    
    x = np.linspace(float(x[0]), float(x[-1]), Z.shape[1])
    y = np.linspace(float(y[0]), float(y[-1]), Z.shape[0])
    Z = Z.astype(np.float32)
    
    img = hv.Image((x, y, Z)).opts(
        cmap='Inferno', clim=(-8, -1), colorbar=True,
        frame_width=500, frame_height=500,
        xlabel=names[idx], ylabel=names[idy], 
        xlim=ranges[idx], ylim=ranges[idy]
    )

    contour = hv.operation.contours(img, levels=[-5]).opts(cmap=['red'], line_width=1, show_legend=False)

    marker = hv.Points([(cffs[idx], cffs[idy])]).opts(marker='x', color='white', size=10, line_width=3)

    return img * marker * contour

app = pn.Column(
    pn.Row(k_sel, q2_sel),
    pn.Row(xb_sel, t_sel),
    pn.Row(key_sel, info),
    pn.panel(make_plot, loading_indicator=False)
).servable()

if __name__ == "__main__":
    pn.serve(
        app,
        port=5006,
        address="0.0.0.0",
        allow_websocket_origin=["*"],
        show=False
    )