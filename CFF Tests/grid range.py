import pandas as pd
import numpy as np
import os 

fn = 'finalised pseudodata.csv'
df = pd.read_csv(fn)
unique = df[['k', 'Q2', 'xB', 't']].drop_duplicates().round(2).sort_values(by=['k', 'Q2', 'xB', 't'])

# print(unique.to_string())

# print(unique[['k', 'Q2']].value_counts().to_string())

# print(unique[(unique['k']==5.88) & (unique['Q2']==2.42)])

stats = (
    unique
    .groupby(['k', 'Q2'])
    .agg(
        count=('k', 'size'),
        xB_min=('xB', 'min'),
        xB_max=('xB', 'max'),
        t_min=('t', 'min'),
        t_max=('t', 'max')
    )
)

# add spreads
stats['xB_range'] = stats['xB_max'] - stats['xB_min']
stats['t_range']  = stats['t_max']  - stats['t_min']

stats = stats[(stats['count']>9) & (stats['xB_range']>0.05)]
stats = stats.sort_values(by= ['k', 'Q2'])
print(stats.to_string())

filename = "grid.csv"
if os.path.exists(filename):
    os.remove(filename)

for (k, Q2), row in stats.iterrows():
    xb_range = np.round(np.linspace(row['xB_min'], row['xB_max'], 5),4)
    t_range = np.round(np.linspace(row['t_min'], row['t_max'], 10),4)
    X, T = np.meshgrid(xb_range, t_range, indexing='ij')
    XT = np.column_stack([X.ravel(), T.ravel()])
    KQ= np.ones(np.shape(XT))
    KQ[:,0]*=k
    KQ[:,1]*=Q2
    KQXT = np.concat([KQ, XT], axis=1) 
    df_out = pd.DataFrame(KQXT, columns=['k', 'Q2', 'xB', 't'])
    df_out.to_csv(filename, mode='a', header=not os.path.exists(filename), index=False)
    