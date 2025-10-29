import pandas as pd
from km15 import ModKM15_CFFs
import numpy as np

df = pd.read_csv('finalised pseudodata.csv')

for i in range(len(df['set'])):
    print(ModKM15_CFFs(df['Q2'][i], df['xB'][i], df['t'][i]))


print(df['exp d4sig (nb/Gev^4)'], df['exp del4sig (nb/GeV^4)'])

# np.random.seed(42)

# for i in range(len(df['set'])):
#     print(np.random.normal(loc=df['exp d4σ (nb/Gev^4)'][i], scale=df['dsig_err'][i]),
#         np.random.normal(loc=df['exp Δ4σ (nb/GeV^4)'][i], scale=df['delsig_err'][i]))
