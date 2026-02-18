import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

pseudo = pd.read_csv('finalised pseudodata.csv')
pseudo = pseudo.where(pseudo['set']==1)

data = pd.read_parquet('./data/sample/set_1.parquet')
history = pd.read_parquet('./data/history/set_1.parquet')

print(data)

plt.plot(data['phi'], data['dsig_fit'])
plt.errorbar(pseudo['phi'], pseudo['dsig'], pseudo['dsig_err'], fmt='.')
plt.plot(data['phi'], data['dsig_sample'], 'x')
plt.show()