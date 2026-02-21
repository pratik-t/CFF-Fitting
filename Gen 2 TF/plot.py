import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

fig, ax = plt.subplots(1,2)

SET = 4

pseudo = pd.read_csv('finalised pseudodata.csv')
pseudo = pseudo.where(pseudo['set']==SET)

data_all = pd.read_parquet(f'./data/sample/set_{SET}.parquet')
history = pd.read_parquet(f'./data/history/set_{SET}.parquet')

replicas = max(data_all['replica'])

print(data_all)

def graph(replica, ls='-'):
    data = data_all.where(data_all['replica'] == replica)
    ax[0].plot(data['phi'], data['dsig_fit'], ls=ls)
    ax[0].errorbar(pseudo['phi'], pseudo['dsig'], pseudo['dsig_err'], fmt='.')
    ax[0].plot(data['phi'], data['dsig_sample'], 'x')

    ax[1].plot(data['phi'], data['delsig_fit'], ls=ls)
    ax[1].errorbar(pseudo['phi'], pseudo['delsig'], pseudo['delsig_err'], fmt='.')
    ax[1].plot(data['phi'], data['delsig_sample'], 'x')

for i in range(replicas):
    graph(i+1)

plt.figure()
plt.plot(history['loss'])
plt.show()