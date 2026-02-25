from bkm10 import BKM10
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

# fn = 'finalised pseudodata.csv'

# df = pd.read_csv(fn)

# df = df[df['set']==4]

# plus = BKM10(df['k'].values[0], df['Q2'].values[0], df['xB'].values[0], df['t'].values[0], helc=1)
# mins = BKM10(df['k'].values[0], df['Q2'].values[0], df['xB'].values[0], df['t'].values[0], helc=-1)

# phi = np.linspace(min(df['phi']), max(df['phi']), 100)

# phi= np.pi-np.deg2rad(phi)
# phi = tf.convert_to_tensor(phi, dtype=tf.float32)
# cffs = [[0,0,0,0,0,0,0,0]]
# cffs = tf.convert_to_tensor(cffs, dtype=tf.float32)

# splus = plus.calculate_cross_section(phi, cffs).numpy()
# smins = mins.calculate_cross_section(phi, cffs).numpy()

# dsig = 0.5*(splus+smins)
# delsig = 0.5*(splus-smins)

df = pd.read_csv('test.csv')
dsig_weight = 1/(df['dsig_err'])
dsig_weight = np.where(np.isfinite(dsig_weight), dsig_weight, 1.0)
print(dsig_weight)
dsig_weight /= np.sum(dsig_weight)
print(dsig_weight)
