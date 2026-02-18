import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

dfa = pd.read_parquet('data/history/set_1.parquet')
print(dfa)

# df = dfa.where(dfa['replica']==1)
# plt.plot(df['phi'], df['dsig_sample'], '.')
# plt.plot(df['phi'], df['dsig_fit'], '--')

# df = dfa.where(dfa['replica'] == 2)
# plt.plot(df['phi'], df['dsig_sample'], '.')
# plt.plot(df['phi'], df['dsig_fit'], '--')
# plt.show()

# import bkm10
# import numpy as np

# phi = 15
# phi = np.pi-np.deg2rad(phi)
# phi = tf.cast(phi, tf.float32)

# n = bkm10.BKM10(5.55, 1.51, 0.359, -0.18)
# cffs = [-2.51418, 1.35815, 2.19029, 18.22822, 3.21987, 1.50386]
# cffs = tf.expand_dims(cffs, axis=0)

# print(n.calculate_cross_section(phi, cffs))
