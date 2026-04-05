import tensorflow as tf
from bkm10 import BKM10
import pandas as pd
import numpy as np

def get_bkm(set_num):
    
    fn = 'finalised pseudodata.csv'
    df = pd.read_csv(fn)

    df_set = df[df['set']==set_num]

    plus = BKM10(df_set['k'].values[0], df_set['Q2'].values[0], df_set['xB'].values[0], df_set['t'].values[0], helc=1)
    mins = BKM10(df_set['k'].values[0], df_set['Q2'].values[0], df_set['xB'].values[0], df_set['t'].values[0], helc=-1)
    
    cffs = df_set[['ReH','ReHt','ReE','ReEt','ImH','ImHt','ImE','ImEt']].to_numpy()[0]
    
    phi = np.linspace(min(df_set['phi']), max(df_set['phi']), 100)
    phi= np.pi-np.deg2rad(phi)
    phi = tf.convert_to_tensor(phi, dtype=tf.float32)

    cffs = tf.convert_to_tensor(cffs, dtype=tf.float32)

    return phi, plus, mins, cffs

