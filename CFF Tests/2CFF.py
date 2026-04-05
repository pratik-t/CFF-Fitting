import tensorflow as tf
import pandas as pd
import numpy as np
from bkm10 import BKM10
import matplotlib.pyplot as plt

fn = 'finalised pseudodata.csv'
df = pd.read_csv(fn)

def get_bkm(set_num):
    
    df_set = df[df['set']==set_num].copy()

    plus = BKM10(df_set['k'].values[0], df_set['Q2'].values[0], df_set['xB'].values[0], df_set['t'].values[0], helc=1)
    mins = BKM10(df_set['k'].values[0], df_set['Q2'].values[0], df_set['xB'].values[0], df_set['t'].values[0], helc=-1)
    
    cffs = df_set[['ReH','ReHt','ReE','ReEt','ImH','ImHt','ImE','ImEt']].to_numpy()[0]
    
    phi = np.linspace(min(df_set['phi']), max(df_set['phi']), 100)
    phi= np.pi-np.deg2rad(phi)
    phi = tf.convert_to_tensor(phi, dtype=tf.float32)

    cffs = tf.convert_to_tensor(cffs, dtype=tf.float32)

    return phi, plus, mins, cffs

def get_sig(phi, plus, mins, cffs):

    splus = plus.calculate_cross_section(phi, cffs)
    smins = mins.calculate_cross_section(phi, cffs)

    dsig = 0.5*(splus+smins)
    delsig = 0.5*(splus-smins)

    return dsig, delsig

def loss(y1, y1_true, y2, y2_true):
    return tf.reduce_sum(tf.abs(y1-y1_true)+tf.abs(y2-y2_true), axis=1)

def get_all_sig(set, i1, i2, N):

    phi, plus, mins, cffs = get_bkm(set)

    true_dsig, true_delsig = get_sig(phi, plus, mins, tf.expand_dims(cffs, axis=0))

    x = cffs[i1]
    y = cffs[i2]

    scan_x = tf.linspace(-tf.abs(2*x)-10, tf.abs(2*x)+10, N-1)
    scan_x = tf.sort(tf.concat([scan_x, [x]], axis=0))
    scan_y = tf.linspace(-tf.abs(2*y)-10, tf.abs(2*y)+10, N-1)
    scan_y = tf.sort(tf.concat([scan_y, [y]], axis=0))
    
    scan_x = tf.repeat(scan_x, N)
    scan_y = tf.tile(scan_y, [N])
    
    cffs_batch = tf.tile(tf.expand_dims(cffs, axis=0), [N*N, 1])

    # Replace both columns at once
    indices_x = tf.stack([tf.range(N*N), tf.fill([N*N], tf.cast(i1, tf.int32))], axis=1)
    indices_y = tf.stack([tf.range(N*N), tf.fill([N*N], tf.cast(i2, tf.int32))], axis=1)

    cffs_batch = tf.tensor_scatter_nd_update(cffs_batch, indices_x, scan_x)
    cffs_batch = tf.tensor_scatter_nd_update(cffs_batch, indices_y, scan_y)

    dsig, delsig = get_sig(phi, plus, mins, cffs_batch)

    chi2 = loss(dsig, true_dsig, delsig, true_delsig)

    return x, y, scan_x, scan_y, chi2

def get_loss_landscape(set_, iters):
    
    dfs_cffs = []

    for i in range(7):

        cffs_other = np.arange(i+1, 8)

        for j in cffs_other:
            
            print(i,j)
            
            true_cff1, true_cff2, cff1, cff2, chi2 = get_all_sig(set_, i, j, iters)
            
            chi2 = chi2.numpy()

            df_sub = pd.DataFrame({
            "i1": i,
            "i2": j,
            "cff1": cff1,
            "cff2": cff2,
            "true_cff1": np.full_like(cff1, true_cff1),
            "true_cff2": np.full_like(cff2, true_cff2),
            "chi2": chi2, })
            dfs_cffs.append(df_sub)
        
    df_cff = pd.concat(dfs_cffs, ignore_index=True)
    df_cff.to_parquet("./data/2CFF_MAE_set4.parquet")

get_loss_landscape(4, iters=1000)