import tensorflow as tf
import pandas as pd
import numpy as np
from bkm10 import BKM10
import matplotlib.pyplot as plt

names = ['Re(H)','Re(Ht)','Re(E)','Re(Et)','Im(H)','Im(Ht)','Im(E)','Im(Et)']

def get_bkm(row):
    
    plus = BKM10(row.k, row.Q2, row.xB, row.t, helc=1)
    mins = BKM10(row.k, row.Q2, row.xB, row.t, helc=-1)
    
    cffs = [row.ReH,row.ReHt,row.ReE,row.ReEt,row.ImH,row.ImHt,row.ImE,row.ImEt]
    
    phi = np.linspace(7.5, 352.5, 100)
    phi= np.pi-np.deg2rad(phi)
    phi = tf.convert_to_tensor(phi, dtype=tf.float32)

    cffs = tf.convert_to_tensor(cffs, dtype=tf.float32)

    ranges = [[-30., 30.], 
              [-30., 30.], 
              [-100., 100.], 
              [-400., 400.], 
              [-20., 20.],
              [-30., 30.],
              [-100., 100.],
              [-400., 400.]]
    
    return phi, plus, mins, cffs, ranges

def get_sig(phi, plus, mins, cffs):

    splus = plus.calculate_cross_section(phi, cffs)
    smins = mins.calculate_cross_section(phi, cffs)

    dsig = 0.5*(splus+smins)
    delsig = 0.5*(splus-smins)

    return dsig, delsig

def loss(y1, y1_true, y2, y2_true):
    return tf.reduce_sum(tf.abs(y1-y1_true)**2+tf.abs(y2-y2_true)**2, axis=1)

def get_all_sig(set, i1, i2, N):

    phi, plus, mins, cffs, ranges = get_bkm(set)
    
    true_dsig, true_delsig = get_sig(phi, plus, mins, tf.expand_dims(cffs, axis=0))

    x = cffs[i1]
    y = cffs[i2]

    x_range = ranges[i1] 
    y_range = ranges[i2]

    # scan_x = tf.linspace(-tf.abs(2*x)-10, tf.abs(2*x)+10, N-1)
    # scan_y = tf.linspace(-tf.abs(2*y)-10, tf.abs(2*y)+10, N-1)

    scan_x = tf.linspace(x_range[0], x_range[1], N)
    scan_y = tf.linspace(y_range[0], y_range[1], N)
    
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

def get_loss_landscape(row, iters):

    grids = {}

    for i in range(7):

        cffs_other = np.arange(i+1, 8)

        for j in cffs_other:
            
            print(i,j)
            
            true_cff1, true_cff2, cff1, cff2, chi2 = get_all_sig(row, i, j, iters)
            
            x = cff1.numpy().reshape(iters, iters)[:,0]
            y = cff2.numpy().reshape(iters, iters)[0]
            Z = chi2.numpy().reshape(iters,iters)
            Z = np.log10(Z + 1e-15)
            Z = Z.T
            mask = (Z <= -1)

            if not np.any(mask):
                cx, cy = iters//2, iters//2
                half = iters//10
                xmin_i, xmax_i = cx-half, cx+half
                ymin_i, ymax_i = cy-half, cy+half
            else:
                x_mask = np.any(mask, axis=0)
                y_mask = np.any(mask, axis=1)
                xmin_i, xmax_i = np.where(x_mask)[0][[0, -1]]
                ymin_i, ymax_i = np.where(y_mask)[0][[0, -1]]

            Z_crop = Z[ymin_i:ymax_i+1, xmin_i:xmax_i+1]
            x_crop = x[xmin_i:xmax_i+1]
            y_crop = y[ymin_i:ymax_i+1]
            
            key = f"{i}_{j}"

            grids[key] = {
            "z": Z_crop.astype(np.float16),
            "x": x_crop.astype(np.float16),
            "y": y_crop.astype(np.float16),
            "true": (float(true_cff1), float(true_cff2))
        }

    np.savez_compressed(f"./data grid/{row.Index}.npz", **grids)

fn = './grid.csv'
df = pd.read_csv(fn)

for row in df.itertuples():
    print(f'\n======{row.Index}=======\n')
    get_loss_landscape(row, iters=1000)
