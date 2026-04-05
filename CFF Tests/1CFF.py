import tensorflow as tf
import pandas as pd
import numpy as np
from bkm10 import BKM10
import matplotlib.pyplot as plt

fn = 'finalised pseudodata.csv'
df = pd.read_csv(fn)

def get_bkm(set_num):
    
    df_set = df[df['set']==set_num]

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
    return tf.reduce_sum(tf.abs(y1-y1_true)**2+tf.abs(y2-y2_true)**2, axis=1)

def get_all_sig(set, index, N):

    phi, plus, mins, cffs = get_bkm(set)

    true_dsig, true_delsig = get_sig(phi, plus, mins, [cffs])

    x = cffs[index]
    scan_x = tf.linspace(-tf.abs(2*x)-10, tf.abs(2*x)+10, N-1)
    scan_x = tf.sort(tf.concat([scan_x, [x]], axis=0))
    
    
    cffs_batch = tf.tile(tf.expand_dims(cffs, axis=0), [N, 1])
    
    cffs_batch = tf.concat(
    [cffs_batch[:, :index],
     tf.expand_dims(scan_x, 1),
     cffs_batch[:, index+1:]],
    axis=1)
    
    dsig, delsig = get_sig(phi, plus, mins, cffs_batch)

    chi2 = loss(dsig, true_dsig, delsig, true_delsig)

    return x, scan_x, chi2

def get_all_grad(set_, index, N, eps):

    phi, plus, mins, cffs = get_bkm(set_)

    true_dsig, true_delsig = get_sig(phi, plus, mins, [cffs])

    x = cffs[index]
    scan_x = tf.linspace(-tf.abs(2*x)-10, tf.abs(2*x)+10, N-1)
    scan_x = tf.sort(tf.concat([scan_x, [x]], axis=0))

    cffs_batch_true = tf.tile(tf.expand_dims(cffs, axis=0), [N, 1])
    
    def cffeps(scan):
        return  tf.concat(
                [cffs_batch_true[:, :index],
                tf.expand_dims(scan, 1),
                cffs_batch_true[:, index+1:]],
                axis=1)
    
    scan_xpp = scan_x+2*eps
    scan_xmm = scan_x-2*eps

    
    cffs_batch = cffeps(scan_x)
    cffs_batch_pp = cffeps(scan_xpp)
    cffs_batch_mm = cffeps(scan_xmm)
    
    dsig, delsig = get_sig(phi, plus, mins, cffs_batch)
    dsig_pp, delsig_pp = get_sig(phi, plus, mins, cffs_batch_pp)
    dsig_mm, delsig_mm = get_sig(phi, plus, mins, cffs_batch_mm)

    chi2 = loss(dsig, true_dsig, delsig, true_delsig)
    chi2_pp = loss(dsig_pp, true_dsig, delsig_pp, true_delsig)
    chi2_mm = loss(dsig_mm, true_dsig, delsig_mm, true_delsig)

    grad_p = (chi2_pp - chi2)/(2*eps)
    grad_m = (chi2 - chi2_mm)/(2*eps)
    
    hess = (grad_p - grad_m)/(2*eps)

    def get_gradient(x_):
        
        x_= tf.Variable(x_, dtype=tf.float32)

        phi, plus, mins, cffs = get_bkm(set_)
        true_dsig, true_delsig = get_sig(phi, plus, mins, [cffs])
        
        with tf.GradientTape() as tape:
            cffs_batch = cffeps(x_)
            dsig, delsig = get_sig(phi, plus, mins, cffs_batch)
            loss_ = loss(dsig, true_dsig, delsig, true_delsig)  # forward pass recorded by tape
            
        grad = tape.gradient(loss_, x_)  # exact autograd gradient
        
        return grad
    
    grad = get_gradient(scan_x)
    
    return grad, hess

def get_loss_landscape(set_, iters):
    
    dfs_cffs = []

    for i in range(8):
        
        true_cff, cff, chi2 = get_all_sig(set_, i, iters)
        grad, hess = get_all_grad(set_, i, iters, 1e-6)
        chi2 = chi2.numpy()

        df_i = pd.DataFrame({
        "index": i,
        "cff": cff,
        "true_cff": np.full_like(cff, true_cff),
        "chi2": chi2, 
        "G": grad.numpy(), 
        "H": hess.numpy()})
    
        dfs_cffs.append(df_i)
        
    df_cff = pd.concat(dfs_cffs, ignore_index=True)
    df_cff.to_parquet("./data/1CFF_MSE_set4.parquet")

get_loss_landscape(4, iters=5000)