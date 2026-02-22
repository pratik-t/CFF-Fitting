from config import CONFIG
from bkm10 import BKM10
from cff_fit_model import CFF_Fit_Model
import os
import gc
import time
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd
import tensorflow as tf
from tqdm import tqdm
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from sklearn.utils import shuffle
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler


def run_replica(set_id, replica_id, kinematics_plus, kinematics_mins, df_inputs, df_outputs):
    
    model_class = CFF_Fit_Model(
        verbose =         CONFIG['verbose'],
        LR =              CONFIG['learning_rate'],
        mod_LR_factor =   CONFIG['modify_LR_factor'],
        mod_LR_patience = CONFIG['modify_LR_patience'],
        min_LR =          CONFIG['minimum_LR'],
        ES_patience =     CONFIG['early_stop_patience'])
    
    model_class.create_model(
        layers =      CONFIG['layers'],
        activation =  CONFIG['activation'], 
        initializer = CONFIG['initializer'], 
        summary =     CONFIG['model_summary'])

    sample_dsig = np.random.normal(loc=df_outputs['dsig'], scale=df_outputs['dsig_err'])
    sample_delsig = np.random.normal(loc=df_outputs['delsig'], scale=df_outputs['delsig_err'])

    while np.any(sample_dsig <= 0):
        mask = sample_dsig <= 0
        sample_dsig[mask] = np.random.normal(loc=df_outputs['dsig'][mask], scale=df_outputs['dsig_err'][mask])

    outs_train = df_outputs.copy()
    outs_train['dsig'] = sample_dsig
    outs_train['delsig'] = sample_delsig

    outs_train = shuffle(outs_train)

    kins_train = tf.convert_to_tensor(df_inputs, dtype=tf.float32)
    outs_train = tf.convert_to_tensor(outs_train, dtype=tf.float32)

    history = model_class.fit_model(kinematics_plus, kinematics_mins,
                                    kins_train, outs_train, 
                                    epochs=CONFIG['max_epochs'], batch=CONFIG['batch_size'], loss_type=CONFIG['loss'])
    
    cffs_pred = model_class.model(kins_train, training=False).numpy()

    cffs_fit = cffs_pred[0]
    phi_fit = tf.convert_to_tensor(df_outputs['phi'], dtype=tf.float32)
    dsig_plus = kinematics_plus.calculate_cross_section(phi_fit, cffs_pred)
    dsig_mins = kinematics_mins.calculate_cross_section(phi_fit, cffs_pred)
    dsig_fit = 0.5*(dsig_plus+dsig_mins)
    delsig_fit = 0.5*(dsig_plus-dsig_mins)

    result_csv = pd.DataFrame({'set': set_id, 'replica': replica_id, 'phi': np.rad2deg(np.pi-phi_fit), 
                               'dsig_sample': sample_dsig, 'dsig_fit': dsig_fit,
                               'delsig_sample': sample_delsig, 'delsig_fit': delsig_fit,
                               'ReH_pred': cffs_fit[0], 'ReHt_pred': cffs_fit[1], 
                               'ReE_pred': cffs_fit[2], 'ReEt_pred': cffs_fit[3], 
                               'ImH_pred': cffs_fit[4], 'ImHt_pred': cffs_fit[5]})
    
    history_csv = pd.DataFrame(history.history)
    history_csv.insert(0, "set", set_id)
    history_csv.insert(1, "replica", replica_id)

    del model_class

    return history_csv, result_csv

def worker(args):
    
    set_id, replica_id, filename = args
    
    if replica_id==1:
        os.system("clear")
        print(f'>> Set {set_id} Progress:\n')

    df = pd.read_csv(filename)

    # only choose columns for which experimental data exists.
    df = df[df['dsig'] > 0]

    # angle in trento convention
    df['phi'] = np.pi-np.deg2rad(df['phi'])

    df = df[df['set'] == set_id]

    df_kins = df[['Q2', 'xB', 't']].copy()

    df_dsig = df[['phi', 'dsig', 'dsig_err', 'delsig', 'delsig_err']].copy()

    dsig_weight = 1/(df_dsig['dsig_err'])
    dsig_weight /= np.sum(dsig_weight)
    delsig_weight = 1/(df_dsig['delsig_err'])
    delsig_weight /= np.sum(delsig_weight)

    df_dsig['dsig_weight'] = dsig_weight
    df_dsig['delsig_weight'] = delsig_weight

    k = df['k'].values[0]
    Q2 = df['Q2'].values[0]
    xb = df['xB'].values[0]
    t = df['t'].values[0]

    kinematics_plus = BKM10(k, Q2, xb, t, helc=1)
    kinematics_mins = BKM10(k, Q2, xb, t, helc=-1)

    input_pipeline = make_pipeline(
        MinMaxScaler()
    )

    df_kins = input_pipeline.fit_transform(df_kins)

    history, result = run_replica(
        set_id, replica_id, kinematics_plus, kinematics_mins, df_kins, df_dsig)

    gc.collect()
    
    return history, result

def run_set(set_id, num_replicas, filename, threads):
    
    ctx = mp.get_context("spawn")  # ensures new Python interpreters

    args_list = [(set_id, j+1, filename) for j in range(num_replicas)]

    results = []
    with ProcessPoolExecutor(max_workers=threads, mp_context=ctx) as ex:
        futures = [ex.submit(worker, args) for args in args_list]
        for f in tqdm(as_completed(futures), total=len(futures)):
            results.append(f.result())

    all_histories, all_results = zip(*results)

    result_writer = pq.ParquetWriter(
        f'data/sample/set_{set_id}.parquet',
        schema=pa.Table.from_pandas(all_results[0]).schema
    )

    history_writer = pq.ParquetWriter(
        f'data/history/set_{set_id}.parquet',
        schema=pa.Table.from_pandas(all_histories[0]).schema
    )

    for i in range(len(all_results)):
        result_writer.write_table(pa.Table.from_pandas(all_results[i]))
        history_writer.write_table(pa.Table.from_pandas(all_histories[i]))
    result_writer.close()
    history_writer.close()

if __name__ == "__main__":
    
    os.makedirs('data', exist_ok=True)
    os.makedirs('data/sample', exist_ok=True)
    os.makedirs('data/history', exist_ok=True)
    
    sets = CONFIG['sets']
    num_replicas = CONFIG['replicas']
    threads = CONFIG['threads']
    filename = CONFIG['data_filename']

    if CONFIG['show_devices']:
        print("Visible devices:", tf.config.get_visible_devices())
        print("Intra threads:", tf.config.threading.get_intra_op_parallelism_threads())
        print("Inter threads:", tf.config.threading.get_inter_op_parallelism_threads())
    
    start_time = time.time()

    for set_id in sets:
        run_set(set_id, num_replicas, filename, threads)

    end_time = time.time()

    elapsed = int(end_time - start_time)
    hours = elapsed // 3600
    minutes = (elapsed % 3600) // 60
    seconds = elapsed % 60
    print(f'\nTotal runtime: {hours}h {minutes}m {seconds}s')