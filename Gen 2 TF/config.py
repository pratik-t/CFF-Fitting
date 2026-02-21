import os

# ---- GPU control ----
# Hide all GPUs (CPU-only mode)
os.environ["CUDA_VISIBLE_DEVICES"] = ""  
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import tensorflow as tf

# # ---- Threading control ----
# Limit intra/inter op threads
tf.get_logger().setLevel('ERROR')
tf.config.set_visible_devices([], 'GPU')
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

import keras

# ---- Global config variables ----
CONFIG = {
    'sets': [4],
    'replicas': 2,
    'threads': 1,
    'data_filename': 'finalised pseudodata.csv', 
    
    'show_devices': False,
    'verbose': 0,
    'max_epochs': 1000, 
    'learning_rate': 1e-4,
    'modify_LR_factor': 0.8,
    'modify_LR_patience': 30,
    'minimum_LR': 1e-5,
    'early_stop_patience': 30,
    'layers': [10, 50, 100, 50, 10],
    'activation': 'relu', 
    'initializer': keras.initializers.glorot_uniform(),
    'model_summary' : False
}