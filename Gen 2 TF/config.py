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

MAE = 1
MSE = 2

# ---- Global config variables ----
CONFIG = {
    'sets': [4],
    'replicas': 100,
    'threads': 45,
    'data_filename': 'test.csv', 
    
    'show_devices': False,
    'verbose': 0,
    'model_summary' : False,

    'max_epochs': 1000, 
    'batch_size': 1,
    'loss': MSE,
    'learning_rate': 1e-3,
    'modify_LR_factor': 0.8,
    'modify_LR_patience': 30,
    'minimum_LR': 1e-5,
    'early_stop_patience': 30,
    'layers': [10, 50, 100, 50, 10],
    'activation': 'relu', 
    'initializer': keras.initializers.glorot_uniform(),
}

FIXED_CFFS = {
        # 'ReH': -2.51484,
        # 'ImH': 3.20275, 
        # 'ReHt': 1.3474,
        # 'ImHt': 1.49975,
        'ReE': 2.1822,
        'ReEt': 126.28265,
        'ImE': 0.0,
        'ImEt': 0.0,
}
