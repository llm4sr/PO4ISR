Dataset_setting = {
    'bundle':{
        'num_node':14240
    },
    'games':{
        'num_node':17389
    },
    'ml-1m':{
        'num_node':3416
    }
}

Model_setting = {
    'MostPop':{
        'model_dir': 'mostpop',
        'pop_n': 100,
        'description': 'MostPop',
    },
    'SKNN':{
        'modir_dir':'sknn',
        'neibor': 50,
        'description': 'SKNN',
    },
    'GCE-GNN': {
        'model_dir': 'gcegnn',
        'dataloader':'GCEDataset',
        'activate': 'relu',
        'n_sample_all': 12,
        'n_sample': 12,
        'lr_dc': 0.1,
        'lr_dc_step': 3,
        'l2': 1e-5,
        'dropout_global': 0.5,
        'alpha': 0.2,
        'patience':3 ,
        'description': 'GCE-GNN',
        'session_len': 50,
        # need to be tuned
        'epochs':100,
        'item_embedding_dim': 32,
        'learning_rate': 0.001,
        'batch_size':64,
        'num_hop':1,
        'dropout_gcn':0,
        'dropout_local':0.5
    },
    'MCPRN': {
        'model_dir': 'mcprn_v4_block',
        'dataloader':'NARMDataset',
        'lr_dc': 0.1,
        'lr_dc_step': 3,
        'l2': 1e-5,
        'alpha': 0.2,
        'patience':3 ,
        'description': 'MCPRN',
        'session_len': 50,
        # need to be tuned
        'epochs':100,
        'item_embedding_dim': 32,
        'learning_rate': 0.001,
        'batch_size':16,
        'tau':0.01,
        'purposes':2
    },
    'STAMP': {
        'model_dir': 'stamp',
        'dataloader':'NARMDataset',
        'lr_dc': 0.1,
        'lr_dc_step': 3,
        'l2': 1e-5,
        'alpha': 0.2,
        'patience':3 ,
        'description': 'STAMP',
        'session_len': 50,
        # need to be tuned
        'epochs':100,
        'item_embedding_dim': 32,
        'learning_rate': 0.001,
        'batch_size':16
    },
    'NARM': {
        'model_dir': 'narm',
        'dataloader':'NARMDataset',
        'lr_dc': 0.1,
        'lr_dc_step': 3,
        'l2': 1e-5,
        'alpha': 0.2,
        'patience':3 ,
        'description': 'NARM',
        'session_len': 50,
        # need to be tuned
        'epochs':100,
        'item_embedding_dim': 32,
        'learning_rate': 0.001,
        'batch_size':16,
        'hidden_size':100,
        'n_layers':1
    },
    'FPMC': {
        'model_dir': 'fpmc',
        'dataloader':'NARMDataset',
        'lr_dc': 0.1,
        'lr_dc_step': 3,
        'l2': 1e-5,
        'alpha': 0.2,
        'patience':3 ,
        'description': 'FPMC',
        'session_len': 50,
        # need to be tuned
        'epochs':100,
        'item_embedding_dim': 32,
        'learning_rate': 0.001,
        'batch_size':16
    },
    'HIDE': {
        'model_dir': 'hide',
        'dataloader':'HIDEDataset',
        'activate': 'relu',
        'n_sample_all': 12,
        'n_sample': 12,
        'n_iter':1,
        'lr_dc': 0.1,
        'lr_dc_step': 3,
        'l2': 1e-5,
        'n_layers': 1,
        'dropout_global': 0.5,
        'alpha': 0.2,
        'patience':3 ,
        'description': 'HIDE',
        'session_len': 50,
        'e':0.4,
        'disen':False, # need to be fixed
        'norm':True,
        'sw_edge': True,
        'item_edge': True,
        # need to be tuned
        'epochs':100,
        'item_embedding_dim': 32,
        'learning_rate': 0.001,
        'batch_size':64,
        'n_factor':3,
        'dropout_gcn':0,
        'dropout_local':0.5,
        'w':5,
        'lamda':0.01,
        'reg':1e-5
    },
    'AttenMixer': {
        'model_dir': 'attenMixer',
        'dataloader':'AttMixerDataset',
        'norm': True,
        'scale': True,
        'use_lp_pool': True,
        'softmax':True,
        'lr_dc': 0.1,
        'lr_dc_step': 3,
        'l2': 1e-5,
        'n_layers': 1,
        'dropout': 0.1,
        'alpha': 0.2,
        'patience':3 ,
        'description': 'HIDE',
        'session_len': 50,
        'dot':True,
        'last_k':7, # need to be fixed
        # need to be tuned
        'epochs':100,
        'item_embedding_dim': 32,
        'learning_rate': 0.001,
        'batch_size':64,
        'l_p':3,
        'heads':8
    }

}


HyperParameter_setting = {
    'GCE-GNN': {
        'categorical': {
            'item_embedding_dim': [32, 64, 128],
            'learning_rate': [0.0001, 0.001, 0.01],
            'batch_size': [64, 128, 256],
            'n_iter': [1, 2],
            'dropout_gcn': [0, 0.2, 0.4, 0.6, 0.8],
            'dropout_local': [0, 0.5],
        }
    },
    'MCPRN': {
        'categorical': {
            'item_embedding_dim': [32, 64, 128],
            'learning_rate': [0.0001, 0.001, 0.01],
            'batch_size': [64, 128, 256],
            'tau': [0.01, 0.1, 1, 10],
            'purposes': [1, 2, 3, 4]
        }
    },
    'STAMP': {
        'categorical': {
            'item_embedding_dim': [32, 64, 128],
            'learning_rate': [0.0001, 0.001, 0.01],
            'batch_size': [64, 128, 256],
        }
    },
    'NARM': {
        'int': {
            'hidden_size': {'min': 50, 'max': 200, 'step': 50}
        },
        'categorical': {
            'item_embedding_dim': [32, 64, 128],
            'learning_rate': [0.0001, 0.001, 0.01],
            'batch_size': [64, 128, 256],
            'n_layers': [1, 2, 3]
        }
    },
    'HIDE': {
        'int': {
            'w': {'min':1, 'max': 10, 'step': 1}
        },
        'categorical': {
            'item_embedding_dim': [32, 64, 128],
            'learning_rate': [0.0001, 0.001, 0.01],
            'batch_size': [64, 128, 256],
            'reg': [1e-5, 1e-4, 1e-3, 1e-2],
            'dropout_gcn': [0, 0.2, 0.4, 0.6, 0.8],
            'dropout_local': [0, 0.5],
            'n_factor': [1, 3, 5, 7, 9],
            'lamda': [1e-5, 1e-4, 1e-3, 1e-2]
        }
    },
    'FPMC': {
        'categorical': {
            'item_embedding_dim': [32, 64, 128],
            'learning_rate': [0.0001, 0.001, 0.01],
            'batch_size': [64, 128, 256],
        }
    },
     'AttenMixer': {
        'int': {
            'l_p': {'min':1, 'max': 10, 'step': 1}
        },
        'categorical': {
            'item_embedding_dim': [32, 64, 128],
            'learning_rate': [0.0001, 0.001, 0.01],
            'batch_size': [64, 128, 256],
            'heads': [1,2,4,8]
        }
    }
}

Best_setting = {
    'MostPop':{
        'bundle':{
            'pop_n':100,
        },
        'games':{
            'pop_n':100,
        },
        'ml-1m':{
            'pop_n':100,
        },
    },
    'SKNN':{
        'bundle':{
            'neibor':50,
        },
        'games':{
            'neibor':50,
        },
        'ml-1m':{
            'neibor':50,
        },
    },
    'GCE-GNN': {
        'bundle':{
            'epochs':100,
            'item_embedding_dim': 128,
            'learning_rate': 0.01,
            'batch_size':256,
            'n_iter':2,
            'dropout_gcn':0,
            'dropout_local':0.5
        },
        'games':{
            'epochs':100,
            'item_embedding_dim': 128,
            'learning_rate': 0.001,
            'batch_size':256,
            'n_iter':2,
            'dropout_gcn':0,
            'dropout_local':0
        },
        'ml-1m':{
            'epochs':100,
            'item_embedding_dim': 32,
            'learning_rate': 0.001,
            'batch_size':64,
            'n_iter':1,
            'dropout_gcn':0,
            'dropout_local':0.5
        }
        # need to be tuned
    },
    'MCPRN': {
        'bundle':{    
            'epochs':100,
            'item_embedding_dim': 32,
            'learning_rate': 0.01,
            'batch_size':128,
            'tau':0.01,
            'purposes':2
        },
        'games':{
            'epochs':100,
            'item_embedding_dim': 128,
            'learning_rate': 0.01,
            'batch_size':64,
            'tau':1,
            'purposes':1
        },
        'ml-1m':{
            'epochs':100,
            'item_embedding_dim': 128,
            'learning_rate': 0.001,
            'batch_size':256,
            'tau':0.1,
            'purposes':4
        }
        # need to be tuned
   
    },
    'STAMP': {
        'bundle':{
            'epochs':100,
            'item_embedding_dim': 128,
            'learning_rate': 0.01,
            'batch_size':256
        },
        'games':{
            'epochs':100,
            'item_embedding_dim': 128,
            'learning_rate': 0.001,
            'batch_size':256
        },
        'ml-1m':{
            'epochs':100,
            'item_embedding_dim': 32,
            'learning_rate': 0.0001,
            'batch_size':64
        }
    },
    'NARM': {
        'bundle':{
            'epochs':100,
            'item_embedding_dim': 32,
            'learning_rate': 0.001,
            'batch_size':64,
            'hidden_size':100,
            'n_layers':2
        },
        'games':{
            'epochs':100,
            'item_embedding_dim': 128,
            'learning_rate': 0.01,
            'batch_size':128,
            'hidden_size':100,
            'n_layers':3
        },
        'ml-1m':{
            'epochs':100,
            'item_embedding_dim': 64,
            'learning_rate': 0.0001,
            'batch_size':256,
            'hidden_size':50,
            'n_layers':2
        }
    },
    'FPMC': {
        'bundle':{
            'epochs':100,
            'item_embedding_dim': 32,
            'learning_rate': 0.01,
            'batch_size':128
        },
        'games':{
            'epochs':100,
            'item_embedding_dim': 128,
            'learning_rate': 0.01,
            'batch_size':64
        },
        'ml-1m':{
            'epochs':100,
            'item_embedding_dim': 64,
            'learning_rate': 0.001,
            'batch_size':128
        }
    },
    'HIDE': {
        'bundle':{
            'epochs':100,
            'item_embedding_dim': 64,
            'learning_rate': 0.0001,
            'batch_size':64,
            'n_factor':1,
            'dropout_gcn':0.6,
            'dropout_local':0,
            'w':6,
            'lamda':0.01,
            'reg':0.001
        },
        'games':{
            'epochs':100,
            'item_embedding_dim': 128,
            'learning_rate': 0.001,
            'batch_size':64,
            'n_factor':1,
            'dropout_gcn':0,
            'dropout_local':0.5,
            'w':3,
            'lamda':1e-5,
            'reg':1e-5
        },
        'ml-1m':{
            'epochs':100,
            'item_embedding_dim': 128,
            'learning_rate': 0.001,
            'batch_size':64,
            'n_factor':1,
            'dropout_gcn':0.2,
            'dropout_local':0.5,
            'w':5,
            'lamda':0.001,
            'reg':0.01
        }
    },
    'AttenMixer': {
        'bundle':{
            'epochs':100,
            'item_embedding_dim': 32,
            'learning_rate': 0.0001,
            'batch_size':256,
            'l_p':7,
            'heads':1
        },
        'games':{
            'epochs':100,
            'item_embedding_dim': 128,
            'learning_rate': 0.001,
            'batch_size':256,
            'l_p':3,
            'heads':4
        },
        'ml-1m':{
            'epochs':100,
            'item_embedding_dim': 32,
            'learning_rate': 0.001,
            'batch_size':64,
            'l_p':10,
            'heads':2
        }
    }

}

