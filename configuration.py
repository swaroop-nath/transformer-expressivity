from json import dumps
##=========COMMON VARS=========##
emb_dim = 4 # CHANGE THIS WITH RESPECT TO THE INPUT EMB-DIM | For ease, keep this powers of 2

##=========WANDB SWEEPING HPARAMS -- HPARAM TUNING=========##
num_heads = [1]
while True:
    append_val = num_heads[-1] * 2
    if append_val <= emb_dim: num_heads.append(append_val)
    else: break
    
SWEEP_CONFIGURATION = {
    "name": f'm2n1-fcbn-cbrt-d{emb_dim}-pf32-classification-c10',
    "method": 'bayes',
    "metric": {
        "name": 'eval/best_failure_rate',
        "goal": 'minimize'
    },
    "parameters": {
        "num-transformer-layers": {
            "values": [2, 4, 6],
        },
        "num-heads": {
            "values": num_heads
        },
        "dropout": {
            "distribution": 'uniform',
            "max": 0.2,
            "min": 0.1
        },
        "epoch": {
            "values": [3, 5, 7],
        },
        "grad-acc": {
            "values": [1, 2, 4, 8],    
        },
        "learning-rate": {
            "distribution": 'uniform',
            "max": 1e-2,
            "min": 5e-6
        },
        "warmup-steps": {
            "values": [800, 1200, 1600]
        }
    }
}

class Configuration:
    def __init__(self):
        ##=========MODEL SPECIFIC HPARAMS=========##
        self.TRANSFORMER_KWARGS = {
            'emb-dim': emb_dim,
            'num-heads': -1,
            'num-encoder-layers': -1, # SET THIS USING SWEEP CONFIG
            'num-decoder-layers': -1, # SET THIS USING SWEEP CONFIG
            'pffn-dim': 32, # CHANGE THIS FOR PFFN-DIM
            'dropout': -1, # SET THIS USING SWEEP CONFIG
            'activation': 'gelu',
            'pre-ln': False
        }
        self.RETURN_LOSS = True

        ##=========DATA SPECIFIC HPARAMS=========##
        self.MODE = 'classification' # One of `regression` and `classification` # CHANGE THIS FOR MODE
        self.DATA_PATH = f'./synth-data/m2n1-fcbn-cbrt-d{emb_dim}' # CHANGE THIS FOR EXPERIMENT LOCATION
        self.PADDING_VALUE = -100
        self.NUM_CLASSES = 10 # CHANGE THIS FOR #CLASSES
        self.NUM_WORKERS = 4

        ##=========TRAINING SPECIFIC HPARAMS=========##
        self.LEARNING_RATE = -1 # SET THIS USING SWEEP CONFIG
        self.MIN_LR = 0.10 * self.LEARNING_RATE
        self.WEIGHT_DECAY = 0.05
        self.TRAIN_BATCH_SIZE = 128
        self.TEST_BATCH_SIZE = 256
        self.GRAD_ACC = -1 # SET THIS USING SWEEP CONFIG
        self.EPOCH = -1 # SET THIS USING SWEEP CONFIG
        self.WARMUP_STEPS = -1 # SET THIS USING SWEEP CONFIG
        self.LOG_STEPS = 50
        self.EVAL_STEPS = 200
        self.DEVICE = 'cuda:5'
        self.MAX_GRAD_NORM = 1.0

    def set_configuration_hparams(self, config):
        self.TRANSFORMER_KWARGS['num-encoder-layers'] = config['num-transformer-layers']
        self.TRANSFORMER_KWARGS['num-decoder-layers'] = config['num-transformer-layers']
        self.TRANSFORMER_KWARGS['num-heads'] = config['num-heads']
        self.TRANSFORMER_KWARGS['dropout'] = config['dropout']
        
        self.EPOCH = config['epoch']
        self.GRAD_ACC = config['grad-acc']
        self.LEARNING_RATE = config['learning-rate']
        self.WARMUP_STEPS = config['warmup-steps']
        self.MIN_LR = 0.9 * self.LEARNING_RATE
        
        return f"tr{config['num-transformer-layers']}-drp{config['dropout']}-gra{config['grad-acc']}-lr{config['learning-rate']:0.4g}-wmp{config['warmup-steps']}"
        
    def serialize(self):
        configuration = {
            "transfomer-kwargs": self.TRANSFORMER_KWARGS,
            "return-loss": self.RETURN_LOSS,
            "mode": self.MODE,
            "data-path": self.DATA_PATH,
            "padding-value": self.PADDING_VALUE,
            "num-classes": self.NUM_CLASSES,
            "num-workers": self.NUM_WORKERS,
            "learning-rate": self.LEARNING_RATE,
            "min-learning-rate": self.MIN_LR,
            "weight-decay": self.WEIGHT_DECAY,
            "train-batch-size": self.TRAIN_BATCH_SIZE,
            "test-batch-size": self.TEST_BATCH_SIZE,
            "grad-acc": self.GRAD_ACC,
            "log-steps": self.LOG_STEPS,
            "eval-steps": self.EVAL_STEPS,
            "warmup-steps": self.WARMUP_STEPS,
            "device": self.DEVICE,
        }
        
        return dumps(configuration, indent=4)
    