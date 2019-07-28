from easydict  import EasyDict  as edict 

model_path  = "./models/"
cfg = edict(
    {

    'num_workers': 3, # Number of workers
    'model_path': model_path,
    'file_path' : model_path + "rnn",
    'output_size': 1,
    'embedding_dim': 64, #200
    'hidden_dim': 32, #128
    'n_layers': 2
    
    }
)