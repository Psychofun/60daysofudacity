import torch as th

from preprocess import   tokenize_message,pad_features, load_obj

# Load dictionaries 
index_to_word= load_obj("data/index_to_word")
word_to_index = load_obj("data/word_to_index")


   


def predict(net, message,use_gpu, sequence_length=200):

    """
    net: pytorch RNN model. Whiout encryption

    message: string
        message to classify on plain text
    use_gpu: Boolean 
        if True use GPU computation
    sequence_length: int
        Length of sequences
    """
    if len(message) == 0:
        return None 
    net.eval()
    
    # tokenize review
    test_ints = tokenize_message(message,word_to_index)
    
    # pad tokenized sequence
    seq_length=sequence_length
    features = pad_features(test_ints, seq_length)
    
    # convert to tensor to pass into your model
    feature_tensor = th.from_numpy(features)
    # cast tensor to int64
    feature_tensor = feature_tensor.to(th.int64)
    
    batch_size = feature_tensor.size(0)
    
    # initialize hidden state
    h = net.init_hidden(batch_size)
    
    if(use_gpu):
        if th.cuda.is_available():
            gpu_device = th.device('cuda')
            feature_tensor = feature_tensor.to(gpu_device)
            net = net.to(gpu_device)
        else: 
            print("Cannot use GPU :(")
    
    # get the output from the model
    output, h = net(feature_tensor, h)
    
    # convert output probabilities to predicted class (0 or 1)
    pred = th.round(output.squeeze()) 
    # printing output value, before rounding
    print('Prediction value, pre-rounding: {:.6f}'.format(output.item()))
    
    # print custom response
    if(pred.item()==1):
        print("SPAM detected!")
    else:
        print("No spam message")
        
    return pred.item()



def to_device(obj, device = 'cpu', is_model = True):
    """
    obj: pytorch model or data
    device: string 
        device to send model or data
    is_model: Boolean 
        Indicates if model is passed
    return data or model
    """

    if device.startswith('cuda'):
        assert th.cuda.is_available(), 'GPU computation is not available'

        if is_model:
            if str(next(obj.parameters()).device).startswith(device):
                return obj
        
        else:

            if str(obj.device).startswith(device):
                return obj

        obj = obj.to(  th.device(device)    )
        return obj

    

    if is_model:
        if str(next(obj.parameters()).device).startswith(device):
                return obj
        
        else:

            if str(obj.device).startswith(device):
                return obj

    obj = obj.to(  th.device(device)    )

    return obj
    

    
    

        


def predict_encrypted(net, message,workers,use_gpu = True, sequence_length=200):
    """
    net: pytorch RNN model. Whiout encryption

    message: string
        message to classify on plain text
    
    workers: list 
        List of virtual workers
    
    use_gpu: Boolean 
        if True use GPU computation

    sequence_length: int
        Length of sequences
    

    
    """

    if len(message) == 0:
        return None 
    
    
    net.eval()
    
    # tokenize message
    test_ints = tokenize_message(message,word_to_index)
    
    # pad tokenized sequence
    seq_length=sequence_length
    features = pad_features(test_ints, seq_length)
    
    # convert to tensor to pass into your model
    feature_tensor = th.from_numpy(features)

    # cast tensor to int64
    feature_tensor = feature_tensor.to(th.int64)
    
    batch_size = feature_tensor.size(0)
    
    
    if(use_gpu == True):
        feature_tensor = to_device(feature_tensor, 'cuda', is_model = False)
        net = to_device(net,'cuda', is_model = True)
    else: # As default is cuda.FloatTensor, move to cpu
        feature_tensor = to_device(feature_tensor, 'cuda', is_model = False)
        net = to_device(net,'cuda', is_model = True)



    # Share with workers
    
    # initialize hidden state
    encrypted_h = net.init_hidden_encrypted(batch_size, workers, use_gpu = use_gpu)
    encrypted_feature_tensor = feature_tensor.fix_precision().share(*workers)
    encrypted_net = net.fix_precision().share(*workers)

    print("encrypted_feature_tensor, encrypted_h ", encrypted_feature_tensor, encrypted_h)

    # get the output from the model
    encrypted_output, encrypted_h = encrypted_net(encrypted_feature_tensor, encrypted_h)
    
    # convert output probabilities to predicted class (0 or 1)
    encrypted_pred = th.round(encrypted_output.squeeze()) 

    pred = encrypted_pred.get().float_precision()

    output = encrypted_output.get().float_precision()

    # printing output value, before rounding
    print('Prediction value, pre-rounding: {:.6f}'.format(output.item()))
    
    # print custom response
    if(pred.item()==1):
        print("SPAM detected!")
    else:
        print("No spam message")
        
    return pred.item()





if __name__ == "__main__":

    from rnn_model import SpamRNN
    
    # To use syft with cuda.
    th.set_default_tensor_type(th.cuda.FloatTensor)
    import syft as sy 
    hook = sy.TorchHook(th)   

    # Encryption 
    num_workers = 3 # Number of workers
    workers  = [sy.VirtualWorker(hook, id = "w" + str(i)).add_worker(sy.local_worker) for i in range(num_workers) ]


    model_path = "./models/"
    file_path = model_path + "rnn"
    
    vocab_size = len(index_to_word) 
    print("Len of vocabulary", len(index_to_word.keys()))
    output_size = 1
    embedding_dim = 64 #200
    hidden_dim = 32 #128
    n_layers = 2
    # Load Model
    # Model class must be defined somewhere
    model = SpamRNN(vocab_size, output_size, embedding_dim, hidden_dim, n_layers, train_on_gpu = False, workers = workers)
    model.load_state_dict(th.load(file_path))
    model.eval()

    message = "Purchase now! Limited offer. Click here. Best deal, never returns alert"

    #c = predict(net = model , message = message , use_gpu =True , sequence_length = 200)
    #print("Class predicted", c)

    c_encrypted = predict_encrypted(net = model , message = message , workers = workers , use_gpu = True, sequence_length = 200  )

    print("Spam or Not?: {}".format( "Yes" if c == 1.0 else "No" ))
