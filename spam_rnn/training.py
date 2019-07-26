from preprocess import tokenize_message,pad_features,csv_to_matrix,split_dataset
from rnn_model import SpamRNN
 
from config import cfg
import torch as th
# To use syft with cuda.
th.set_default_tensor_type(th.cuda.FloatTensor) 
import syft as sy 
hook = sy.TorchHook(th)
import torch.nn as nn
import torch.optim as optim 
from torch.utils.data import TensorDataset, DataLoader

import numpy as np
import os 


# Load dictionaries 
#index_to_word= load_obj("data/index_to_word")
#word_to_index = load_obj("data/word_to_index")

FNAME = "../data/spam.csv" #training set file
MIN_COUNT = 5
SEQ_LENGTH = 200
SPLIT_FRAC = 0.8
BATCH_SIZE = 32


# Get matrix of features.
word_to_index, index_to_word, matrix,labels = csv_to_matrix(fname=FNAME, min_count = MIN_COUNT,seq_length= SEQ_LENGTH)

# Get loaders.
train_loader, valid_loader, test_loader, tensor_train, tensor_validation, tensor_test = split_dataset(matrix,labels, split_frac = SPLIT_FRAC, batch_size = BATCH_SIZE)
# MODEL PARAMETERS
# Instantiate the model w/ hyperparams
vocab_size = len(index_to_word) 
output_size = 1
embedding_dim = 64 #200
hidden_dim = 32 #128
n_layers = 2

DEVICE = th.device( 'cuda' if th.cuda.is_available()  else 'cpu')


# First checking if GPU is available
train_on_gpu=th.cuda.is_available()
print("Train on GPU", train_on_gpu)


# TRAINING PARAMS 
epochs =1 # 3-4 is approx where I noticed the validation loss stop decreasing

print_every = 100
clip=5 # gradient clipping
lr=0.001
counter = 0
print_every = 100
clip=5 # gradient clipping



# Encryption 
num_workers = 3 # Number of workers
workers  = [sy.VirtualWorker(hook, id = "w" + str(i)).add_worker(sy.local_worker) for i in range(num_workers) ]

# Make RNN
net = SpamRNN(vocab_size, output_size, embedding_dim, hidden_dim, n_layers, train_on_gpu, workers = workers)

print("Network architecture: \n",net)


def train_model(net, epochs = 5, print_every = 100, lr = 0.001, clip = 5, train_on_gpu = True) :
    """
    net: pytoch model
        RNN

    """
    

    #print("Model device", next(net.parameters()).device)
    net.crypto_testing = False

    # move model to GPU, if available
    if(train_on_gpu):
        net = net.to(DEVICE)

    print("Model device", next(net.parameters()).device)
    

        
    # loss and optimization functions
    criterion = nn.BCELoss()
    optimizer = th.optim.Adam(net.parameters(), lr=lr)



    net.train()

    # train for some number of epochs

    counter = 0
    for e in range(epochs):

        # batch loop
        for inputs, labels in train_loader:
            counter += 1

            
            # If only one label, convert to int then to tensor of shape ()
            if labels.shape[0] == 1:
                labels= th.tensor(labels.item())


            if(train_on_gpu):
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

            # initialize hidden state
            # Variable batch size for len of dataset  not divisible by batch_size
            current_batch_size = inputs.shape[0] 
            h = net.init_hidden(current_batch_size)
            

            # Creating new variables for the hidden state, otherwise
            # we'd backprop through the entire training history
            h = tuple([each.data for each in h])

            # zero accumulated gradients
            net.zero_grad()
            
            #print("h shape", h[0].shape)
            # get the output from the model
            output, h = net(inputs, h)

            # calculate the loss and perform backprop
            loss = criterion(output.squeeze(), labels.float())
            loss.backward()
            # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
            nn.utils.clip_grad_norm_(net.parameters(), clip)
            optimizer.step()

            # loss stats
            if counter % print_every == 0:
                # Get validation loss
                #val_h = net.init_hidden(batch_size)
                val_losses = []
                net.eval()
                for inputs, labels in valid_loader:
                    # initialize hidden state
                    # Variable batch size for len of  dataset  not divisible by batch_size
                    current_batch_size = inputs.shape[0] 
                    val_h = net.init_hidden(current_batch_size)

                    # Creating new variables for the hidden state, otherwise
                    # we'd backprop through the entire training history
                    val_h = tuple([each.data for each in val_h])

                    
                    
                    
                    # If only one label, convert to int then to tensor of shape ()
                    if labels.shape[0] == 1:
                        labels= th.tensor(labels.item())
                    
                    if(train_on_gpu):
                        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                    


                    output, val_h = net(inputs, val_h)
                    val_loss = criterion(output.squeeze(), labels.float())

                    val_losses.append(val_loss.item())

                net.train()
                print("Epoch: {}/{}...".format(e+1, epochs),
                    "Step: {}...".format(counter),
                    "Loss: {:.6f}...".format(loss.item()),
                    "Val Loss: {:.6f}".format(np.mean(val_losses)))




# TESTING 

def test_model(net,test_loader, train_on_gpu):
    """
    net: pytoch model
        RNN
    test_loader:pytorch  Data Loader

    """

    # Get test data loss and accuracy

    test_losses = [] # track loss
    num_correct = 0

    # First checking if GPU is available
    train_on_gpu=th.cuda.is_available()

     # loss 
    criterion = nn.BCELoss()

    net.eval()
    # iterate over test data
    for inputs, labels in test_loader:
        
        
        # initialize hidden state
        # Variable batch size for len of dataset  not divisible by batch_size
        current_batch_size = inputs.shape[0] 
        h = net.init_hidden(current_batch_size)

       


        # Creating new variables for the hidden state, otherwise
        # we'd backprop through the entire training history
        h = tuple([each.data for each in h])

        if(train_on_gpu):
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        
        # get predicted outputs
        output, h = net(inputs, h)
        
        # calculate loss
        test_loss = criterion(output.squeeze(), labels.float())
        test_losses.append(test_loss.item())
        
        # convert output probabilities to predicted class (0 or 1)
        pred = th.round(output.squeeze())  # rounds to the nearest integer
        
        # compare predictions to true label
        correct_tensor = pred.eq(labels.float().view_as(pred))
        correct = np.squeeze(correct_tensor.numpy()) if not train_on_gpu else np.squeeze(correct_tensor.cpu().numpy())
        num_correct += np.sum(correct)


    # -- stats! -- ##
    # avg test loss
    mean_loss = np.mean(test_losses)
    print("Test loss: {:.3f}".format(mean_loss))

    # accuracy over all test data
    test_acc = num_correct/len(test_loader.dataset)
    print("Test accuracy: {:.3f}".format(test_acc))


    return mean_loss,  test_acc


# TESTING 

def test_model_encrypted(net,test_loader, train_on_gpu, workers):
    """
    net: pytoch model
        RNN
    test_loader:pytorch  Data Loader

    """

    #for param in iter(net.embedding.parameters()):
    #    param = param.share(*workers)
    net.crypto_testing = True
   
    # Get test data loss and accuracy
    test_losses = [] # track loss
    num_correct = 0
 
     # loss 
    criterion = nn.BCELoss()

    net.eval()
    # iterate over test data
    for inputs, labels in test_loader:
        # initialize hidden state
        # Variable batch size for len of dataset  not divisible by batch_size
        current_batch_size = inputs.shape[0] 
        h = net.init_hidden_encrypted(current_batch_size,workers)

        # Creating new variables for the hidden state, otherwise
        # we'd backprop through the entire training history
        h = tuple([each.clone().get().data.share(*workers) for each in h ])
       

        if(train_on_gpu):
            inputs, labels = encrypt_data(inputs.to(DEVICE), workers),encrypt_data(labels.to(DEVICE), workers)
        
        
        # get predicted outputs
        output, h = net(inputs, h)
                
        # calculate loss
        test_loss = criterion(output.squeeze(), labels.float())
        test_losses.append(test_loss.item())
        
        # convert output probabilities to predicted class (0 or 1)
        pred = th.round(output.squeeze())  # rounds to the nearest integer
        
        # compare predictions to true label
        correct_tensor = pred.eq(labels.float().view_as(pred))
        correct = np.squeeze(correct_tensor.numpy()) if not train_on_gpu else np.squeeze(correct_tensor.cpu().numpy())
        num_correct += np.sum(correct)


    # -- stats! -- ##
    # avg test loss
    mean_loss = np.mean(test_losses)
    print("Test loss: {:.3f}".format(mean_loss))

    # accuracy over all test data
    test_acc = num_correct/len(test_loader.dataset)
    print("Test accuracy: {:.3f}".format(test_acc))


    return mean_loss,  test_acc

def encrypt_data(data, workers ):
    """
    data: torch tensor
    workers: list
        List of virtual workers
    """

   
    encrypted_data = data.share(*workers)

   
    
    return encrypted_data


def encrypt_model(model, workers):
    """
    model: pytorch model
    data: torch tensor
    workers: list 
        List of virtual workers
    """

    encrypted_model = model.share(*workers)

    print("Encrypted model parameters", list(encrypted_model.parameters()))

    return encrypted_model


def encrypt_prediction(text):
    """
    text: string 
        message to classify
    return 1 for spam 0 for no spam
    """
    





if __name__ == "__main__":
    print("Version of syft", sy.__version__)
    
    train_model(net, epochs =epochs , print_every = print_every, lr = lr, clip = clip)

    
    # Test Net 
    test_model(net,test_loader, train_on_gpu = True)
    

    # Test with encryption.
    #test_model_encrypted(encrypted_net,test_loader, train_on_gpu = True, workers = workers)
    #encrypted_net = encrypt_model(net, workers)
   
    # Save model

    model_path = "./models/"
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    file_path = model_path + "rnn"
        
    th.save(net.state_dict(),file_path)
