from preprocess import *
from rnn_model import SpamRNN
import torch as th 
from config import cfg
import torch.nn as nn

# Load dictionaries 
#index_to_word= load_obj("data/index_to_word")
#word_to_index = load_obj("data/word_to_index")

FNAME = "../data/spam.csv" #training set file
MIN_COUNT = 5
SEQ_LENGTH = 200
SPLIT_FRAC = 0.8


# Get matrix of features.
word_to_index, index_to_word, matrix,labels = csv_to_matrix(fname=FNAME, min_count = MIN_COUNT,seq_length= SEQ_LENGTH)

# Get loaders.
train_loader, valid_loader, test_loader = split_dataset(matrix,labels, split_frac = SPLIT_FRAC)

# MODEL PARAMETERS
# Instantiate the model w/ hyperparams
vocab_size = len(index_to_word) 
output_size = 1
embedding_dim = 64 #200
hidden_dim = 32 #128
n_layers = 2


# Make RNN
net = SpamRNN(vocab_size, output_size, embedding_dim, hidden_dim, n_layers)

print("Network architecture: \n",net)

# TRAINING PARAMS 
epochs =8 # 3-4 is approx where I noticed the validation loss stop decreasing

print_every = 100
clip=5 # gradient clipping
lr=0.001
counter = 0
print_every = 100
clip=5 # gradient clipping




def train_model(net, epochs = 5, print_every = 100, lr = 0.001, clip = 5) :
    """
    net: pytoch model
        RNN

    """
    # First checking if GPU is available
    train_on_gpu=th.cuda.is_available()

    # move model to GPU, if available
    if(train_on_gpu):
        net = net.cuda()


        
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

            if(train_on_gpu):
                inputs, labels = inputs.cuda(), labels.cuda()




            # If only one label, convert to int then to tensor of shape ()
            if labels.shape[0] == 1:
                labels= th.tensor(labels.item())

                if(train_on_gpu):
                    labels = labels.cuda()

                
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

                    if(train_on_gpu):
                        inputs, labels = inputs.cuda(), labels.cuda()
                    
                    
                    
                    # If only one label, convert to int then to tensor of shape ()
                    if labels.shape[0] == 1:
                        labels= th.tensor(labels.item())
                        if(train_on_gpu):
                            labels = labels.cuda()
                        
                    output, val_h = net(inputs, val_h)
                    val_loss = criterion(output.squeeze(), labels.float())

                    val_losses.append(val_loss.item())

                net.train()
                print("Epoch: {}/{}...".format(e+1, epochs),
                    "Step: {}...".format(counter),
                    "Loss: {:.6f}...".format(loss.item()),
                    "Val Loss: {:.6f}".format(np.mean(val_losses)))




# TESTING 




def test_model(net):
    """
    net: pytoch model
        RNN

    """

    # Get test data loss and accuracy

    test_losses = [] # track loss
    num_correct = 0



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
            inputs, labels = inputs.cuda(), labels.cuda()
        
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




if __name__ == "__main__":
    train_model(net, epochs =epochs , print_every = print_every, lr = lr, clip = clip)

