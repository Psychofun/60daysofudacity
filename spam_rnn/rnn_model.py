

import torch as th
# To use syft with cuda.
"""th.set_default_tensor_type(th.cuda.FloatTensor)
import syft as sy 
hook = sy.TorchHook(th)"""
import torch.nn as nn

class SpamRNN(nn.Module):
    """
    The RNN model that will be used to perform Spam Detection.
    """

    def __init__(self, vocab_size, output_size, embedding_dim, hidden_dim, n_layers,train_on_gpu,drop_prob=0.5):
        """
        Initialize the model by setting up the layers.
        """
        super(SpamRNN, self).__init__()

        self.output_size = output_size
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.train_on_gpu = train_on_gpu
        self.crypto_testing = True
        #self.workers = workers
        
        # embedding and LSTM layers
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers, 
                            dropout=drop_prob,
                            batch_first=True
                           )
        
        # dropout layer
        self.dropout = nn.Dropout(0.3)
        
        # linear and sigmoid layers
        self.fc = nn.Linear(hidden_dim, output_size)
        self.sig = nn.Sigmoid()
        

    def forward(self, x, hidden):
        """
        Perform a forward pass of our model on some input and hidden state.
        """
        batch_size = x.size(0)
        #print("EMBEDDING PARAMETERS",list(self.embedding.parameters() ))
        #print("EMBEDDING PARAMETERS",self.embedding.weight.clone().get() )

        
        # embeddings and lstm_out
        embeds = self.embedding(x)
        lstm_out, hidden = self.lstm(embeds, hidden)

       
        # stack up lstm outputs
        lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim)

        
        # dropout and fully-connected layer
        out = self.dropout(lstm_out)

       
        out = self.fc(out)

        # sigmoid function

        sig_out = self.sig(out)
        
        # reshape to be batch_size first
        sig_out = sig_out.view(batch_size, -1)
        sig_out = sig_out[:, -1] # get last batch of labels
        
        
        # return last sigmoid output and hidden state
        return sig_out, hidden

        
    def init_hidden(self, batch_size):
        ''' Initializes hidden state '''
        # Create two new tensors with sizes n_layers x batch_size x hidden_dim,
        # initialized to zero, for hidden state and cell state of LSTM
        weight = next(self.parameters()).data
        

        if self.train_on_gpu:
            device_str = 'cuda' if self.train_on_gpu else  'cpu'
            device = th.device(device_str)
            if device_str == 'cpu':
                self.train_on_gpu = False
                print("Cannot train on GPU.")


        if (self.train_on_gpu):
            hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device),
                  weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device))
        else:
            hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_(),
                      weight.new(self.n_layers, batch_size, self.hidden_dim).zero_())
        
        return hidden
    
    
    def init_hidden_encrypted(self, batch_size, workers, use_gpu ):
        ''' Initializes hidden state '''
        # Create two new tensors with sizes n_layers x batch_size x hidden_dim,
        # initialized to zero, for hidden state and cell state of LSTM
        weight = next(self.parameters()).data
  
        device_str =  'cuda' if use_gpu == True  else  'cpu'
        device = th.device(device_str)
        print(device_str)

        if str(weight.device).startswith(device_str):
            hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().fix_precision().share(*workers),
                      weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().fix_precision().share(*workers))
        
        else:
            hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().fix_precision().share(*workers).to(device),
                      weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().fix_precision().share(*workers).to(device))   

       
        return hidden
    
if __name__ == "__main__":
    pass
   
   
    
   