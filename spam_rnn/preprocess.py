import numpy as np
from numpy import  genfromtxt
import os
import pickle
import torch as th
# To use syft with cuda.
th.set_default_tensor_type(th.cuda.FloatTensor)
from collections import Counter
from torch.utils.data import TensorDataset, DataLoader
UNK_TOKEN = 'UNK'
from string import punctuation


def save_obj(obj,path ):
    direct, fname = os.path.split(path)

    if not os.path.exists(direct):
        os.makedirs(direct)

    with open(path + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(path):
    with open(path  + '.pkl', 'rb') as f:
        return pickle.load(f)


def get_word(index_to_word, index):
    """
    index_to_word: dictionary
        index to word dict
    index: int

    return word given index. If index (key) not in dict returns 'UNK' unknow token
    """

    result = index_to_word.get(index,None)

    if result:
        return result
    return UNK_TOKEN


def get_index(word_to_index, word):
    """
    word_to_index: dictionary
        word to index dict
    word: string
    return index of the word from word_to_index
    if word not in word_to_index return 0, index of unknow token.

    """

    result = word_to_index.get(word,None)

    if result:
        return result
    return 0



def pad_features(vectors, seq_length):
    ''' Return features of vectors, where each review is padded with 0's
        or truncated to the input seq_length.
    '''

    # getting the correct rows x cols shape
    features = np.zeros((len(vectors), seq_length), dtype=int)

    # for each review, I grab that review and
    for i, row in enumerate(vectors):
        features[i, -len(row):] = np.array(row)[:seq_length]

    return features


def csv_to_matrix(fname, min_count,seq_length=200):
    """
    fname: string
        path to csv file.
    min_count: int
        minumum count

    Convert csv to matrix of word embeddings
    """


    data = genfromtxt(fname,delimiter = '\n', dtype='str')
    #Throw first row. Column names
    data = data[1:]

    # Separate into messages and labels
    labels,messages =zip(*list(map(
            lambda x: (x[:3]  , x[4:-3]) if x.startswith('h') else (x[:4],x[5:-3])

                      ,data)))

    labels = np.array(labels)
    messages = np.array(messages)

    # Labels to ints
    labels[labels == "spam"] = 1.0
    labels[labels == "ham"] = 0.0

    #convert to float
    labels = labels.astype('float')
    #print(labels[:10])



    from string import punctuation

    for k in range(messages.shape[0]):
        messages[k] = messages[k].lower()
        messages[k] = "".join( [s for s in   messages[k] if s not in punctuation])

    all_messages="".join( [s.lower() for s in messages if s not in punctuation] )

    words = all_messages.split()

    ## Build a dictionary that maps words to integers
    #Count words
    word_count ={}
    for word in words:
        r = word_count.get(word,None)

        if r :
            word_count[word]+=1
        else:
            word_count[word] = 1



    #word to index
    word_to_index = {}

    keys = word_count.keys()
    # Begin indexing with 1
    i= 1
    for key in  keys:
        #Only use words with minimum count appearences.
        if word_count[key] >= min_count:
            word_to_index[key] = i
            i+= 1

    ### Add Unknow token
    word_to_index[UNK_TOKEN] = 0


    ###print(len(word_to_index.keys() ))
    ###word_to_index


    index_to_word = {}

    for key in word_to_index.keys():
        index_to_word[ word_to_index[key] ] =key


    # Save Dictionaries to file.
    save_obj(index_to_word, "data/index_to_word")
    save_obj(word_to_index, "data/word_to_index")

    # Messages to vectors

    vectors = []

    for message in messages:

        vector = [ get_index(word_to_index,w) for w in message.split()]
        vectors.extend([vector])

    # Remove Outliers
    #outliers messages stats
    messages_lens = Counter([len(x)  for x in vectors])
    print("Zero-length messages: {}".format(messages_lens[0]))
    print("Maximum message length: {}".format(max(messages_lens)))
    print()

    print("Number of messages before removing outliers: ", len(vectors))

    non_zero_idx = [i for i,message in enumerate(vectors) if len(message)!= 0 ]

    #remove 0 length messages end their labels
    vectors = [vectors[i] for i in non_zero_idx ]
    labels = np.array([labels[i] for i in non_zero_idx])

    print("Number of messages after removing outliers: ",len(vectors))



    assert len(vectors) == labels.shape[0], "Number of vectors differ with number of labels :'("

    #seq_length = 200
    features = pad_features(vectors, seq_length=seq_length)

    ## test statements - do not change - ##
    assert len(features)==len(vectors), "Your features should have as many rows as vectors."
    assert len(features[0])==seq_length, "Each feature row should contain seq_length values."



    # print first 200 values of the first 10 batches
    #print(features[:10,:200])


    return word_to_index, index_to_word, features,labels

def split_dataset(features,labels,split_frac = 0.8,batch_size = 32):
    """
    features: numpy array
        Shape of (num of examples) x (num of features/sequence length)
    split_frac: float
        percentage for training set, remainig goes to validation 50% and test set 50%.

    return data loaders and  raw tensors for train, validation and test
    train_loader, valid_loader, test_loader, tensor_train, tensor_validation, tensor_test
    where tensor_* is a tuple (X (features), y (labels))
    """


    ## split data into training, validation, and test data (features and labels, x and y)

    split_idx = int(len(features)*0.8)
    train_x, remaining_x = features[:split_idx], features[split_idx:]
    train_y, remaining_y = labels[:split_idx], labels[split_idx:]

    test_idx = int(len(remaining_x)*0.5)
    val_x, test_x = remaining_x[:test_idx], remaining_x[test_idx:]
    val_y, test_y = remaining_y[:test_idx], remaining_y[test_idx:]

    ## print out the shapes of your resultant feature data
    print("\t\t\tFeature Shapes:")
    print("Train set: \t\t{}".format(train_x.shape),
        "\nValidation set: \t{}".format(val_x.shape),
        "\nTest set: \t\t{}".format(test_x.shape))


    # CONVERT TO int64 for embedding layer.
    tensor_train = (th.from_numpy(train_x).to(th.int64),   th.from_numpy(train_y))
    tensor_validation = (th.from_numpy(val_x).to(th.int64),th.from_numpy(val_y) )
    tensor_test = (th.from_numpy(test_x).to(th.int64),     th.from_numpy(test_y) )


    # Create Tensor datasets
    train_data = TensorDataset( tensor_train[0].clone(),tensor_train[1].clone() )
    valid_data = TensorDataset(tensor_validation[0].clone(), tensor_validation[1].clone() )
    test_data = TensorDataset(tensor_test[0].clone(), tensor_test[1].clone())



    # make sure the SHUFFLE your training data
    train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
    valid_loader = DataLoader(valid_data, shuffle=True, batch_size=batch_size)
    test_loader = DataLoader(test_data, shuffle=True, batch_size=batch_size)

    return train_loader, valid_loader, test_loader, tensor_train, tensor_validation, tensor_test




def tokenize_message(message, word_to_index):
    message = message.lower() # lowercase
    # get rid of punctuation
    test_text = ''.join([c for c in message if c not in punctuation])

    # splitting by spaces
    test_words = test_text.split()

    # tokens
    test_ints = []
    test_ints.append([get_index(word_to_index,word) for word in test_words])

    return test_ints



