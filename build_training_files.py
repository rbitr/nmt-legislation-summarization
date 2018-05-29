import re
from nltk.tokenize import word_tokenize
import string
import numpy as np

#porter = PorterStemmer()

batch_size = 64  # Batch size for training.
epochs = 25  # Number of epochs to train for.
latent_dim = 256  # Latent dimensionality of the encoding space.
num_samples = 10000  # Number of samples to train on.
# Path to the data txt file on disk.
data_path = 'data/'

#minumum to be included in dictionary
min_mentions = 2


#table = str.maketrans('', '', string.punctuation)
table = str.maketrans('', '', "".join(string.punctuation.split('.')))

r = re.compile("[A-Za-z']+")
# Vectorize the data.
input_texts = []
target_texts = []

#data_tokens_in_throw = set()
#data_tokens_out_throw = set()

data_tokens_in = set()
data_tokens_out = set()

data_tokens_in_dict = dict()
data_tokens_out_dict = dict()


#arget_tokens = set()


# should match now...
with open(data_path+"/actsT2.txt") as f:
    all_titles = [line for line in f]
    #all_titles = [' '.join(r.findall(line)) for line in f]
    #all_titles = [' '.join(line.split()) for line in f]
#print(len(all_titles))    

with open(data_path+"/actsS2.txt") as f:
    all_summaries = [line for line in f]
    #all_summaries = [' '.join(r.findall(line)) for line in f]
    #all_summaries = [' '.join(line.split()) for line in f]
#print(len(all_summaries))  

# worry about this later
#toppers = [i[0] for i in Counter(all_titles).most_common(20)]


#open a test, train, dev file
train_in = open(data_path+"/train.in", 'w') 
train_out = open(data_path+"/train.out", 'w') 
dev_in = open(data_path+"/dev.in", 'w') 
dev_out = open(data_path+"/dev.out", 'w') 
test_in = open(data_path+"/test.in", 'w') 
test_out = open(data_path+"/test.out", 'w') 


for input_text, target_text in zip(all_summaries,all_titles):
    #input_text = input_text.replace('.', ' ')
    in_tokens = word_tokenize(input_text)
    in_tokens = [w.lower() for w in in_tokens]

    stripped = [w.translate(table) for w in in_tokens]
    # remove remaining tokens that are not alphabetic
    words = [word for word in stripped if word.isalpha() or word=="."]
    words_in = words[:200]
    #stemmed_in = [porter.stem(word) for word in words]
    for token in words_in:
        if token in data_tokens_in_dict.keys():
            data_tokens_in_dict[token]+=1
        else:
            data_tokens_in_dict[token]=1
        #if token in data_tokens_in_throw:
        #    data_tokens_in.add(token)
        #else:
        #    data_tokens_in_throw.add(token)

        if token==('afteracquired'):
            print(input_text)
    input_texts.append(' '.join(words_in))
    
    #target_text = target_text.replace('.', ' ')
    targ_tokens = word_tokenize(target_text)
    targ_tokens = [w.lower() for w in targ_tokens]

    stripped = [w.translate(table) for w in targ_tokens]
    # remove remaining tokens that are not alphabetic
    words = [word for word in stripped if word.isalpha() or word=="."]
    words_out = words[:50]
    #stemmed_out = [porter.stem(word) for word in words]
    for token in words_out:
        if token in data_tokens_out_dict.keys():
            data_tokens_out_dict[token]+=1
        else:
            data_tokens_out_dict[token]=1
        #if token in data_tokens_out_throw:
        #    data_tokens_out.add(token)
        #else:
        #    data_tokens_out_throw.add(token)
        if token==('aadministr'):
            print(target_text)
    target_text = ' '.join(words_out)
    #target_text = '\t' + ' '.join(stemmed) + '\n'
    target_texts.append(target_text)

    rn = np.random.uniform()
    target_text = target_text + '\n'
    input_text = ' '.join(words_in)+ '\n'
    
    
    if rn < .167:
        test_in.write(input_text)
        test_out.write(target_text)
    elif rn < .333:
        dev_in.write(input_text)
        dev_out.write(target_text)
    else:
        train_in.write(input_text)
        train_out.write(target_text)

test_in.close()
test_out.close()
dev_in.close()
dev_out.close()
train_in.close()
train_out.close()



# build the tokens lists
for k in data_tokens_in_dict.keys():
    if data_tokens_in_dict[k]>=min_mentions:
        data_tokens_in.add(k)

for k in data_tokens_out_dict.keys():
    if data_tokens_out_dict[k]>=min_mentions:
        data_tokens_out.add(k)    



data_tokens_in = sorted(list(data_tokens_in))
data_tokens_out = sorted(list(data_tokens_out))


vocab = open(data_path+"/vocab.in", 'w') 
for dt in data_tokens_in:
    vocab.write(dt+'\n')
vocab.close()

vocab = open(data_path+"/vocab.out", 'w') 
for dt in data_tokens_out:
    vocab.write(dt+'\n')
vocab.close()

#target_characters = sorted(list(target_characters))
num_tokens_in = len(data_tokens_in)
num_tokens_out = len(data_tokens_out)
#num_decoder_tokens = len(target_characters)
#max_encoder_seq_length = max([len(txt) for txt in input_texts])
#max_decoder_seq_length = max([len(txt) for txt in target_texts])

print('Number of samples:', len(input_texts))
print('Number of unique tokens - inouts:', num_tokens_in)
print('Number of unique tokens - outputs:', num_tokens_out)
