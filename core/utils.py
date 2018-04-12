import numpy as np
import cPickle as pickle
import hickle
import time
import os
import torch
from torch.autograd import Variable


def load_coco_data(data_path='./data', split='train'):
    data_path = os.path.join(data_path, split)
    start_t = time.time()
    data = {}
  
    data['features'] = hickle.load(os.path.join(data_path, '%s.features.hkl' %split))
    with open(os.path.join(data_path, '%s.file.names.pkl' %split), 'rb') as f:
        data['file_names'] = pickle.load(f)   
    with open(os.path.join(data_path, '%s.captions.pkl' %split), 'rb') as f:
        data['captions'] = pickle.load(f)
    with open(os.path.join(data_path, '%s.image.idxs.pkl' %split), 'rb') as f:
        data['image_idxs'] = pickle.load(f)
            
    if split == 'train':       
        with open(os.path.join(data_path, 'word_to_idx.pkl'), 'rb') as f:
            data['word_to_idx'] = pickle.load(f)
          
    for k, v in data.iteritems():
        if type(v) == np.ndarray:
            print k, type(v), v.shape, v.dtype
        else:
            print k, type(v), len(v)
    end_t = time.time()
    print "Elapse time: %.2f" %(end_t - start_t)
    return data

def decode_captions(captions, idx_to_word):
    if captions.ndim == 1:
        T = captions.shape[0]
        N = 1
    else:
        N, T = captions.shape

    decoded = []
    for i in range(N):
        words = []
        for t in range(T):
            if captions.ndim == 1:
                word = idx_to_word[captions[t]]
            else:
                word = idx_to_word[captions[i, t]]
            if word == '<END>':
                words.append('.')
                break
            if word != '<NULL>':
                words.append(word)
        decoded.append(' '.join(words))
    return decoded


def write_bleu(scores, path, epoch):
    if epoch == 0:
        file_mode = 'w'
    else:
        file_mode = 'a'
    with open(os.path.join(path, 'val.bleu.scores.txt'), file_mode) as f:
        f.write('Epoch %d\n' %(epoch+1))
        f.write('Bleu_1: %f\n' %scores['Bleu_1'])
        f.write('Bleu_2: %f\n' %scores['Bleu_2'])
        f.write('Bleu_3: %f\n' %scores['Bleu_3'])  
        f.write('Bleu_4: %f\n' %scores['Bleu_4']) 
        f.write('METEOR: %f\n' %scores['METEOR'])  
        f.write('ROUGE_L: %f\n' %scores['ROUGE_L'])  
        f.write('CIDEr: %f\n\n' %scores['CIDEr'])

def load_pickle(path):
    with open(path, 'rb') as f:
        file = pickle.load(f)
        print ('Loaded %s..' %path)
        return file  

def save_pickle(data, path):
    with open(path, 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
        print ('Saved %s..' %path)

def get_batch_train(t,batch_size,features,caption_in,caption_out,image_idxs):
    captions_batch_in = Variable(torch.from_numpy(caption_in[t*batch_size:(t +1)*batch_size]).type(torch.LongTensor)).cuda()
    captions_batch_out = torch.from_numpy(caption_out[t*batch_size:(t +1)*batch_size])
    mask = np.not_equal(captions_batch_out, 0)
    mask = Variable(mask.type(torch.cuda.FloatTensor))
    captions_batch_out = Variable(captions_batch_out.type(torch.LongTensor)).cuda()
    image_idxs_batch = image_idxs[t*batch_size:(t +1)*batch_size]
    features_batch = Variable(torch.from_numpy(features[image_idxs_batch])).cuda()
    return captions_batch_in, captions_batch_out,features_batch,mask

def sample_minibatch(data, batch_size):
    rand_idxs = np.random.permutation(data['captions'].shape[0])
    mask = np.random.choice(data['captions'].shape[0], batch_size)
    captions = data['captions'][rand_idxs][mask]
    image_idxs = data['image_idxs'][rand_idxs][mask]
    features = data['features'][image_idxs]
    file_names = data['file_names'][image_idxs]
    return features, file_names, captions

