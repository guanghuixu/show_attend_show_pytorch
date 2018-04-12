import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
from utils import save_pickle
from utils import load_pickle
from utils import load_coco_data
import os
import numpy as np
from utils import get_batch_train
from utils import sample_minibatch
from utils import decode_captions
# import matplotlib.pyplot as plt
# from scipy import ndimage
# import skimage.transform
from bleu import evaluate

class ATTENDModel(nn.Module):
    def __init__(self, args,vocab_size):
        super(ATTENDModel, self).__init__()
        self.batch_size=args.batch_size
        self.num_layers = args.nlayers
        self.time_step = args.time_step
        self.vocab_size = vocab_size #23110
        self.embedding_size = args.embedding_size #512
        self.hidden_dim = args.hidden_dim #1024
        self.word_embeds = nn.Embedding(vocab_size, args.embedding_size)
        self.lstm = nn.LSTMCell(self.hidden_dim , self.hidden_dim)
        self.linear1 = nn.Linear(self.embedding_size, self.embedding_size)  # features_proj
        self.linear2 = nn.Linear(self.embedding_size, self.embedding_size)  # hidden_features
        self.linear3 = nn.Linear(self.embedding_size, 1)  # out_att
        self.linear4 = nn.Linear(self.embedding_size, self.hidden_dim)  # h
        self.linear5 = nn.Linear(self.embedding_size, self.hidden_dim)  # c
        self.linear6 = nn.Linear(self.hidden_dim, 1)  # beta
        self.linear7 = nn.Linear(self.hidden_dim, self.embedding_size)  # h_logits
        self.linear8 = nn.Linear(self.embedding_size, self.embedding_size)  # context
        self.linear9 = nn.Linear(self.embedding_size, self.vocab_size)
        self.softmax = nn.Softmax()
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(p=args.dropout)
        self.simgoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.batch_norm = nn.BatchNorm2d(args.batch_norm)
        self.loss_fn = nn.CrossEntropyLoss()

    def _init_hidden(self, features):
        features_mean = torch.mean(features, 1)
        h = self.tanh(self.linear4(features_mean))
        c = self.tanh(self.linear5(features_mean))
        return h, c

    def _init_beta(self, h):
        beta = self.simgoid(self.linear6(h))
        return beta


    def _attend_img(self, features):
        features_proj = self.linear1(features)
        hidden_features = torch.mean(self.linear2(features_proj), 1)
        tmp_size = hidden_features.size()[0]
        hidden_features = self.relu( hidden_features.view(tmp_size, 1, -1))
        h_att = hidden_features + features_proj
        out_att = self.linear3(h_att)
        alpha = self.softmax(out_att)
        tmp_size = alpha.size()
        context = torch.sum(features * (alpha.view(tmp_size[0], tmp_size[1], -1)), 1)
        return context, alpha

    def _attend_text(self, text, h, context):
        h = self.dropout(h)
        h_logits = self.linear7(h) + text + self.linear8(context)
        h_logits = self.dropout(self.tanh(h_logits))
        logits = self.linear9(h_logits)
        return logits  # next_word

    def forward(self, caption_in, caption_out, features, mask):
        loss = 0.0
        alpha_list = []
        caption_in = self.word_embeds(caption_in)
        features = self.batch_norm(features)
        h, c = self._init_hidden(features)

        for t in range(self.time_step):
            context, alpha = self._attend_img(features)
            alpha_list.append(alpha)
            beta = self._init_beta(h)
            context = beta * context
            inputs = torch.cat((caption_in[:, t, :], context), 1)
            tmp_size = inputs.size()
            inputs = inputs.view(tmp_size[0], 1, -1)
            h = h.view(-1,1,self.hidden_dim)
            _, h = self.lstm(inputs, (h, c))
            h = h.view(-1, self.hidden_dim)
            logits = self._attend_text(caption_in[:, t, :], h, context)
            loss += torch.sum(self.loss_fn(logits, caption_out[:, t])*mask[:,t])

        alpha_all = torch.transpose(torch.stack(alpha_list),1,0)
        alpha_all=torch.sum(torch.squeeze(alpha_all),1)
        alpha_reg=torch.sum((self.time_step/196.0-alpha_all)**2)
        loss += alpha_reg
        return loss / self.batch_size

    def build_sample(self, features, max_len=20):
        alpha_list = []
        beta_list = []
        sample_word_list = []
        features = self.batch_norm(features)
        h, c = self._init_hidden(features)
        batch_size = features.size()[0]
        for t in range(max_len):
            if t==0:
                caption_in = self.word_embeds(Variable(torch.ones([batch_size]).
                                                       type(torch.LongTensor)).cuda())
            else:
                caption_in = self.word_embeds(sample_word)

            context, alpha = self._attend_img(features)
            alpha_list.append(alpha)
            beta = self._init_beta(h)
            beta_list.append(beta)
            context = beta * context
            inputs = torch.cat([caption_in, context], 1)
            tmp_size = inputs.size()
            inputs = inputs.view(tmp_size[0], 1, -1)
            h = h.view(-1, 1, self.hidden_dim)
            _, h = self.lstm(inputs, (h, c))
            h = h.view(-1, self.hidden_dim)
            logits = self._attend_text(caption_in, h, context)
            _, sample_word = torch.max(logits,1)
            sample_word_list.append(sample_word)

        alpha_all = torch.transpose(torch.stack(alpha_list), 1, 0)
        betas = torch.transpose(torch.squeeze(torch.stack(beta_list)), 1, 0)
        sample_caption = torch.transpose(torch.stack(sample_word_list), 1, 0)

        return alpha_all,betas,sample_caption


class RUNModel():
    def __init__(self,args,best_val_loss):
        self.args = args
        self.best_val_loss = best_val_loss
        # download the data
        self.train_data = load_coco_data(data_path=self.args.data, split='train')
        self.word_to_idx = self.train_data['word_to_idx']
        self.idx_to_word = {i: w for w, i in self.train_data['word_to_idx'].iteritems()}
        self.vocab_size = len(self.word_to_idx)
        self.val_data = load_coco_data(data_path=self.args.data, split='val')
        self.test_data = load_coco_data(data_path=self.args.data, split='test')
        print '******download train_val data successful *******'

        # build attend model
        self.atten_model = ATTENDModel(self.args,self.vocab_size)
        if os.path.isfile(self.args.save):
            self.atten_model.load_state_dict(torch.load(self.args.save))
        if torch.cuda.is_available():
            self.atten_model.cuda()
        self.optimizer = optim.SGD(self.atten_model.parameters(), lr=self.args.lr, momentum=0.9)


    def random_choice(self):
        features = self.train_data['features']
        n_examples = features.shape[0]
        rand_idxs = np.random.permutation(n_examples)
        captions = self.train_data['captions'][rand_idxs]
        caption_in = captions[:, :16]
        caption_out = captions[:, 1:]
        image_idxs = self.train_data['image_idxs'][rand_idxs]
        return features,n_examples,image_idxs,caption_in,caption_out

    # build train core
    def train(self, epoch):
        self.atten_model.train()

        # prepare the train data
        features, n_examples, image_idxs, caption_in, caption_out =self.random_choice()
        train_times = int(np.ceil(n_examples / self.args.batch_size))
        for t in range(train_times):
            captions_batch_in,captions_batch_out,features_batch,mask = \
                get_batch_train(t,self.args.batch_size,features,caption_in, caption_out,image_idxs)

            loss = self.atten_model(captions_batch_in, captions_batch_out,
                         features_batch, mask)
            print (epoch, t, loss.data[0])
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    # build val core
    def val(self,epoch):
        self.atten_model.eval()

        # prepare val data: random choice a batch, compute the loss and generate sentence
        features_batch, image_files, cur_captions = sample_minibatch(self.val_data, self.args.batch_size)
        features = Variable(torch.from_numpy(features_batch)).cuda()
        caption_in = cur_captions[:, :16]
        caption_out = cur_captions[:, 1:]
        captions_batch_in = Variable(torch.from_numpy(caption_in).type(torch.LongTensor)).cuda()
        captions_batch_out = torch.from_numpy(caption_out)
        mask = np.not_equal(captions_batch_out, 0)
        mask = Variable(mask.type(torch.cuda.FloatTensor))
        captions_batch_out = Variable(captions_batch_out.type(torch.LongTensor)).cuda()

        loss = self.atten_model(captions_batch_in, captions_batch_out,
                                features, mask)

        alpha_all, betas, sample_caption = self.atten_model.build_sample(features)
        decoded = decode_captions(np.squeeze(np.array(sample_caption.data)), self.idx_to_word)
        alpha_all = np.squeeze(np.array(alpha_all.data))
        betas = np.squeeze(np.array(betas.data))
        cur_decoded = decode_captions(np.stack(cur_captions), self.idx_to_word)
        if epoch%(int(self.args.epochs*0.1))==0:
            file_decoded = {image_files[i]: (decoded[i], cur_decoded[i],alpha_all[i],betas[i]) for i in range(self.args.batch_size)}
            val_samples_path = os.path.join(self.args.val_samples,'val-'+str(epoch)+'-samples.pkl')
            save_pickle(file_decoded, val_samples_path)
        val_loss = torch.sum(loss) / self.args.batch_size
        # Save the model if the validation loss is the best we've seen so far.
        if not self.best_val_loss or val_loss.data[0] < self.best_val_loss:
            torch.save(self.atten_model.state_dict(), self.args.save)
            self.best_val_loss = val_loss.data[0]
            save_pickle(self.best_val_loss,self.args.loss_log)
            print 'save train model'
        elif epoch!=0 and epoch%100==0:
           self.args.lr /= 2.0

        return self.args.lr,self.best_val_loss,decoded

    # build test core
    def test(self,save_sampled_captions=True,evaluate_score=True,generate_demo_sample=False):
        self.atten_model.eval()
        self.atten_model.load_state_dict(torch.load(self.args.save))
        self.atten_model.cuda()

        if save_sampled_captions:
            features = self.test_data['features']
            n_examples = features.shape[0]
            all_sam_cap = np.ndarray((n_examples, 20))
            test_times = int(np.ceil(float(n_examples) / self.args.batch_size))
            for t in range(test_times):
                features_batch = Variable(torch.from_numpy(features[t*self.args.batch_size:(t+1)*self.args.batch_size])).cuda()
                _,_,sampled_captions = self.atten_model.build_sample(features_batch)
                all_sam_cap[t*self.args.batch_size:(t+1)*self.args.batch_size]=np.array(sampled_captions.data)
            decoded = decode_captions(all_sam_cap, self.idx_to_word)
            save_pickle(decoded, self.args.test_samples)
            print 'test all sccessful'

        if evaluate_score:
            ref = load_pickle('./data/test/test.references.pkl')
            try:
                evaluate(ref, decoded)
            except KeyboardInterrupt:
                decoded = load_pickle(self.args.test_samples)
                evaluate(ref, decoded)

        if generate_demo_sample:
            features = self.args.demo_feat
            features_batch = Variable(torch.from_numpy(features)).cuda()
            _, _, sampled_captions = self.atten_model.build_sample(features_batch)
            decoded = decode_captions(sampled_captions, self.idx_to_word)
            print decoded




