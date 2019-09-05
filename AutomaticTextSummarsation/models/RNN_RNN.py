from .BasicModule import BasicModule
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class RNN_RNN(BasicModule):
    #initialization module
    def __init__(self, args, embed=None):
        super(RNN_RNN, self).__init__(args)
        self.model_name = 'RNN_RNN'
        self.args = args
        #vocab size
        V = args.embed_num
        #embedding dimension
        D = args.embed_dim
        #hidden state size
        H = args.hidden_size
        #number of possible segment vectors
        S = args.seg_num
        #number of possible position vectors
        P_V = args.pos_num
        #dimension of positon vector
        P_D = args.pos_dim
        # helps to create a table of embeddings for position and segments in document
        self.abs_pos_embed = nn.Embedding(P_V,P_D)
        self.rel_pos_embed = nn.Embedding(S,P_D)
        # helps to create a table of embeddings for word in vocabulary (V*D)
        self.embed = nn.Embedding(V,D,padding_idx=0) #embedding at zero index is all zero.
        #if embediing table passed as parameter isnot null
        if embed is not None:
            #initialize the embed table with parameter embed
            self.embed.weight.data.copy_(embed)

        #bi-directional GRU-RNN model for word level
        #para: input size: word-embedding size , batch-first=true--->data would be passed in batches
        self.word_RNN = nn.GRU(
                        input_size = D, 
                        hidden_size = H,
                        batch_first = True,
                        bidirectional = True
                        )
        #bi-directional GRU-RNN model for sentence level
        #para: input size: avg-pooled concatenated hidden states from bi-direc wordlevel RNN  
        #batch-first=true--->data would be passed in batches
        self.sent_RNN = nn.GRU(
                        input_size = 2*H,
                        hidden_size = H,
                        batch_first = True,
                        bidirectional = True
                        )
        #create a model for linear transformation with --->input size: 2*h ,output_size: 2*h
        #linear func : output= w* input + b where w,b are learnable 
        self.fc = nn.Linear(2*H,2*H)

        # Parameters of Classification Layer needed while training
        #each output is of size 1 so that we can add for prob
        #linear(input_size,output_size)
        #bilinear(input1_size, input2_size, output_size)
        self.content = nn.Linear(2*H,1,bias=False)

        #create a model for Bilinear transformation with --->1st input size: 2*h , 2nd input size: 2*h, output_size: 1
        #Bilinear func : output= input1*w* input2 + b where w,b are learnable 
        self.salience = nn.Bilinear(2*H,2*H,1,bias=False)
        self.novelty = nn.Bilinear(2*H,2*H,1,bias=False)


        self.abs_pos = nn.Linear(P_D,1,bias=False)
        self.rel_pos = nn.Linear(P_D,1,bias=False)

        self.bias = nn.Parameter(torch.FloatTensor(1).uniform_(-0.1,0.1))


    def max_pool1d(self,x,seq_lens):
        # x:[N,L,O_in]
        out = []
        for index,t in enumerate(x):
            t = t[:seq_lens[index],:]
            t = torch.t(t).unsqueeze(0)
            out.append(F.max_pool1d(t,t.size(2)))
        
        out = torch.cat(out).squeeze(2)
        return out


    def avg_pool1d(self,x,seq_lens):
        # x:[N,L,O_in]
        out = []
        for index,t in enumerate(x):
            t = t[:seq_lens[index],:]
            t = torch.t(t).unsqueeze(0)
            out.append(F.avg_pool1d(t,t.size(2)))
        
        out = torch.cat(out).squeeze(2)
        return out

    def forward(self,x,doc_lens):
        #sum across column[reduce colmns---> 1]
        sent_lens = torch.sum(torch.sign(x),dim=1).data 
        #store embedding for indx x into x
        x = self.embed(x)                                                      # (N,L,D)
        # word level GRU
        H = self.args.hidden_size
       
        #tensor containing the output features (h_k) from the last layer of the word level RNN stored into x
        x = self.word_RNN(x)[0]      
        #print("x is", x)                                          # (N,2*H,L)
        #word_out = self.avg_pool1d(x,sent_lens)
        word_out = self.max_pool1d(x,sent_lens)
        # make sent features(pad with zeros)
        x = self.pad_doc(word_out,doc_lens)

        # sent level GRU
        #tensor containing the output features (h_k) from the last layer of the sent level RNN stored into x
        sent_out = self.sent_RNN(x)[0]                                           # (B,max_doc_len,2*H)
        #docs = self.avg_pool1d(sent_out,doc_lens)                               # (B,2*H)
        docs = self.max_pool1d(sent_out,doc_lens)                                # (B,2*H)
        probs = []
        for index,doc_len in enumerate(doc_lens):
            valid_hidden = sent_out[index,:doc_len,:]                            # (doc_len,2*H)
            doc = F.tanh(self.fc(docs[index])).unsqueeze(0)
            s = Variable(torch.zeros(1,2*H))
 
            for position, h in enumerate(valid_hidden):
                h = h.view(1, -1)                                                # (1,2*H)
                # get position embeddings
                abs_index = Variable(torch.LongTensor([[position]]))
                
                abs_features = self.abs_pos_embed(abs_index).squeeze(0)
                
                rel_index = int(round((position + 1) * 9.0 / doc_len))
                rel_index = Variable(torch.LongTensor([[rel_index]]))
                
                rel_features = self.rel_pos_embed(rel_index).squeeze(0)
                
                # classification layer
                content = self.content(h) 
                salience = self.salience(h,doc)
                novelty = -1 * self.novelty(h,F.tanh(s))
                abs_p = self.abs_pos(abs_features)
                rel_p = self.rel_pos(rel_features)
                #prob(i)---->p(sentence being part of summary , yi=1/ d,s,hi) : dim 1
                prob = F.sigmoid(content + salience + novelty + abs_p + rel_p + self.bias)
                #matrix multiply
                s = s + torch.mm(prob,h)
                probs.append(prob)
                #torch.squeeze(input, dim=None, out=None) → Returns a tensor with all the dimensions of input of size 1 removed.
        #Concatenates the given sequence of seq tensors in the given dimension. All tensors must either have the same shape 
        return torch.cat(probs).squeeze()
