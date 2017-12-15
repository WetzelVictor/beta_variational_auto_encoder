""" modAttentionV2loop.py
Implementation of a Attention RNN model derived from https://medium.com/datalogue/attention-in-keras-1892773a4f22 """

#Librairies
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class Attention(nn.Module):

    def __init__(self, sample_size, h_dim, z_dim, batch_size):
        """
        Implements an Attention model that takes in a sequence encoded by an
        encoder and outputs the decoded states.
        
        Arguments :
            sample_size: size of the samples
            h_dim: dimension of the hidden state and the attention matrices
            z_dim: the number of labels in the encoded space
            batch_size: size of the minibatch
        
        Returns : 
            Attention object
            
        References:
            Bahdanau, Dzmitry, Kyunghyun Cho, and Yoshua Bengio. 
            "Neural machine translation by jointly learning to align and translate." 
            arXiv preprint arXiv:1409.0473 (2014).
        """
        super(Attention, self).__init__()
        self.h_dim = h_dim
        self.z_dim = z_dim
        self.sample_size = sample_size
        self.batch_size = batch_size
        self.hidden = self.init_hidden()
        self.lstm = nn.LSTM(sample_size, h_dim)
        self.hiddentoz = nn.Linear(h_dim, z_dim*2)


    def init_hidden(self):
        # Before we've done anything, we don't have any hidden state

        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (Variable(torch.zeros(1, self.batch_size, self.h_dim)),
                Variable(torch.zeros(1, self.batch_size, self.h_dim)))
        
    def get_initial_state(self, encoded):
        # apply the matrix on the first time step to get the initial s0
        fs = nn.Linear(self.z_dim, self.h_dim)
        s0 = F.tanh(fs(encoded[0,:]))

        # initialize a vector of (batchsize,output_dim)
        # apply the matrix on the first time step to get the initial s0.
        fs = nn.Linear(self.z_dim, self.h_dim)
        s0 = F.tanh(fs(encoded[0,:]))
        s0 = s0.view(self.batch_size,-1)

        # initialize a vector of (batchsize,
        # output_dim)
        y0 = Variable(torch.zeros(self.batch_size, self.sample_size))

        return [y0, s0]
        
    def reparametrize(self, mu, log_var):
        """"z = mean + eps * sigma where eps is sampled from N(0, 1)."""
        eps = Variable(torch.randn(mu.size(0), mu.size(1), mu.size(2)))
        output = mu + eps * torch.exp(log_var/2)    # 2 for convert var to std
        return output

    def forward(self, data):
        """ Do the forward pass in the Attention model: encodes, reparameterize and
        decodes the input data.
        
        Arguments:
            - data (array[TIME_INDEX][BATCH_INDEX][DATA_VECTOR]):
        Returns:
            - output (array like data): data encoded, reparametrized then decoded
            -[yt, st]: llast hidden state
            - mu: mean 
            - log_var: log value of the variance """
        
        timesteps, batch_size, sample_size = data.size()
        
        #ENCODING
        xtemp, self.hidden = self.lstm(data, self.hidden)
        x = self.hiddentoz(xtemp)
        mu, log_var = torch.chunk(x, 2, dim=2)
        x_encoded = self.reparametrize(mu, log_var)
        
        #DECODING
        ytm, stm = self.get_initial_state(x_encoded)
        outputs = ytm.view(1,self.batch_size,-1)
        
        for t in range(timesteps):
            # repeat the hidden state to the length of the sequence
            _stm = stm.repeat(timesteps,1)
            _stm = _stm.view(timesteps, batch_size, -1)
            
            # now multiply the weight matrix with the repeated hidden state
            fa1 = nn.Linear(self.h_dim, self.h_dim)
            fa2 = nn.Linear(self.z_dim, self.h_dim)
            fa3 = nn.Linear(self.h_dim, 1)
            attention_input = fa1(_stm) + fa2(x_encoded)
            # calculate the attention probabilities

            # this relates how much other timesteps contributed to this one
            et = F.tanh(attention_input)
            et = fa3(et)
            at = torch.exp(et)
            at_sum = torch.sum(at, 0)
            at = torch.div(at,at_sum)  # vector of size (timesteps, batchsize, 1)
            
            # calculate the context vector
            context = torch.sum(torch.mul(at, x_encoded),0)

            # ~~~> calculate new hidden state
            # first calculate the "r" gate:
            fr1 = nn.Linear(sample_size, self.h_dim)
            fr2 = nn.Linear(self.h_dim, self.h_dim)
            fr3 = nn.Linear(self.z_dim, self.h_dim)
            rgate_input = fr1(ytm) + fr2(stm) + fr3(context)
            rt = F.sigmoid(rgate_input)
            
            # now calculate the "z" gate
            fz1 = nn.Linear(sample_size, self.h_dim)
            fz2 = nn.Linear(self.h_dim, self.h_dim)
            fz3 = nn.Linear(self.z_dim, self.h_dim)
            zgate_input = fz1(ytm) + fz2(stm) + fz3(context)
            zt = F.sigmoid(zgate_input)
            
            # calculate the proposal hidden state:
            fp1 = nn.Linear(sample_size, self.h_dim)
            fp2 = nn.Linear(self.h_dim, self.h_dim)
            fp3 = nn.Linear(self.z_dim, self.h_dim)
            proposal_input = fp1(ytm) + fp2(rt*stm) + fp3(context)
            s_tp = F.tanh(proposal_input)
            
            # new hidden state:
            st = (1-zt)*stm + zt * s_tp
            

            #calculate output:
            fo1 = nn.Linear(sample_size, sample_size)
            fo2 = nn.Linear(self.h_dim, sample_size)
            fo3 = nn.Linear(self.z_dim, sample_size)
            output_input = fo1(ytm) + fo2(stm) + fo3(context)
            yt = F.softmax(output_input)
            outputs = torch.cat((outputs, yt.view(1,batch_size,-1)),0)
            ytm, stm = yt, st
        outputs = outputs[1:,:]

        return outputs, [yt, st], mu, log_var
