""" modAttentiondef.py
Implementation of an Attention RNN model derived from
https://medium.com/datalogue/attention-in-keras-1892773a4f22 """

# Librairies
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class AttentionRnn(nn.Module):

    def __init__(self, sample_size, h_dim, z_dim):
        """
        Implements an Attention model that takes in a sequence encoded by an
        encoder and outputs the decoded states.

        Arguments:
            sample_size: size of the samples
            h_dim: dimension of the hidden state and the attention matrices
            z_dim: dimension of the encoded space
        Returns:
            Attention object

        References:
            Bahdanau, Dzmitry, Kyunghyun Cho, and Yoshua Bengio.
            "Neural machine translation
            by jointly learning to align and translate."
            arXiv preprint arXiv:1409.0473 (2014).
        """
        super(AttentionRnn, self).__init__()
        self.h_dim = h_dim
        self.z_dim = z_dim
        self.sample_size = sample_size
        self.lstm = nn.LSTM(sample_size, h_dim)
        self.hiddentoz = nn.Linear(h_dim, z_dim*2)
        self.fa1 = nn.Linear(h_dim, h_dim)
        self.fa2 = nn.Linear(z_dim, h_dim)
        self.fa3 = nn.Linear(h_dim, 1)
        self.fr1 = nn.Linear(sample_size, h_dim)
        self.fr2 = nn.Linear(h_dim, h_dim)
        self.fr3 = nn.Linear(z_dim, h_dim)
        self.fz1 = nn.Linear(sample_size, h_dim)
        self.fz2 = nn.Linear(h_dim, h_dim)
        self.fz3 = nn.Linear(z_dim, h_dim)
        self.fp1 = nn.Linear(sample_size, h_dim)
        self.fp2 = nn.Linear(h_dim, h_dim)
        self.fp3 = nn.Linear(z_dim, h_dim)
        self.fo1 = nn.Linear(sample_size, sample_size)
        self.fo2 = nn.Linear(h_dim, sample_size)
        self.fo3 = nn.Linear(z_dim, sample_size)
        self.fs = nn.Linear(z_dim, h_dim)

    def init_hidden(self, batch_size):
        # initiates hidden state for encoder lstm
        # the axes semantics are (num_layers, batch_size, hidden_dim)
        return (Variable(torch.zeros(1, batch_size, self.h_dim)),
                Variable(torch.zeros(1, batch_size, self.h_dim)))

    def reparametrize(self, mu, log_var):
        """"z = mean + eps * sigma where eps is sampled from N(0, 1)."""
        eps = Variable(torch.randn(mu.size(0), mu.size(1), mu.size(2)))
        output = mu + eps * torch.exp(log_var/2)    # 2 for convert var to std
        return output

    def encoder(self, data):
        batch_size = data.size(1)
        hidden = self.init_hidden(batch_size)
        xtemp, hidden = self.lstm(data, hidden)
        x = self.hiddentoz(xtemp)
        mu, log_var = torch.chunk(x, 2, dim=2)
        x_encoded = self.reparametrize(mu, log_var)
        return x_encoded, mu, log_var

    def get_initial_state(self, encoded):
        # apply the matrix on the first time step of encoded data
        # to get the initial s0
        batch_size = encoded.size(1)
        s0 = F.tanh(self.fs(encoded[0, :]))
        # initialize a vector of (batchsize, output dim)
        y0 = Variable(torch.zeros(batch_size, self.sample_size))
        return [y0, s0]

    def decoder(self, encoded):
        timesteps = encoded.size(0)
        batch_size = encoded.size(1)
        ytm, stm = self.get_initial_state(encoded)
        outputs = ytm.view(1, batch_size, -1)

        for t in range(timesteps):
            # repeat the hidden state to the length of the sequence
            _stm = stm.repeat(timesteps, 1)
            _stm = _stm.view(timesteps, batch_size, -1)

            # calculate the attention probabilities
            # this relates to how much other timesteps contributed to this one
            et = F.tanh(self.fa1(_stm) + self.fa2(encoded))
            et = self.fa3(et)
            at = torch.exp(et)
            at_sum = torch.sum(at, 0)
            at = torch.div(at, at_sum)  # vector of size
            # (timesteps, batchsize, 1)

            # calculate the context vector
            context = torch.sum(torch.mul(at, encoded), 0)

            # ~~~> calculate new hidden state
            # first calculate the "r" gate:
            rt = F.sigmoid(self.fr1(ytm) + self.fr2(stm) + self.fr3(context))

            # now calculate the "z" gate:
            zt = F.sigmoid(self.fz1(ytm) + self.fz2(stm) + self.fz3(context))

            # calculate the proposal hidden state:
            s_tp = F.tanh(self.fp1(ytm) + self.fp2(rt*stm) + self.fp3(context))

            # new hidden state:
            st = (1-zt)*stm + zt * s_tp

            # calculate output:
            yt = F.softmax(self.fo1(ytm) + self.fo2(stm) + self.fo3(context))
            outputs = torch.cat((outputs, yt.view(1, batch_size, -1)), 0)
            ytm, stm = yt, st
        outputs = outputs[1:, :]  # deleting yO
        return outputs, [yt, st]

    def forward(self, data):
        """ Do the forward pass in the Attention model: encodes, reparameterizes and
        decodes the input data.

        Arguments:
            - data (array[TIME_INDEX][BATCH_INDEX][DATA_VECTOR]):
        Returns:
            - output (array like data): data encoded, reparametrized then
            decoded
            - [yt, st]: last hidden state
            - mu: mean
            - log_var: log value of the variance """

        x_encoded, mu, log_var = self.encoder(data)
        outputs, [yt, st] = self.decoder(x_encoded)

        return outputs, [yt, st], mu, log_var

    def sample(self, latent_vec):
        """ Decodes the given latent vector.

        Arguments:
            - latent_vec: matrix containing a number of encoded vectors
        Returns:
            - output: decoded vector
        """
        return self.decoder(latent_vec)
