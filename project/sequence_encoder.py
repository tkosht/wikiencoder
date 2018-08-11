# -*- coding: utf-8 -*-

import pathlib
import numpy
import torch
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as pyplot


class SequenceEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, device=torch.device("cpu"), model_file="model/seqenc.py"):
        super().__init__()
        self.max_seqlen = 7
        self.input_dim = self.output_dim = input_dim
        self.hidden_dim = hidden_dim
        self.device = device
        self.model_file = model_file
        self.loss_records = []

        # for encoder
        self.encoder_lstm = nn.LSTM(input_dim, hidden_dim)
        self.m = nn.Linear(hidden_dim, self.hidden_dim)
        self.v = nn.Linear(hidden_dim, self.hidden_dim)
        
        # for decoder
        self.decoder_lstm = nn.LSTM(hidden_dim, hidden_dim)
        self.decoder_fc = nn.Linear(hidden_dim, self.output_dim)

        self.init_hidden()
        self.to(device)

    def init_hidden(self):
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        hidden_shape = (1, 1, self.hidden_dim)
        self.encoder_hidden = (torch.zeros(*hidden_shape).to(self.device),
                               torch.zeros(*hidden_shape).to(self.device))
        self.decoder_hidden = (torch.zeros(*hidden_shape).to(self.device),
                               torch.zeros(*hidden_shape).to(self.device))
        return self
    
    def encode(self, sequence):
        sequence = sequence.to(self.device)
        encoder_out, self.encoder_hidden = self.encoder_lstm(sequence, self.encoder_hidden)
        mu, logvar = self.m(encoder_out), self.v(encoder_out)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu
    
    def decode(self, sequence):
        decoded, self.encoder_hidden = self.decoder_lstm(sequence, self.decoder_hidden)
        return decoded
    
    def is_similar(self, src, trg):
        _s = numpy.array(src.data)
        _t = numpy.array(trg.data)
        #delta = numpy.linalg.norm(_s) * 0.01
        delta = 1e-6
        return (numpy.linalg.norm(_s - _t) < delta)

    def forward(self, sequence):
        mu, logvar = self.encode(sequence)
        z = self.reparameterize(mu, logvar)
        embeded = z[-1].unsqueeze(0)
        
        y = []
        d = embeded
        for t in range(self.max_seqlen):
            # print(f"t: {t} d.shape: {d.shape}")
            d = self.decode(d)
            _x = self.decoder_fc(d)
            y.append(_x)
            
            eos = sequence[-1]
            is_eos = self.is_similar(_x, eos)
            if is_eos:
                break
        y = torch.cat(y)
        return y, mu, logvar

    def do_train(self, training_data, max_epoch, optimizer):
        assert len(training_data) == 2
        loss_records = []
        for epoch in tqdm(range(max_epoch)):
            loss_mean = 0
            for seq, trg in tqdm(zip(*training_data)):
                self.zero_grad()
                self.init_hidden()

                y, mu, logvar = self(seq)

                loss = get_loss(y, trg, mu, logvar)
                loss_mean += loss
                loss.backward()
                optimizer.step()
            loss_mean /= len(training_data[0])
            # print(f" / epoch: {epoch:05d} loss_mean: {loss_mean}", end="")
            loss_records.append(loss_mean)
            self.loss_records = loss_records
        return self

    def do_predict(self, X):
        with torch.no_grad():
            for seq in X:
                self.zero_grad()
                self.init_hidden()
                y, mu, logvar = self(seq)
        return y

    def load(self):
        weight_params = torch.load(self.model_file)
        self.load_state_dict(weight_params)

    def save(self):
        torch.save(self.state_dict(), self.model_file)

    def save_figure(self, img_file):
        x = self.loss_records
        pyplot.plot(range(len(x)), x)
        pathlib.Path(img_file).parent.mkdir(parents=True, exist_ok=True)
        pyplot.savefig(img_file)


def get_loss(y, t, mu, logvar):
    if len(t.shape) == 2:
        t = t.unsqueeze(1)
    cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    mse = nn.MSELoss()

    n = min(len(y), len(t))
    _y, _t = y[:n], t[:n]

    loss_mse = mse(_y, _t)
    loss_cos = 0.25 * torch.mean(1 - cos(_y, _t))
    loss_cor = 0.25 * torch.mean(1 - cos(_y - torch.mean(_y), _t - torch.mean(_t)))
    loss_similarity = loss_cos + loss_cor
    kl_div = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    loss = kl_div + loss_mse + loss_similarity
    return loss
