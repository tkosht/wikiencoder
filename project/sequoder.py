# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F

def reverse_tensor(tensor, device=torch.device("cpu")):
    indices = [i for i in range(tensor.size(0)-1, -1, -1)]
    indices = torch.LongTensor(indices).to(device)
    rev_tensor = tensor.index_select(0, indices)
    return rev_tensor

def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_normal_(m.weight)


class SequenceEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_layers=1, bidirectional=False, batch_size=1,
                 device=torch.device("cpu"), model_file="model/seqenc.pth"):
        super().__init__()
        if bidirectional:
            raise NotImplementedError()

        self.max_seqlen = 7
        self.input_dim = self.output_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.bidirectional = bidirectional
        self.batch_size = batch_size
        self.device = device
        self.model_file = model_file
        self.vae_mode = False

        # for encoder
        self.encoder_lstm = nn.LSTM(input_dim, hidden_dim, num_layers=n_layers, bidirectional=bidirectional)
        self.encoder_embed = nn.Linear(hidden_dim * self.max_seqlen, hidden_dim)
        self.encoder_embed_m = nn.Linear(hidden_dim * self.max_seqlen, hidden_dim)
        self.encoder_embed_v = nn.Linear(hidden_dim * self.max_seqlen, hidden_dim)

        # for decoder
        self.decoder_lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers=n_layers, bidirectional=bidirectional)
        self.decoder_expand = nn.Linear(hidden_dim * self.max_seqlen, self.output_dim * self.max_seqlen)

        self.encoder_hidden = None
        self.decoder_hidden = None

        self.apply(init_weights)
        self.init_hidden()
        self.to(device)

    def init_hidden(self):
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        depth = self.n_layers * (1 + int(self.bidirectional))
        hidden_shape = (depth, 1, self.hidden_dim)
        self.encoder_hidden = (torch.zeros(*hidden_shape).to(self.device),
                               torch.zeros(*hidden_shape).to(self.device))
        self.decoder_hidden = (torch.zeros(*hidden_shape).to(self.device),
                               torch.zeros(*hidden_shape).to(self.device))
        return self

    def encode(self, sequence):
        sequence = sequence.to(self.device)
        # if self.training:
        #     sgm = 1e-6
        #     noise = 1 + sgm * torch.randn(sequence.shape)
        #     noise = noise.to(self.device)
        #     sequence *= noise
        encseq, self.encoder_hidden = self.encoder_lstm(sequence, self.encoder_hidden)
        mu = logvar = None
        if self.vae_mode:
            mu = self.encoder_embed_m(F.relu(encseq.view(-1)))
            logvar = self.encoder_embed_v(F.relu(encseq.view(-1)))
            embeded = self.reparameterize(mu, logvar)
        else:
            embeded = self.encoder_embed(F.relu(encseq.view(-1)))
        return embeded, mu, logvar

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, sequence):
        shp = (self.max_seqlen, 1, -1)
        seq = sequence.expand(*shp)
        decseq, self.decoder_hidden = self.decoder_lstm(seq, self.decoder_hidden)
        decoded = self.decoder_expand(F.relu(decseq.view(-1)))
        return decoded.view(*shp)
    
    def forward(self, sequence):
        embeded, mu, logvar = self.encode(sequence)
        decoded = self.decode(embeded)
        return decoded, mu, logvar

    def do_predict(self, X):
        self.eval()
        with torch.no_grad():
            predicted = []
            for seq in X:
                self.zero_grad()
                self.init_hidden()
                y, mu, logvar = self(seq)
                predicted.append(y)
        return predicted

    def load(self):
        weight_params = torch.load(self.model_file)
        self.load_state_dict(weight_params)

    def save(self):
        torch.save(self.state_dict(), self.model_file)

    def get_loss(self, y, t, mu, logvar):
        assert len(y) == len(t)
        for idx, ttsr in enumerate(t):
            if (ttsr == 0).all():
                break

        mask = torch.zeros(self.max_seqlen).to(self.device)
        mask[:idx] = 1
        y *= mask.view(self.max_seqlen, 1, 1).expand(*y.shape)

        # diff penalty
        loss = self.get_penalty(y, t)

        if self.vae_mode:
            kl_div = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            loss += kl_div

        return loss

    def get_penalty(self, y, t):
        err = nn.SmoothL1Loss()
        cos = nn.CosineSimilarity(dim=2, eps=1e-6)
        loss_err = err(y, t)
        loss_cos = torch.mean(torch.sum(1 - cos(y, t), dim=1))
        penalty = loss_err + loss_cos
        return penalty

    def cos_embedding_loss(self, y, t, is_positive, margin=0.5):
        cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        cos_tensor = cos(y.squeeze(1), t.squeeze(1))
        if is_positive:
            return torch.mean(1 - cos_tensor)
        else:
            zr = torch.zeros(len(y)).to(self.device)
            cs = cos_tensor - margin
            return torch.max(zr, cs).mean()


class SimilarityEncoder(nn.Module):
    def __init__(self, title_model, doc_model, device=torch.device("cpu"), model_file="model/simenc.pth"):
        super().__init__()
