# -*- coding: utf-8 -*-


class EncoderWordVectors(nn.Module):
    def __init__(self, input_size, hidden_size=64, n_layers=1):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers

        self.enc_lstm = nn.LSTM(input_size, hidden_size, n_layers)
        self.enc_mn = nn.LSTM(hidden_size, hidden_size, 1)
        self.enc_lv = nn.LSTM(hidden_size, hidden_size, 1)
        self.dec_lstm = nn.LSTM(hidden_size, input_size, n_layers)

    def encode_step(self, x: torch.Tensor, h=None, h_mn=None, h_lv=None):
        o, h = self.enc_lstm(x, h)
        mn = self.enc_mn(o, h_mn)   # mean
        lv = self.enc_lv(o, h_lv)   # log of variance
        return mn, lv, h, h_mn, h_lv

    def forward(self, sentence, hidden=None):
        h_mn = h_lv = None
        for w in sentence:
            mn, lv, h, h_mn, h_lv = self.encode_step(w, hidden, h_mn, h_lv)
        output = mn, lv
        hidden = h, h_mn, h_lv
        return output, hidden

    # def initHidden(self):
    #     return torch.randn(1, 1, self.hidden_size, device=device)
