#!/usr/bin/env python
# -*- coding: utf-8 -*-

import re
import argparse
import pathlib
import numpy
import random
import torch
import torchnet
import pathlib
import warnings
from tqdm import tqdm
from gensim.models.doc2vec import Doc2Vec
from torchnet.engine import Engine
from torchnet.logger import VisdomPlotLogger

import project.deco as deco
from project.wikitext import tokenize_word
from project.config import Config
from project.sequoder import SequenceEncoder


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true", default=False,
                        help="if you specified, execute as debug mode. default: 'False'")
    parser.add_argument("--trace", action="store_true", default=False,
                        help="if you specified, execute as trace mode. default: 'False'")
    # parser.add_argument("--gpu", type=int, default="0",
    #                     help="you can specify the number of gpu processer, "
    #                     "but negative value means cpu. default: '-1'")
    # parser.add_argument("--load", action="store_true", default=False,
    #                     help="if you can specified, load the saved model_file which set on config file. "
    #                     "default: 'False'")
    args = parser.parse_args()
    return args


def get_config():
    cfg = Config()
    keys = [
        ["seq2vec", "gpu"],
        ["seq2vec", "load"],
        ["seq2vec", "epochs"],
        ["seq2vec", "lr"],
        ["seq2vec", "weight_decay"],
        ["seq2vec", "hidden_size"],
        ["seq2vec", "n_layers"],
        ["seq2vec", "bidirectional"],
        ["seq2vec", "batch_size"],
        ["seq2vec", "max_seqlen"],
        ["seq2vec", "train_samples"],
        ["seq2vec", "model_dir"],
        ["seq2vec", "predict_intervals"],
        ["seq2vec", "predict_samples"],
        ["visdom", "server"],
        ["visdom", "port"],
    ]
    cfg.setup(keys)
    return cfg


def create_wikidataset(vector_model, max_seqlen, device):
    indir="data/parsed"
    title_data = []
    doc_data = []
    p = pathlib.Path(indir).joinpath("samples.list")
    with p.open("r") as f:
        for doc_file in f:
            title_file = re.sub("/doc/", "/title/", doc_file.strip())
            tp = pathlib.Path(title_file)
            with tp.open("r") as tfp:
                title_text = tfp.read()
            title_words = tokenize_word(title_text)[1:]     # except bos
            title_words = title_words[:max_seqlen]
            seq = [torch.Tensor(vector_model.wv[w]).unsqueeze(0) for w in title_words]
            k = max_seqlen - len(title_words)
            for idx in range(k):
                seq.append(torch.zeros_like(seq[0]))
            assert len(seq) == max_seqlen
            seq = torch.stack(seq)
            seq = seq.to(device)
            title_data.append(seq)
    return title_data, doc_data


@deco.trace
@deco.excep(return_code=True)
@deco.excep(return_code=True, type_exc=KeyboardInterrupt, warn=True)
def main():
    # args = get_args()
    cfg = get_config()

    # setup device
    if cfg.gpu < 0:
        device = torch.device("cpu")
    else:
        device = torch.device(f"cuda:{cfg.gpu}")
        try:
            torch.Tensor([]).to(device)
        except Exception as e:
            warnings.warn(f"your gpu number:[{cfg.gpu}] may be invalid. will use default gpu. [{str(e)}]")
            device = torch.device(f"cuda")  # set as default

    # load trained doc2vec model
    @deco.trace
    def load_vector_model():
        return Doc2Vec.load("model/doc2vec.model")

    vector_model = load_vector_model()

    model_params = {
            "input_dim": vector_model.wv.vector_size,
            "hidden_dim": cfg.hidden_size,
            "n_layers": cfg.n_layers,
            "bidirectional": cfg.bidirectional,
            "batch_size": cfg.batch_size,
            "device": device,
            "model_file": pathlib.Path(model_dir).joinpath("title.pth"),
    }
    words_model = SequenceEncoder(**model_params)

    model_params = {
        "input_dim": vector_model.vector_size,
        "hidden_dim": cfg.hidden_size,
        "n_layers": cfg.n_layers,
        "bidirectional": cfg.bidirectional,
        "batch_size": cfg.batch_size,
        "device": device,
        "model_file": pathlib.Path(model_dir).joinpath("doc.pth"),
    }
    doc_model = SequenceEncoder(**model_params)

    if cfg.load:

        @deco.trace
        @deco.excep(warn=True)
        def load_model():
            words_model.load()

        load_model()

    # make data
    title_data, doc_data = create_wikidataset(vector_model, cfg.max_seqlen, device)

    def create_trainset(n_samples = -1):
        trainset = []
        for x in title_data:
            trainset.append((x, x))
        numpy.random.shuffle(trainset)
        if n_samples > 0:
            trainset = random.sample(trainset, n_samples)
        return trainset

    trainset = create_trainset(cfg.train_samples)

    # setup optimizer
    optim_params = {
        "params": words_model.parameters(),
        "weight_decay": cfg.weight_decay,
        "lr": cfg.lr,
    }
    optimizer = torch.optim.Adam(**optim_params)

    meter_loss = torchnet.meter.AverageValueMeter()
    train_loss_logger = VisdomPlotLogger('line', port=cfg.port, opts={'title': 'seq2vec - train loss'})


    def network(sample):
        x = sample[0]   # sequence
        t = sample[1]   # target sequence
        o = y, mu, logvar = words_model(x)
        loss = words_model.get_loss(y, t, mu, logvar)
        o2 = y2, mu2, logvar2 = doc_model(x)
        loss += doc_model.get_loss(y, t, mu, logvar)
        return loss, o

    def reset_meters():
        meter_loss.reset()

    def on_start(state):
        words_model.train()
        # state['dataset'] = state['iterator']
        state['trainset'] = trainset

    def on_sample(state):
        state['sample'] = list(state['sample'])
        state['sample'].append(state['train'])
        words_model.zero_grad()
        words_model.init_hidden()

    def on_forward(state):
        loss_value = state['loss'].data
        meter_loss.add(loss_value)

    def on_start_epoch(state):
        words_model.train()
        reset_meters()
        trainset = state['trainset']
        numpy.random.shuffle(trainset)
        # trainset = create_trainset(cfg.train_samples)
        state['iterator'] = tqdm(trainset)

    # do predict
    def predict(n_samples=-1):
        _data = title_data
        if n_samples > 0:
            _data = title_data[:n_samples]
        y = words_model.do_predict(X=_data)
        predicted = y
        get_word = vector_model.wv.similar_by_vector
        for pseq, tseq in zip(predicted, _data):
            pseq = pseq.squeeze(1)
            tseq = tseq.squeeze(1)
            psim = [get_word(numpy.array(tsr.data, dtype=numpy.float32), topn=1)[0] for tsr in pseq]
            tsim = [get_word(numpy.array(tsr.data, dtype=numpy.float32), topn=1)[0] for tsr in tseq]
            # pwords = [f"{elm[0]}({elm[1]:.2f})" for elm in psim]
            pwords = []
            for word, similarity in psim:
                pwords.append(f"{word}({similarity:.2f})")
                if word == "__eos__":
                    break
            twords = [elm[0] for elm in tsim]
            print(f'{" ".join(twords)}\n\t-> {" ".join(pwords)}\n')

    def on_end_epoch(state):
        loss_value = meter_loss.value()[0]
        epoch = state['epoch']
        print(f'loss[{epoch}]: {loss_value:.6f}')
        train_loss_logger.log(epoch, loss_value)
        if cfg.predict_intervals <= 0:
            return
        if epoch % cfg.predict_intervals == 0:
            predict(cfg.predict_samples)

    engine = Engine()
    engine.hooks['on_start'] = on_start
    engine.hooks['on_sample'] = on_sample
    engine.hooks['on_forward'] = on_forward
    engine.hooks['on_start_epoch'] = on_start_epoch
    engine.hooks['on_end_epoch'] = on_end_epoch

    engine.train(network, trainset, maxepoch=cfg.epochs, optimizer=optimizer)

    @deco.trace
    def save():
        # save the trained model
        words_model.save()

    save()

    predict(cfg.predict_samples)


if __name__ == '__main__':
    r = main()
    if r != 0:
        logfile = deco.logger.logger.handlers[0].baseFilename
        print(f"Abort with error. see logfile '{logfile}'")
    exit(r)
