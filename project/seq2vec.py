#!/usr/bin/env python
# -*- coding: utf-8 -*-

import re
import argparse
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
from project.sequoder import SequenceEncoder, get_loss


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true", default=False,
                        help="if you specified, execute as debug mode. default: 'False'")
    parser.add_argument("--trace", action="store_true", default=False,
                        help="if you specified, execute as trace mode. default: 'False'")
    # parser.add_argument("--gpu", type=int, default="0",
    #                     help="you can specify the number of gpu processer, "
    #                     "but negative value means cpu. default: '-1'")
    parser.add_argument("--load", action="store_true", default=False,
                        help="if you can specified, load the saved model_file which set on config file. "
                        "default: 'False'")
    args = parser.parse_args()
    return args


def get_wikidata(vector_model, device):
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
            title_words = tokenize_word(title_text)
            seq = [torch.Tensor(vector_model.wv[w]).unsqueeze(0) for w in title_words]
            seq = torch.stack(seq)
            seq = seq.to(device)
            title_data.append(seq)
    return title_data, doc_data

def reverse_tensor(tensor, device=torch.device("cpu")):
    indices = [i for i in range(tensor.size(0)-1, -1, -1)]
    indices = torch.LongTensor(indices).to(device)
    rev_tensor = tensor.index_select(0, indices)
    return rev_tensor


@deco.trace
@deco.excep(return_code=True)
@deco.excep(return_code=True, type_exc=KeyboardInterrupt, warn=True)
def main():
    args = get_args()
    cfg = Config()
    keys = [
            ["encoder", "gpu"],
            ["encoder", "epochs"],
            ["encoder", "lr"],
            ["encoder", "weight_decay"],
            ["encoder", "hidden_size"],
            ["encoder", "train_samples"],
            ["encoder", "model_file"],
            ["encoder", "predict_intervals"],
            ["encoder", "predict_samples"],
            ["visdom", "server"],
            ["visdom", "port"],
            ]
    cfg.setup(keys)

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
    vector_model = Doc2Vec.load("model/doc2vec.model")

    model_params = {
            "input_dim": vector_model.wv.vector_size,
            "hidden_dim": cfg.hidden_size,
            "device": device,
            "model_file": cfg.model_file,
    }
    model = SequenceEncoder(**model_params)
    if args.load:
        @deco.trace
        @deco.excep(warn=True)
        def load_model():
            model.load()
        load_model()

    # make data
    title_data, doc_data = get_wikidata(vector_model, device)
    teacher = [reverse_tensor(seq, device) for seq in title_data]
    training_data = (title_data, teacher)
    trainset = [(x, t) for x, t in zip(*training_data)]
    if cfg.train_samples > 0:
        trainset = random.sample(trainset, cfg.train_samples)

    # setup optimizer
    optim_params = {
        "params": model.parameters(),
        "weight_decay": cfg.weight_decay,
        "lr": cfg.lr,
    }
    optimizer = torch.optim.Adam(**optim_params)

    meter_loss = torchnet.meter.AverageValueMeter()
    train_loss_logger = VisdomPlotLogger('line', port=cfg.port, opts={'title': 'encoder_toy - train loss'})


    def network(sample):
        x = sample[0]   # sequence
        t = sample[1]   # target sequence
        y, mu, logvar = model(x)
        loss = get_loss(y, t, mu, logvar)
        o = y, mu, logvar
        return loss, o

    def reset_meters():
        meter_loss.reset()

    def on_start(state):
        state['dataset'] = state['iterator']

    def on_sample(state):
        state['sample'] = list(state['sample'])
        state['sample'].append(state['train'])
        model.zero_grad()
        model.init_hidden()

    def on_forward(state):
        loss_value = state['loss'].data
        meter_loss.add(state['loss'].data)

    def on_start_epoch(state):
        reset_meters()
        dataset = state['dataset']
        numpy.random.shuffle(dataset)
        state['iterator'] = tqdm(dataset)

    # do predict
    def predict(n_samples=-1):
        _data = title_data
        if n_samples > 0:
            _data = title_data[:n_samples]
        y = model.do_predict(X=_data)
        predicted = [reverse_tensor(seq, device) for seq in y]
        get_word = vector_model.wv.similar_by_vector
        for pseq, tseq in zip(predicted, _data):
            pseq = pseq.squeeze(1)
            tseq = tseq.squeeze(1)
            psim = [get_word(numpy.array(tsr.data, dtype=numpy.float32), topn=1)[0] for tsr in pseq]
            tsim = [get_word(numpy.array(tsr.data, dtype=numpy.float32), topn=1)[0] for tsr in tseq]
            pwords = [elm[0] for elm in psim]
            twords = [elm[0] for elm in tsim]
            print(f'{" ".join(twords)} -> {" ".join(pwords)}')

    def on_end_epoch(state):
        loss_value = meter_loss.value()[0]
        epoch = state['epoch']
        print(f'loss[{epoch}]: {loss_value:.4f}')
        train_loss_logger.log(epoch, loss_value)
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
        model.save()

    save()

    predict(cfg.predict_samples)


if __name__ == '__main__':
    r = main()
    if r != 0:
        logfile = deco.logger.logger.handlers[0].baseFilename
        print(f"Abort with error. see logfile '{logfile}'")
    exit(r)
