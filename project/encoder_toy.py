#!/usr/bin/env python
# -*- coding: utf-8 -*-

import re
import argparse
import numpy
import torch
import torchnet
import pathlib
from tqdm import tqdm
import matplotlib.pyplot as pyplot
from gensim.models.doc2vec import Doc2Vec
from torchnet.engine import Engine
from torchnet.logger import VisdomPlotLogger, VisdomLogger

import project.deco as deco
from project.sequence_encoder import SequenceEncoder, get_loss


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true", default=False,
                        help="if you specified, execute as debug mode. default: 'False'")
    parser.add_argument("--trace", action="store_true", default=False,
                        help="if you specified, execute as trace mode. default: 'False'")
    # parser.add_argument("-i", "--indir", type=str, default="data/parsed",
    #                     help="you can specify the string of the input directory"
    #                     " must includes subdir 'doc/', and 'title/'. default: 'data/parsed'")
    parser.add_argument("--epochs", type=int, default="500")
    parser.add_argument("--lr", type=float, default="0.001")
    parser.add_argument("--weight-decay", type=float, default="0")
    args = parser.parse_args()
    return args

def get_toydata(n_data, device):
    toydata = []
    for _n in range(n_data):
        t = numpy.random.randint(5) + 2
        seq = [torch.randn(1, 3) for _t in range(t)]  # make a sequence of length 5
        seq = torch.stack(seq)
        seq = seq.to(device)
        toydata.append(seq)
    return toydata

def reverse_tensor(tensor, device=torch.device("cpu")):
    indices = [i for i in range(tensor.size(0)-1, -1, -1)]
    indices = torch.LongTensor(indices).to(device)
    rev_tensor = tensor.index_select(0, indices)
    return rev_tensor


@deco.trace
@deco.excep(return_code=True)
def main():
    args = get_args()

    device = torch.device("cuda:1")
    # device = torch.device("cpu")
    model = SequenceEncoder(3, 2, device)

    n_data = 10
    data = get_toydata(n_data, device)
    teacher = [reverse_tensor(seq, device) for seq in data]
    training_data = (data, teacher)

    optim_params = {
        "params": model.parameters(),
        "weight_decay": args.weight_decay,
        "lr": args.lr,
    }
    optimizer = torch.optim.Adam(**optim_params)

    meter_loss = torchnet.meter.AverageValueMeter()
    port = 8097
    train_loss_logger = VisdomPlotLogger(
        'line', port=port, opts={'title': 'encoder_toy - train loss'})

    def network(sample):
        x = sample[0]   # sequence
        t = sample[1]   # target sequence
        y, mu, logvar = model(x)
        loss = get_loss(y, t, mu, logvar)
        o = y, mu, logvar
        return loss, o

    def reset_meters():
        meter_loss.reset()

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
        if 'dataset' not in state:
            dataset = state['iterator']
            state['dataset'] = dataset
        dataset = state['dataset']
        state['iterator'] = tqdm(zip(*dataset))

    def on_end_epoch(state):
        loss_value = meter_loss.value()[0]
        epoch = state['epoch']
        print(f'loss[{epoch}]: {loss_value:.4f}')
        train_loss_logger.log(epoch, loss_value)
        dataset = state['dataset']
        state['iterator'] = tqdm(zip(*dataset))

    engine = Engine()
    engine.hooks['on_sample'] = on_sample
    engine.hooks['on_forward'] = on_forward
    engine.hooks['on_start_epoch'] = on_start_epoch
    engine.hooks['on_end_epoch'] = on_end_epoch

    engine.train(network, training_data, maxepoch=args.epochs, optimizer=optimizer)


    # loss_records = model.do_train(training_data, args.epochs, optimizer)

    # def save_fig(x, img_file):
    #     pyplot.plot(range(len(x)), x)
    #     pathlib.Path(img_file).parent.mkdir(parents=True, exist_ok=True)
    #     pyplot.savefig(img_file)

    # save_fig(loss_records, "results/loss_toydata.png")


if __name__ == '__main__':
    r = main()
    if r != 0:
        logfile = deco.logger.logger.handlers[0].baseFilename
        print(f"Abort with error. see logfile '{logfile}'")
    exit(r)
