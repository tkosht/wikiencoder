#!/usr/bin/env python
# -*- coding: utf-8 -*-

import re
import argparse
import numpy
import torch
import pathlib
import matplotlib.pyplot as pyplot
from gensim.models.doc2vec import Doc2Vec

import project.deco as deco
from project.sequence_encoder import SequenceEncoder


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
@deco.excep
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

    loss_records = model.do_train(training_data, args.epochs, optimizer)

    def save_fig(x, img_file):
        pyplot.plot(range(len(x)), x)
        pathlib.Path(img_file).parent.mkdir(parents=True, exist_ok=True)
        pyplot.savefig(img_file)

    save_fig(loss_records, "results/loss_toydata.png")


if __name__ == '__main__':
    r = main()
    if r != 0:
        logfile = deco.logger.logger.handlers[0].baseFilename
        print(f"Abort with error. see logfile '{logfile}'")
    exit(r)
