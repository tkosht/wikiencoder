#!/usr/bin/env python
# -*- coding: utf-8 -*-

import re
import argparse
import numpy
import torch
import pathlib
import warnings
import matplotlib.pyplot as pyplot
from gensim.models.doc2vec import Doc2Vec

import project.deco as deco
from project.wikitext import tokenize_word
from project.config import Config
from project.sequence_encoder import SequenceEncoder


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
def main():
    args = get_args()
    cfg = Config()
    keys = [
            ["encoder", "gpu"],
            ["encoder", "epochs"],
            ["encoder", "lr"],
            ["encoder", "weight_decay"],
            ["encoder", "hidden_size"],
            ["encoder", "model_file"],
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
    }
    model = SequenceEncoder(**model_params, model_file=cfg.model_file)
    if args.load:
        @deco.trace
        def load_model():
            model.load()
        load_model()

    # make data
    title_data, doc_data = get_wikidata(vector_model, device)
    teacher = [reverse_tensor(seq, device) for seq in title_data]
    training_data = (title_data, teacher)

    # setup optimizer
    optim_params = {
        "params": model.parameters(),
        "weight_decay": cfg.weight_decay,
        "lr": cfg.lr,
    }
    optimizer = torch.optim.Adam(**optim_params)

    # do train
    loss_records = model.do_train(training_data, cfg.epochs, optimizer)

    # save figure for loss_records
    def save_fig(x, img_file):
        pyplot.plot(range(len(x)), x)
        pathlib.Path(img_file).parent.mkdir(parents=True, exist_ok=True)
        pyplot.savefig(img_file)

    save_fig(loss_records, "results/loss_wikidata.png")

    # save the trained model
    @deco.trace
    def save_model():
        model.save(self.model_file)

    save_model()

    # do predict
    y = model.do_predict(X=title_data)
    predicted = [reverse_tensor(seq, device) for seq in y]
    for pseq, tseq in zip(predicted, teacher):
        tseq = tseq.squeeze(1)
        psim = [vector_model.wv.similar_by_vector(numpy.array(tsr.data), topn=1)[0] for tsr in pseq]
        tsim = [vector_model.wv.similar_by_vector(numpy.array(tsr.data), topn=1)[0] for tsr in tseq]
        pwords = [elm[0] for elm in psim]
        twords = [elm[0] for elm in tsim]
        print(f'{" ".join(twords)} -> {" ".join(pwords)}')


if __name__ == '__main__':
    r = main()
    if r != 0:
        logfile = deco.logger.logger.handlers[0].baseFilename
        print(f"Abort with error. see logfile '{logfile}'")
    exit(r)
