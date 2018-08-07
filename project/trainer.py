#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import torch
from torch.autograd import Variable
import project.decorator as deco
from gensim.models.doc2vec import Doc2Vec
from project.wikitext import WikiTextLoader

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true", default=False,
                        help="if you specified, execute as debug mode. default: 'False'")
    parser.add_argument("--trace", action="store_true", default=False,
                        help="if you specified, execute as trace mode. default: 'False'")
    parser.add_argument("-i", "--indir", type=str, default="data/parsed",
                        help="you can specify the string of the input directory"
                        " must includes subdir 'doc/', and 'title/'. default: 'data/parsed'")
    parser.add_argument("-b", "--batch-size", type=int, default="32",
                        help="you can specify the number of mini batch size."
                        " default: '32'")
    parser.add_argument("-v", "--vector-model", type=str, default="model/doc2vec.model",
                        help="you can specify the string of the vector model file"
                        " default: 'model/doc2vec.model'")
    args = parser.parse_args()
    return args


@deco.trace
@deco.excep
def main():
    args = get_args()

    vm = Doc2Vec.load(args.vector_model)
    wt = WikiTextLoader(args.indir, batch_size=args.batch_size, with_title=True, residual=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    def batch_converter(batch, cuda):
        from nltk.tokenize import word_tokenize
        for title, path in zip(titles, paths):
            words = word_tokenize(title)
            doc_id = re.sub('{args.indir}/doc/', '', str(path))
            word_vectors = []
            for w in words:
                vector = vm.wv[w]
                word_vectors.append(vector)
            title_vectors.append(word_vectors)
            document_vector = vm[doc_id]
            document_vectors.append(document_vector)
        title_vectors = torch.Tensor(title_vectors)
        document_vectors = torch.Tensor(document_vectors)
        assert title_vectors.shape[0] == document_vectors.shape[0]
        return title_vectors, document_vectors

    for batch in wt.iter():
        titles, docs, paths = batch
        title_vectors, document_vectors = batch_converter(batch, cuda)
        tv = Variable(title_vectors)
        dv = Variable(document_vectors)

        optimizer.zero_grad()

        break


if __name__ == '__main__':
    r = main()
    if r != 0:
        logfile = deco.logger.logger.handlers[0].baseFilename
        print(f"Abort with error. see logfile '{logfile}'")
    exit(r)
