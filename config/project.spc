[vectorizer]
    batch_size = integer(default=1024)
    samples = integer(default=100)
    vector_size = integer(default=128)
    epochs = integer(default=5)

[seq2vec]
    gpu = integer
    load = boolean(default=False)
    lr = float(min=0, default=0.001)
    weight_decay = float(min=0, default=0)
    hidden_size = integer(min=1, default=100)
    batch_size = integer(min=1, default=1)
    max_seqlen = integer(min=1, default=7)
    epochs = integer(min=1)
    n_layers = integer(min=1, default=1)
    bidirectional = boolean(default=False)
    train_samples = integer(default=-1)
    model_file = string(default=model/seq2vec.pth)
    predict_intervals = integer(default=5)
    predict_samples = integer(default=-1)

[visdom]
    server = string(default=0.0.0.0)
    port = integer(min=1025, default=8097)
