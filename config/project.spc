[vectorizer]
    batch_size = integer(default=1024)
    samples = integer(default=100)
    vector_size = integer(default=128)
    epochs = integer(default=5)

[encoder]
    gpu = integer
    epochs = integer(min=1)
    lr = float(min=0, default=0.001)
    weight_decay = float(min=0, default=0)
    hidden_size = integer(min=1, default=100)
    model_file = string(default=model/seq2vec.pth)

[visdom]
    server = string(default=0.0.0.0)
    port = integer(min=1025, default=8097)
