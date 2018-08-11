[encoder]
    gpu = integer
    epochs = integer(min=1)
    lr = float(min=0, default=0.001)
    weight_decay = float(min=0, default=0)
