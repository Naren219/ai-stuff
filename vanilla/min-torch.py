# minimum code for neural network in pytorch (no torch.nn)

import torch
from pathlib import Path
import requests
import pickle
import gzip
import math

if torch.backends.mps.is_available():
    torch.set_default_device("mps")
else:
    torch.set_default_device("cpu")

# -- getting da data --
DATA_PATH = Path("data")
PATH = DATA_PATH / "mnist"

PATH.mkdir(parents=True, exist_ok=True)

URL = "https://github.com/pytorch/tutorials/raw/main/_static/"
FILENAME = "mnist.pkl.gz"

if not (PATH / FILENAME).exists():
        content = requests.get(URL + FILENAME).content
        (PATH / FILENAME).open("wb").write(content)

with gzip.open((PATH / FILENAME).as_posix(), "rb") as f:
        ((x_train, y_train), (x_valid, y_valid), _) = pickle.load(f, encoding="latin-1")

# convert to torch tensor
x_train, y_train, x_valid, y_valid = map(
    torch.tensor, (x_train, y_train, x_valid, y_valid)
)
n, c = x_train.shape

# -- init model --
weights = torch.randn(c, 10) / math.sqrt(c)
weights.requires_grad_()
bias = torch.zeros(10, requires_grad=True)

def log_softmax(x):
    return x - torch.logsumexp(x, dim=-1, keepdim=True)

def model(xb):
    return log_softmax(xb @ weights + bias)

def nll(inputs, target):
    # pick the log probs of correct class for each example, then average the negative
    idx = torch.arange(target.shape[0])
    return -inputs[idx, target].mean()

def accuracy(out, yb):
    # find label with highest probability
    preds = torch.argmax(out, dim=1)
    return (preds == yb).float().mean()

# -- before --
bs = 64
print("-- before --")
with torch.no_grad():
    out = model(x_valid[:bs])
    print("Valid loss:", nll(out, y_valid[:bs]).item())
    print("Valid accuracy", accuracy(out, y_valid[:bs]).item())

# -- training --
lr = 0.25
epochs = 10
for epoch in range(epochs):
    for i in range((n - 1) // bs + 1):
        start = i * bs
        end = start + bs
        xb = x_train[start:end]
        yb = y_train[start:end]
        preds = model(xb)
        loss = nll(preds, yb)

        loss.backward()
        with torch.no_grad():
            weights -= weights.grad * lr
            bias -= bias.grad * lr
            weights.grad.zero_()
            bias.grad.zero_()

# -- after --
print("-- after --")
with torch.no_grad():
    out = model(x_valid[:bs])
    print("Valid loss:", nll(out, y_valid[:bs]).item())
    print("Valid accuracy", accuracy(out, y_valid[:bs]).item())

