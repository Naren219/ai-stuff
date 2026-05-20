"""
minimal rnn implementation (based on karpathy's gist). thank you karpathy!

TODO
- allow multiple layers
- implement LSTM too
"""

import numpy as np

# Hyperparameters
HIDDEN_SIZE = 100
SEQ_LENGTH = 25
LEARNING_RATE = 1e-1
WEIGHT_SCALE = 0.01
GRAD_CLIP = 5
PRINT_EVERY = 100
SAMPLE_LENGTH = 200
MAX_ITERS = None  # Set to an integer to limit training, or None for infinite
NUM_LAYERS = 3

# Load data
with open('input.txt', 'r') as f:
    data = f.read()
chars = list(set(data))
data_size, vocab_size = len(data), len(chars)
print('data has %d characters, %d unique.' % (data_size, vocab_size))
char_to_ix = { ch:i for i,ch in enumerate(chars) }
ix_to_char = { i:ch for i,ch in enumerate(chars) }

# Initialize weights
layers = []
Why = np.random.randn(vocab_size, HIDDEN_SIZE) * WEIGHT_SCALE
Wxh = np.random.randn(HIDDEN_SIZE, vocab_size) * WEIGHT_SCALE
by = np.zeros((vocab_size, 1))
for i in range(NUM_LAYERS):
    Whh = np.random.randn(HIDDEN_SIZE, HIDDEN_SIZE) * WEIGHT_SCALE
    Wph = np.random.randn(HIDDEN_SIZE, HIDDEN_SIZE) * WEIGHT_SCALE
    bh = np.zeros((HIDDEN_SIZE, 1))

    layer = {"Whh": Whh, "Wph": Wph, "bh": bh}
    layers.append(layer)

def softmax(x):
    """Numerically stable softmax."""
    e_x = np.exp(x - np.max(x))
    return e_x / np.sum(e_x)

def loss_fun(inputs, targets, hprev):
    xs, hs, ys, ps = {}, {}, {}, {}
    hs[-1] = np.copy(hprev)
    loss = 0

    # FORWARD PASS
    for t in range(len(inputs)):
        xs[t] = np.zeros((vocab_size, 1))
        xs[t][inputs[t]] = 1
        hs[t][0] = np.tanh(np.dot(Wxh, xs[t]) + np.dot(layers[0]['Whh'], hs[t-1][0]) + layers[0]['bh'])
        
        for i in range(1, NUM_LAYERS):
            hs[t][i] = np.tanh(np.dot(layers[i]['Wph'], hs[t][i-1]) + np.dot(layers[i]['Whh'], hs[t-1][i]) + layers[i]['bh'])

        ys[t] = np.dot(Why, hs[t][-1]) + by
        ps[t] = softmax(ys[t])
        loss += -np.log(ps[t][targets[t], 0]) # cross entropy loss

    # BACKWARD PASS
    dWxh, dWhh, dWhy = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why)
    dbh, dby = np.zeros_like(bh), np.zeros_like(by)
    dhnext = np.zeros_like(hs[0][0])
    dhnext = [np.zeros_like(hs[0][0]) for _ in range(NUM_LAYERS)]
    dhraw = [np.zeros_like(hs[0][0]) for _ in range(NUM_LAYERS)]
    for t in reversed(range(len(inputs))):
        dy = np.copy(ps[t])
        dy[targets[t]] -= 1 # d(loss) / d(ys) -> very simple loss formula
        dby += dy
        dWhy += np.dot(dy, hs[t][-1].T)
        dyh = np.dot(Why.T, dy) + dhnext[NUM_LAYERS-1] # d/dx(Why @ hs[t]) = Why[.T]
        dhraw[NUM_LAYERS-1] = (1 - hs[t][-1] ** 2) * dyh # full gradient for the topmost layer

        # Update top layer weights
        layers[NUM_LAYERS-1]['dWph'] += np.dot(dhraw[NUM_LAYERS-1], hs[t][NUM_LAYERS-2].T)
        layers[NUM_LAYERS-1]['dWhh'] += np.dot(dhraw[NUM_LAYERS-1], hs[t-1][NUM_LAYERS-1].T)
        layers[NUM_LAYERS-1]['dbh'] += dhraw[NUM_LAYERS-1]

        # Pass gradient back in time
        dhnext[NUM_LAYERS-1] = np.dot(layers[NUM_LAYERS-1]['Whh'].T, dhraw[NUM_LAYERS-1])

        dh = []
        for i in reversed(range(1, NUM_LAYERS-1)): # skipping the topmost layer
            dh[i] = np.dot(layers[i+1]['Wph'].T, dhraw[i+1]) + dhnext[i]
            dhraw[i] = (1 - hs[t][i]**2) * dh[i] # grad from this layer to previous
            layers[i]['dWph'] += np.dot(dhraw[i], hs[t][i-1].T)  # gradient w.r.t Wph
            layers[i]['dWhh'] += np.dot(dhraw[i], hs[t-1][i].T)  # gradient w.r.t Whh  
            layers[i]['dbh'] += dhraw[i]  # gradient w.r.t bh
            dhnext[i] = np.dot(layers[i]['Whh'].T, dhraw[i])
        
        # Layer 0 (bottom layer - uses Wxh instead of Wph)
        dh = np.dot(layers[1]['Wph'].T, dhraw[1]) + dhnext[0]
        dhraw[0] = (1 - hs[t][0]**2) * dh
        
        layers[0]['dWxh'] += np.dot(dhraw[0], xs[t].T)
        layers[0]['dWhh'] += np.dot(dhraw[0], hs[t-1][0].T)
        layers[0]['dbh'] += dhraw[0]
        dhnext[0] = np.dot(layers[0]['Whh'].T, dhraw[0])

    for dparam in [dWhh, dWhy, dWxh, dbh, dby]:
        np.clip(dparam, -GRAD_CLIP, GRAD_CLIP, out=dparam)
    
    return loss, dWxh, dWhh, dWhy, dbh, dby, hs[len(inputs)-1]

def sample(h, seed_idx, num_chars):
    x = np.zeros((vocab_size, 1))
    x[seed_idx] = 1
    ids = []
    for t in range(num_chars):
        h = np.tanh(np.dot(Wxh, x) + np.dot(Whh, h) + bh)
        y = np.dot(Why, h) + by
        p = softmax(y)
        ix = np.random.choice(range(vocab_size), p=p.ravel())
        x = np.zeros((vocab_size, 1))
        x[ix] = 1
        ids.append(ix)
    return ids

iteration, pointer = 0, 0
mWxh, mWhh, mWhy = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why)
mbh, mby = np.zeros_like(bh), np.zeros_like(by)  # memory variables for Adagrad
smooth_loss = -np.log(1.0/vocab_size) * SEQ_LENGTH  # loss at iteration 0

while MAX_ITERS is None or iteration < MAX_ITERS:
    if pointer + SEQ_LENGTH + 1 >= len(data) or iteration == 0:
        hprev = np.zeros((HIDDEN_SIZE, 1))
        pointer = 0
    inputs = [char_to_ix[ch] for ch in data[pointer:pointer+SEQ_LENGTH]]
    targets = [char_to_ix[ch] for ch in data[pointer+1:pointer+SEQ_LENGTH+1]]

    if iteration % PRINT_EVERY == 0:
        sample_idx = sample(hprev, inputs[0], SAMPLE_LENGTH)
        txt = ''.join(ix_to_char[ix] for ix in sample_idx)
        print('----\n %s \n----' % (txt,))

    loss, dWxh, dWhh, dWhy, dbh, dby, hprev = loss_fun(inputs, targets, hprev)
    smooth_loss = smooth_loss * 0.999 + loss * 0.001
    if iteration % PRINT_EVERY == 0:
        print('iter %d, loss: %f' % (iteration, smooth_loss))

    for param, dparam, mem in zip([Wxh, Whh, Why, bh, by],
                                  [dWxh, dWhh, dWhy, dbh, dby],
                                  [mWxh, mWhh, mWhy, mbh, mby]):
        mem += dparam ** 2
        param += -LEARNING_RATE * dparam / np.sqrt(mem + 1e-8)

    pointer += SEQ_LENGTH
    iteration += 1
