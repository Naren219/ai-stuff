### how this bigram model works (no NeuralNet)
bigram: using only the previous char to predict the next char. character-level prediction.
1. create a 2d list that has all english characters (and '.'). each element in the table is a certain character order.
2. normalize each row so that you can feed it in a sampler (torch.Generator and torch.multinomial). retrieve each character and store it into a list. 
3. loss: minimizing the average negative log likelihood (lower means the model is performing better).

### how this bigram model works (NeuralNet)
1. dataset: the previous character is fed into x and the next character is y. this structure is repeated across all the names.
2. weights are randomly initialized and inputs are one-hot encoded so they can be fed into the neural network.
3. probabilities calculated from matmul of input and weights.
4. Negative log likelihood loss is calculated using the true next characters and the highest predicted one.

### MLP
1. the first layer looks at 3 previous words and embeds them in n dimensions using a lookup table.
    think of an embedding as a coordinate system that can represent the similarities of different words (through training) through proximity. however, lower dims restrict the info that can be expressed.
2. each embedding set is concatenated and sent through a tanh layer.
3. the final layer holds logits (length is vocab size).
4. logits get softmaxed into a probability distribution.

#### Calculating loss
prob[torch.arange(32), Y] - 
probs holds the probabilities of the next occurring character for each character in x. we use arange in the first index to see if the probability that that specific character appears next is close to the ground truth, determining the loss.

### WaveNet (RNN)
* when increasing the size of the embedding layer, all we do is squash the input into a few neurons, not leading to improved performance. 
* dilated causal convolutional layers - slowly squishing the inputs -- first layer is squishing two characters into a bigram, second layer squishes bigram into one layer, etc.

### you should always be scared of
* tanh or sigmoid nonlinearities can become fully saturated at initialization. we don't want this!
* ReLU gets initialized to 0 or gets a massive update during training, making it permanently dead.
* TODO: vanishing gradients (RNN)

### important concepts
* broadcasting semantics
* regularization loss -> tries to make weights to go zero which allow for uniform distribution. why do we want this?
* we know the model is underfitting when the validation/test loss is close to the test loss. this means the models need to be larger.
* batchnorm - makes a layer's distribution gaussian
* why use nonlinearities? if we didn't have any nonlinearities, all the layers would collapse into a single linear function. this limits a neural net's ability to approximate arbitrary functions, hence we include nonlinearities. 
* TODO: what's the purpose of batches? parallelization

### Links
[Backprop ninja](https://colab.research.google.com/drive/1CJQeyOSecEJ8-lKK8g7QhP_ZEfWz_lQs?usp=sharing) \
[Bigram + MLP](https://colab.research.google.com/drive/187XiSKugmMAYVUM4-5R95_nEmCfhoOAc?usp=sharing) \
[Wavenet](https://colab.research.google.com/drive/1rEEw4A-LlYwJDGWIvTv5MoMvL6rgVsn4?usp=sharing)