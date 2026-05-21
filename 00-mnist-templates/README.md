# Step 0 - Looking inside a digit classifier

Welcome to the very beginning. This is the first project in a 6-step series that teaches mechanistic interpretability from "nothing" to "you can read research papers." If you've never trained a neural network in your life, start here.

Each step is structured as a *notebook*: a written walkthrough up top, then runnable Python code cells further down. You read, you run, you read more.

The goal of this project is to do the single most foundational move in interpretability: train a tiny model, then look directly at its weights and see that they mean something a human can recognise.

Specifically:

- We'll train logistic regression on MNIST (a famous dataset of 70,000 hand-drawn digits, used by everyone learning ML).
- We'll look at the model's weight matrix and see that each row is a fuzzy template of a digit.
- We'll then train a slightly bigger model (1 hidden layer) and look at its weights - they'll be messier and harder to interpret, which is a teaser for project 1 (superposition).

> **Suggested diagram - the whole picture: input, model, output, training loop**
>
> Why generate this: this is the first ML pipeline most readers will ever see. Showing forward pass + training loop in one image makes "what's actually happening when we train" concrete in a way no sequence of bullets can.
>
> Prompt for ChatGPT image generation:
>
> > A two-row technical diagram on a white background, sans-serif labels.
> >
> > TOP ROW (forward pass, left to right):
> > 1. A small 28×28 grid showing a fuzzy hand-drawn "3", labelled "INPUT: MNIST image (28×28)".
> > 2. Arrow labelled "flatten" pointing right to a tall thin vertical column of 784 small squares, labelled "x: input vector (784 numbers)".
> > 3. Arrow into a wide rectangular box labelled "MODEL: one linear layer". Inside the box, two smaller annotations: "W (10×784) - one row per digit class" and "b (10) - one bias per class".
> > 4. Arrow into 10 stacked horizontal bars on the right, each labelled with a digit 0-9, the bar for "3" tallest and highlighted in colour. Labelled "OUTPUT: 10 class scores".
> > 5. Final small arrow labelled "argmax" into a tiny box: "predicted: 3".
> >
> > BOTTOM ROW (training loop), drawn beneath the model box as a feedback loop:
> > • Arrow going down from the 10 class scores into an oval labelled "loss (how wrong was the prediction vs the true label?)".
> > • Arrow labelled "backprop → gradient" going left from "loss" into a rectangle labelled "Adam optimiser".
> > • Arrow labelled "nudge W and b downhill" curling up from "Adam optimiser" back into the model box.
> > • Small annotation below the loop: "repeat ~5,800 times: 5 epochs × ~1,170 mini-batches of 256 examples".
> >
> > Caption underneath: "Training = forward pass, measure wrongness, nudge weights to be less wrong. Repeat thousands of times until the weights settle into digit templates."

That single move - train, look, find something - is what every subsequent step in this series elaborates on.

By the end you'll have:

- Trained your first neural network.
- Visualised what it learnt by reshaping a weight matrix into images.
- Built the mental habit that "weights are not random - they have meaning, you can just look."

---

## Table of contents

1. [What is mechanistic interpretability?](#1-what-is-mechanistic-interpretability)
2. [Glossary - terms you'll meet here](#2-glossary--terms-youll-meet-here)
3. [The experiment in plain English](#3-the-experiment-in-plain-english)
4. [Just enough PyTorch to follow along](#4-just-enough-pytorch-to-follow-along)
5. [Section-by-section walkthrough of the notebook](#5-section-by-section-walkthrough-of-the-notebook)
6. [What you should see when you run it](#6-what-you-should-see-when-you-run-it)
7. [How to run it](#7-how-to-run-it)
8. [Where to go next](#8-where-to-go-next)

---

## 1. What is mechanistic interpretability?

When you train a neural network you get back a model defined by millions (or billions) of numbers - its weights. Nobody hand-designed those numbers. Gradient descent did. So they look, on first inspection, like noise.

Mech interp is the project of opening that "noise" up and discovering that, actually, the weights are doing something specific and understandable. The dream is a sort of "decompiler" for neural networks.

This first step is the simplest possible demonstration that opening the model up is worth doing. We train a tiny model, look at its weights, and see that they encode something a human can recognise.

If that move feels too obvious or too easy - good. The point is that all of mech interp is variations on this move, just applied to messier models with more sophisticated tools. Everything else in the curriculum complicates this picture.

---

## 2. Glossary - terms you'll meet here

If you're brand new to ML, some of these might be your first time seeing the word. Skim now, refer back as needed.

- **Notebook**: an interactive document where prose and runnable Python code cells are interleaved. Each step of this curriculum is rendered as a notebook - the page you're reading right now is one. Code cells have a Run button; everything else is just text.
- **Vector**: an ordered list of numbers. A 28×28 MNIST image becomes a vector of 784 numbers (one per pixel, read row by row). Both the input and the output of a neural network are vectors.
- **Layer**: a chunk of the network that takes a vector in and outputs a different (usually different-sized) vector. Stacking layers is how you build deeper models.
- **Linear layer**: the simplest possible layer - just one matrix multiplication plus a bias: `out = x @ W.T + b`. No nonlinearity. Logistic regression is a single linear layer; an MLP is two or more linear layers with a nonlinearity (like ReLU) between them.
- **Bias**: a small per-output offset added after the matrix multiply. Lets the layer shift its outputs up or down independently of the input.
- **Neural network**: a function from input vectors to output vectors, defined by a collection of learnable numbers (the weights). The function is built by stacking layers (linear maps, nonlinearities) on top of each other.
- **Weights**: the numbers inside the model. The thing training adjusts. Before training they're random; after training they're carefully tuned for the task. Everything we care about in interpretability lives in the weights.
- **Logistic regression**: the simplest kind of neural network. Literally one matrix multiply followed by a softmax. No hidden layer. The whole model is a single weight matrix `W` and a bias vector `b`.
- **MLP (multilayer perceptron)**: logistic regression with one or more "hidden layers" stacked in between input and output, each followed by a nonlinearity. The first non-trivial neural network.
- **Hidden layer**: a layer of neurons between the input and the output. Called "hidden" because it's not directly observed during training - you only see input and output.
- **Neuron**: one scalar value (one number) inside a layer of the network. A layer is a vector of neurons. "100 hidden neurons" means the hidden layer is a 100-number vector. Each neuron's value is a weighted sum of the previous layer, optionally followed by a nonlinearity.
- **Input pixel space**: for MNIST, each image is 28×28 = 784 pixels. The input to the model is a length-784 vector.
- **Class**: one of the 10 possible labels (digit 0 through digit 9). The model predicts a probability distribution over classes.
- **Softmax**: a function that turns any vector of real numbers into a probability distribution (positive, sums to 1). Applied at the output of a classifier.
- **Loss**: a single number that measures how wrong the model is right now. Lower is better. We tune the weights to make it as small as possible.
- **Cross-entropy loss**: the specific loss we use for classification. Penalises low predicted probability for the correct class. You don't need to understand the formula yet; just know it's what you optimise.
- **Gradient**: for each weight in the model, "if I nudged this weight up a tiny bit, would the loss go up or down, and by how much?" The gradient is the collection of those answers, one per weight.
- **Gradient descent**: the algorithm that trains every neural network you'll ever meet. Start with random weights. Repeatedly: feed in data, compute the loss, compute the gradient, then nudge every weight a tiny step in the direction that decreases the loss. Thousands of steps later the weights have "descended" to a low-loss configuration. That's the whole show. Adam, AdamW, SGD - those are all just different recipes for *how big* a step to take.
- **Backpropagation**: the calculus trick that computes the gradient efficiently. In code it's literally one line: `loss.backward()`.
- **Optimiser**: the bit that actually applies the gradient-descent step. We'll use Adam, a sensible default. In code you'll see it spelled `optimizer` because PyTorch uses American spelling.
- **Training loop**: the cycle of (1) forward pass: model makes predictions, (2) compute loss, (3) compute gradients via backpropagation, (4) optimiser updates the weights. Repeat thousands of times.
- **MNIST**: short for *Modified National Institute of Standards and Technology* - a dataset of 70,000 28×28 greyscale images of handwritten digits, each labelled 0-9. Old, small, perfect for tutorials.
- **Template matching**: an algorithm that compares the input to a stored prototype and outputs how similar it is. As you'll see, that's essentially what logistic regression on MNIST does.

---

## 3. The experiment in plain English

There are two experiments in this notebook, back-to-back.

### Experiment A: logistic regression as template matching

Train the simplest possible model on MNIST:

- Input: a 784-dim vector (flattened 28×28 image).
- Model: a single linear layer with 10 outputs. Total weights: `W` of shape `(10, 784)`, plus a 10-dim bias.
- Output: the model picks the class whose score is highest.
- Training: 5 epochs of Adam with mini-batches of 256 (~1,170 steps/epoch). Gets to ~92% test accuracy. Far from state-of-the-art, fine for our purposes.


![Three-step pipeline: weight matrix W of shape (10, 784) with row 3 highlighted; an arrow labelled "reshape (28×28)"; the reshaped row 3 rendered as a 28×28 image showing a fuzzy red template of the digit 3. Caption: each row of W is a digit template; reshaping makes it readable.](diagrams/weight-reshape-pipeline.png)

After training, take the weight matrix `W` of shape `(10, 784)`. Each row is a 784-dim vector - one per digit class. Reshape each row to 28×28 and plot it as a heatmap.

What you'll see: 10 images, each looking like a fuzzy digit. The row for class "0" looks like a hollow circular blob. The row for class "1" looks like a vertical stripe. And so on. These are the templates the model has learnt: it predicts the class whose template most resembles the input.

That's the entire algorithm. Logistic regression on MNIST is template matching. You can read this directly off the weights.

### Experiment B: a tiny MLP, and the first hint of messiness

Now train a slightly bigger model: a 1-hidden-layer MLP (multilayer perceptron) with 32 hidden neurons. The first-layer weights are now a matrix of shape `(32, 784)` - 32 hidden neurons, each with a 784-dim "input filter".

Plot all 32 filters as 28×28 heatmaps. What you'll see: some look like edges and strokes (cleaner features than the templates), but many look messy - like blends of two or three digits, or patterns that don't have an obvious meaning. Some are barely active. Some are clearly multi-purpose.

That messiness is your first encounter with polysemanticity - the phenomenon where one neuron is involved in many features at once. It's the central puzzle of mech interp, and the entire next project (Toy Models of Superposition) is dedicated to understanding why it happens.

---

## 4. Just enough PyTorch to follow along

If you've never trained a model in PyTorch before, this is the place to learn the absolute basics. The notebook itself is heavily commented; this section gives you the conceptual scaffolding.

### Tensors

A tensor is PyTorch's name for a multi-dimensional array (basically a NumPy array that can live on a GPU and track gradients).

```python
x = torch.zeros(3, 4)         # 3×4 tensor of zeros
y = torch.randn(10, 784)      # 10×784 tensor of normal-distributed random numbers
z = x @ y.T                   # 3×10 - @ is matrix multiplication
```

### Models as `nn.Module`

A model is a Python class that inherits from `nn.Module`. You declare the learnable parts in `__init__` and the forward computation in `forward`:

```python
class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.W = nn.Parameter(torch.randn(10, 784) * 0.01)
        self.b = nn.Parameter(torch.zeros(10))

    def forward(self, x):
        return x @ self.W.T + self.b
```

`nn.Parameter` says "this is a learnable weight - track gradients for it, let the optimiser update it." That's the whole machinery.

We can also use `nn.Linear(784, 10)` which packages exactly this in one line.

### The training loop

Every PyTorch training loop has the same skeleton:

```python
for step in range(n_steps):
    logits = model(x)                       # 1. forward pass - predictions
    loss   = F.cross_entropy(logits, y)     # 2. compute loss
    optimizer.zero_grad()                   # 3. clear old gradients
    loss.backward()                         # 4. compute new gradients (backprop)
    optimizer.step()                        # 5. update weights
```

That's it. You don't need to understand backpropagation in detail - `loss.backward()` is one line of code that computes "if I nudged each weight, would the loss go up or down?" The optimiser uses that to take a step in the direction that decreases the loss.

### Two helpers we'll use

- `torch.optim.Adam(model.parameters(), lr=...)` - the standard "smart" optimiser. Treat as a recipe.
- `F.cross_entropy(logits, targets)` - the standard classification loss. Compares predicted scores against integer class labels.

---

## 5. Section-by-section walkthrough of the notebook

The notebook is `mnist_templates.ipynb`.

### Section 1: Setup
Imports + device + seed. Standard plumbing.

### Section 2: Load MNIST
We use `torchvision.datasets.MNIST` to download the dataset (~10MB). We flatten each image to a 784-dim vector and stack them into one big tensor. No `DataLoader` - we just slice random mini-batches from this tensor each step, which keeps the code simple.

### Section 3: Logistic regression
A `LogReg` module with one `nn.Linear(784, 10)`. Train 5 epochs of Adam (mini-batches of 256), log loss and accuracy per epoch.

### Section 4: Visualise the weights as digit templates
Reshape each row of `W` to 28×28 and plot. Each plot has the corresponding digit label. The wow moment.

### Section 5: A 1-hidden-layer MLP
Same training loop but with one hidden layer of 32 ReLU (rectified linear unit) neurons in between. Roughly the same accuracy or slightly better.

### Section 6: Visualise the first-layer weights
Reshape each of the 32 hidden neurons' input weights to 28×28 and plot in a grid. Compare to the clean templates from Section 4. The pivot to project 1.

### Section 7: Discussion
What we just saw. Why it matters. Where it falls short. (Spoiler: the messy MLP weights motivate project 1.)

---

## 6. What you should see when you run it

Concrete expectations so you can tell whether things worked:

1. Logistic regression accuracy: ~91-93% on the test set. Anything in that range is fine.

2. Logistic regression weight visualisation (Section 4): 10 images, each roughly the shape of the corresponding digit. The "0" row looks like a hollow oval. The "1" looks like a tall vertical stripe. The "8" looks like two stacked blobs. They will be fuzzy, not crisp - they're an average of all that digit's training examples.

3. MLP accuracy: ~96-97%. Better than logistic regression because the hidden layer can do more than template matching.

4. MLP first-layer weights (Section 6): 32 heatmaps. Some will look like clear edges or strokes. Others will look like noisy blends. A few may look almost empty. This is the point - the picture isn't as clean as in the logistic-regression case.

---

## 7. How to run it
1. Go to [colab.research.google.com](https://colab.research.google.com).
2. `File → Upload notebook` → upload `mnist_templates.ipynb`.
3. No GPU needed - this runs on CPU in seconds. (GPU is fine too, just unnecessary.)
4. `Runtime → Run all`. Total runtime under.

If running locally: you need `torch`, `torchvision`, `matplotlib`. Conda or pip both fine.

---

## 8. Where to go next

You've just done the most foundational move in mech interp. You're ready for project 1.

The natural question raised by Section 6 is: why are the MLP's hidden-neuron weights so messy? Why doesn't each hidden neuron correspond to one clean feature like the logistic-regression templates do?

The answer is superposition, and project 1 (`../01-toy-models-superposition/`) is dedicated to it. In a tiny toy model with no MNIST or images at all, you'll watch the same messiness emerge and understand exactly why it happens.

After that, each later project adds exactly one new big idea:

- 01 - Toy Models of Superposition: features ≠ neurons (the answer to the puzzle this project just raised)
- 02 - Grokking modular addition: models learn algorithms, not just templates
- 03 - Induction heads: real transformers do work via attention-head circuits
- 04 - IOI circuit: activation patching - proving circuits are causally real
- 05 - Sparse autoencoders: automatically finding features in any model

See the home page for the broader roadmap.
