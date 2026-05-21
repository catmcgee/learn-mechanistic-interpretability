# Grokking modular addition

This is project 2 in our mech interp series. We replicate the core finding of [Progress Measures for Grokking via Mechanistic Interpretability](https://arxiv.org/abs/2301.05217) (Nanda, Chan, Lieberum, Smith, Steinhardt - 2023), one of the most striking results in mech interp.

By the end you'll have:

- Trained a tiny 1-layer transformer on a simple maths problem and watched it do something genuinely strange: sit at ~chance performance for thousands of training steps, then suddenly "wake up" and generalise perfectly.
- A mechanistic explanation for that delayed generalisation - you'll look inside the trained model and discover it learnt a trigonometric algorithm based on the **Fourier basis**. ("Fourier" is the idea that any repeating pattern can be written as a sum of sines and cosines at different frequencies. In this project, the model uses sines and cosines to represent numbers as rotations around a circle - because the task wraps modulo 113, "rotations" is the natural way to think about it.)
- Your first transformer, built from scratch in PyTorch.

This is the spiritual successor to project 1 (Toy Models of Superposition). Same recipe - train a small model from scratch, then reverse-engineer it - but now we get to use a transformer.

---

## Table of contents

1. [What is grokking?](#1-what-is-grokking)
2. [The task: modular addition](#2-the-task-modular-addition)
3. [Glossary - terms you'll see throughout](#3-glossary--terms-youll-see-throughout)
4. [The experiment in plain English](#4-the-experiment-in-plain-english)
5. [Just enough transformer to follow along](#5-just-enough-transformer-to-follow-along)
6. [Section-by-section walkthrough of the notebook](#6-section-by-section-walkthrough-of-the-notebook)
7. [What you should see when you run it](#7-what-you-should-see-when-you-run-it)
8. [How to run it](#8-how-to-run-it)
9. [Where to go next](#9-where-to-go-next)

---

## 1. What is grokking?

In normal supervised learning, a model's training loss and test loss both improve together. The model gets better at the data it has seen and better at unseen data, in lockstep, roughly.

Grokking is what happens when those two curves come apart in a really weird way:

1. Memorisation phase (steps 0 → ~1k): training loss drops to near zero - the model has memorised the answers to every training example. Test loss stays at chance level - the model has learnt nothing generalisable.
2. Plateau (steps ~1k → ~10k+): training loss stays low, test loss stays bad. Looks like the model has just memorised and that's it. For a long, long time.
3. Grokking (suddenly, much later): test loss falls off a cliff and reaches near-zero. The model has, somehow, figured out the underlying rule and now generalises perfectly - long after it could have just stopped at memorisation.

![Log-scale loss curve showing the three phases of grokking. A blue train-loss curve drops sharply within the first ~500 steps and stays flat near zero. An orange test-loss curve stays high for a long plateau, then drops off a cliff around step ~15,000. Three labelled background bands: Memorisation (~0–1k), Plateau (~1k–15k), Grokking (~15k+).](diagrams/grokking-loss-curve.png)

This was first reported by [Power et al. 2022](https://arxiv.org/abs/2201.02177) ("Grokking: Generalisation Beyond Overfitting on Small Algorithmic Datasets") at OpenAI. They observed it but had no mechanistic explanation.

Nanda et al.'s 2023 paper took the same setup, fully reverse-engineered the trained model, and showed exactly what was happening: during the "boring" plateau, the model was slowly building up a generalising circuit alongside the memorised solution. When the circuit got good enough, it took over.

That circuit is what makes this whole thing beautiful - it's a clean, human-understandable algorithm based on Fourier series. We'll find it.

---

## 2. The task: modular addition

The model learns to add two numbers modulo a prime `p`.

Pick `p = 113`. The model sees three input tokens - `a`, `b`, `=` - where `a, b ∈ {0, 1, ..., 112}`, and has to predict the answer `(a + b) mod 113` at the `=` position.

Examples:

- input `(0, 0, =)` → answer `0`
- input `(5, 10, =)` → answer `15`
- input `(100, 50, =)` → answer `37`  (because `150 mod 113 = 37`)
- input `(112, 1, =)` → answer `0`  (the "wrap-around")

There are `113 × 113 = 12,769` total possible `(a, b)` pairs. We train on a random 30% of them and test on the remaining 70%. With this setup, the model has plenty of room to overfit (memorise the 30%) but only generalises if it learns the actual rule.

The choice of `p` being prime matters - it makes the maths have nice Fourier structure, which is what the model ends up exploiting.

---

## 3. Glossary - terms added in this step

Just the new ones. For earlier terms (feature, direction, dimension, superposition) see the step 1 glossary.

**Grokking-specific**

- **Grokking**: late, delayed generalisation. The phenomenon this project is about.
- **Memorisation**: the model perfectly fits training data without learning a rule. Looks like learning but doesn't transfer to new examples.
- **Generalisation**: the model performs well on examples it has never seen - evidence that it learnt the underlying rule, not the table of answers.
- **Train / test split**: a random partition of the data. The model only ever sees the train half during training; the test half is used to measure whether it actually learnt a generalising rule.
- **Weight decay**: a regularisation technique that pulls weights toward zero. Crucial for grokking - without weight decay, grokking does not happen. We'll explain why below.
- **AdamW**: the variant of the Adam optimiser that implements weight decay correctly. Just use it; you don't need to understand the details. (In code you'll see it spelt `optimizer` because PyTorch uses American spelling.)

**Transformer anatomy**

- **Token**: a single discrete input symbol. In our setup, the vocabulary is 114 tokens (`0`, `1`, …, `112`, `=`), and each input sequence has 3 tokens (`a`, `b`, `=`).
- **Vocabulary** (`d_vocab`): how many distinct tokens exist. For us, `114`.
- **Embedding** (`W_E`): a learnable lookup table that maps each token to a vector of size `d_model`. The very first thing the model does to your input.
- **Positional embedding** (`W_pos`): a small learnable vector added per position in the sequence - lets the model tell apart "this is the first input" vs "this is the second input."
- **Residual stream**: the running vector at each position as it flows through the layers. The transformer's "main bus" - every component reads from it and writes back into it.
- **Attention head**: a component that lets the model mix information across positions. We have 4 heads in 1 layer.
- **MLP** (multilayer perceptron): a simple stack of linear layers with a nonlinearity in between (`linear → ReLU → linear`). Sits after attention and does per-position computation.
- **Unembedding** (`W_U`): a final linear layer that maps the `d_model`-sized vector back to `d_vocab` logits - one score per possible output token. The model picks the one with the highest score.
- **Logits**: the raw, unnormalised scores the model outputs. Softmax of the logits gives a probability distribution over tokens.
- **Cross-entropy loss**: the standard classification loss - penalises low predicted probability for the correct answer.

**Mech interp / Fourier**

- **Fourier basis**: a way of writing any function on `{0, 1, ..., p-1}` as a sum of sines and cosines at different frequencies. The mathematical structure the model ends up using.
- **Frequency**: an integer `k` indexing one of the trig functions `cos(2πkx/p)`, `sin(2πkx/p)`. With `p=113`, valid frequencies are `0` through `56`.
- **Progress measure**: a quantity you can compute from the model's weights that smoothly increases even during the "flat" plateau. Reveals that the model is slowly building its circuit underneath, even when the test loss isn't dropping.

---

## 4. The experiment in plain English

The whole experiment, before any code:

The model. A 1-layer transformer with 4 attention heads, hidden size `d_model=128`, MLP hidden size `d_mlp=512`. No layer norm, no dropout (we want the cleanest possible mech-interp target). Total parameters: ~200k. Tiny by modern standards but plenty for this task.

The data. All `113 × 113 = 12,769` pairs `(a, b)`, each labelled with `(a+b) mod 113`. We split randomly into 30% train, 70% test.

Training. AdamW with learning rate `1e-3` and weight decay 1.0 (this is large - weight decay is doing serious work here). Full-batch (we use all 3,830 training examples per step). Cross-entropy loss on the prediction at the `=` position.

What we plot. Train loss and test loss vs step, both on a log scale, for ~25,000 steps. You should see:

- Train loss crashes to near zero within the first ~500 steps. (Memorisation)
- Test loss stays bad - close to `log(113) ≈ 4.7`, which is the loss of random guessing among 113 classes - for a long time.
- Then, somewhere around step 10,000–20,000, test loss falls off a cliff and reaches near zero. (Grokking - the wow moment.)

The reverse-engineering (stretch). We take the trained model's embedding matrix `W_E` (shape `(d_vocab, d_model)` - one vector per number `0..112` plus `=`). We do a discrete Fourier transform along the vocab dimension. We discover that almost all of the embedding energy lives in just a handful of Fourier frequencies - the model is representing each number as `cos(2πkx/p)` and `sin(2πkx/p)` for a small set of `k`. This is the Fourier algorithm the model has discovered, sitting right there in the embedding weights.

Why does this work mathematically? The trig identity:

```
cos(2π·k·(a+b)/p) = cos(2π·k·a/p) · cos(2π·k·b/p) − sin(2π·k·a/p) · sin(2π·k·b/p)
```

If the model represents `a` and `b` as Fourier features, an MLP can implement that identity using multiplications and combinations. Then the unembedding reads out the answer at the right frequency. The whole thing is a clean, human-understandable algorithm - built by SGD, not by us.

![Four-step pipeline: (1) integer inputs a and b, (2) each embedded onto a unit circle at (cos(2π·k·x/113), sin(2π·k·x/113)) - the Fourier basis, (3) MLP combines them via cos(α+β) = cos(α)cos(β) − sin(α)sin(β), (4) result lands on the unit circle at angle 2π·k·(a+b)/113. Unembedding reads the angle to produce the predicted answer. Caption: the model discovered a Fourier algorithm for modular addition.](diagrams/fourier-algorithm.png)

---

## 5. Just enough transformer to follow along

You did project 1, so you already know `nn.Module`, `nn.Parameter`, the training-loop skeleton, ReLU, loss, optimiser. The new concept here is the transformer.

A transformer is just a particular arrangement of standard neural-net pieces (linear layers, ReLU, softmax). Here's the entire architecture in one diagram, conceptually:

> **Suggested diagram - the transformer, top to bottom**
>
> Why generate this: the ASCII version of this is fiddly to read and breaks on small screens. A clean image makes "data flows down through a stack of layers, each layer reads and writes the residual stream" obvious.
>
> Prompt for ChatGPT image generation:
>
> > A clean vertical flow diagram on a white background, read top to bottom. Sans-serif labels. Use rounded rectangles for layers and rectangles with dashed borders for the residual stream tensor between layers. Single straight arrows pointing down between every block.
> >
> > 1. At the top: a small horizontal row of three coloured token squares labelled "a", "b", "=". Caption above: "input tokens".
> > 2. Arrow down to a rounded rectangle labelled "embedding". To the right of the box, a small caption: "W_E: lookup table, each token → 128-dim vector. Plus W_pos: per-position offset."
> > 3. Arrow down to a dashed-border rectangle labelled "residual stream — shape (3, 128): 3 positions × 128 features each". This is the central object; draw it slightly wider than the layer boxes.
> > 4. Arrow down to a rounded rectangle labelled "attention". Caption to the right: "4 heads. Moves information between positions. Writes output back into the residual stream."
> > 5. Arrow down to another dashed-border rectangle labelled "residual stream (updated)".
> > 6. Arrow down to a rounded rectangle labelled "MLP". Caption to the right: "linear(128 → 512) → ReLU → linear(512 → 128). Per-position computation."
> > 7. Arrow down to another dashed-border rectangle labelled "residual stream (updated again)".
> > 8. Arrow down to a rounded rectangle labelled "unembedding". Caption to the right: "W_U: linear(128 → 114). One score per possible answer."
> > 9. Arrow down to a small final block labelled "logits → softmax → predicted answer (read off at the `=` position)".
> >
> > Visual hint: colour the three dashed "residual stream" rectangles the same pale colour to emphasise that they are the same object passing through, just with more information added at each step.

Three concepts you need to internalise:

### 5.1 The residual stream

The transformer keeps a running `(n_positions, d_model)` tensor that flows through the layers. For us this is `(3, 128)` - three positions (one for `a`, one for `b`, one for `=`), each holding a 128-dim vector.

Every component (embedding, attention, MLP) reads from this tensor and adds its output back into it. Nothing is destructively overwritten. This is the "residual" part - outputs are added, not replaced.

Think of the residual stream as the model's working memory: a shared whiteboard that every component reads from and writes to.

### 5.2 Attention

An attention head's job is to move information between positions. In our setup, it's the only mechanism that lets the model combine information about `a` (at position 0) and `b` (at position 1) into a single computation at the `=` position (position 2). The MLP, by contrast, only operates within one position at a time.

Each head has four small matrices: `W_Q` (query), `W_K` (key), `W_V` (value), `W_O` (output). The mechanics:

1. For each position, the head computes a query (`q = W_Q @ residual`) and a key (`k = W_K @ residual`).
2. For each pair of positions, it computes an attention score: `score[i, j] = q_i · k_j`.
3. It softmaxes the scores along the source dimension to get attention weights (each position's "attention" sums to 1).
4. It computes values (`v = W_V @ residual`) and outputs a weighted sum of values, weighted by the attention weights.
5. It projects back via `W_O` and adds the result into the residual stream.

You don't have to memorise this. The mantra is: attention moves information between positions; everything else runs within a position.

### 5.3 Weight decay (and why it matters for grokking)

Weight decay is a regularisation technique: at every optimiser step, in addition to the gradient update, it pulls all weights slightly toward zero. Mathematically: `w ← w - lr · gradient - lr · weight_decay · w`.

Why does this matter for grokking? Roughly: there are two ways to solve the task.

- Memorisation: store a giant lookup table of (a, b) → answer in the weights. This works perfectly on the train set but uses a lot of weight "mass."
- Generalisation: learn a small, clean Fourier algorithm. Uses very little weight mass.

Without weight decay, the model is happy to sit at memorisation forever - both solutions get loss 0 on train, and there's no pressure to find the cleaner one. With weight decay, big-weight memorisation is penalised. The model has to slowly carve out the smaller, cleaner generalising solution. That carving is what we see as the long plateau, and grokking is the moment the cleaner solution becomes good enough to take over.

This is why the paper is called Progress Measures - even during the plateau, weight decay is slowly squeezing weight mass from the memorised solution into the generalising one. The progress is real but invisible in the loss curves until the crossover.

---

## 6. Section-by-section walkthrough of the notebook

The notebook is `grokking.ipynb`. Each section below maps to a section in the notebook.

### Section 1: Setup
Imports, device, seeds. Same as before.

### Section 2: Hyperparameters
A single cell of constants - `p = 113`, `d_model = 128`, `n_heads = 4`, `d_head = 32`, `d_mlp = 512`, `train_frac = 0.3`, `weight_decay = 1.0`, `n_steps = 25_000`, etc. Keeping them in one place makes experiments easier.

### Section 3: The data
Generate all `p²` pairs `(a, b)` with their labels `(a+b) mod p`. Format each example as a length-3 sequence `[a, b, p]` where token `p` is used as the `=` marker (we just stuff `=` into the same vocab as a 114th token at index `p`). Random shuffle, take the first 30% as train.

### Section 4: The transformer
A hand-rolled tiny transformer in ~50 lines of PyTorch. We define:

- An `Embedding` (token + position).
- An `Attention` layer with 4 heads. We use raw `Q, K, V, O` linear layers; you'll be able to inspect them later.
- An `MLP` layer (linear → ReLU → linear).
- A `Transformer` class that stacks them, applies the unembedding, returns logits at the `=` position.

Hand-rolled is the mech-interp tradition - using `nn.Transformer` would hide the internals we want to inspect.

### Section 5: Training
Standard loop, AdamW, cross-entropy on the `=`-position logits. Every 100 steps we record both train loss and test loss, so we can plot them at the end. ~25k steps takeson a Colab T4.

### Section 6: The grokking curve
The headline plot. Train loss vs test loss vs step, log scale. This is the wow moment. You'll see the train loss crash to zero almost immediately, then a long flat plateau on test loss, then a sudden cliff. We mark the rough memorisation phase, plateau, and grokking phase on the plot.

### Section 7 (stretch): Fourier analysis of the embedding
We take the trained `W_E` matrix and do an FFT along the vocabulary axis. We then plot the magnitude of each Fourier component across hidden dimensions. The plot should show that energy is concentrated in just a handful of frequencies - the model has rediscovered the Fourier basis.

### Section 8: Discussion
Why what you just saw matters. Why the paper is called "Progress Measures." How it connects to the broader story about generalisation, simplicity bias, and singular learning theory.

---

## 7. What you should see when you run it

Concrete expectations:

1. Train loss curve (Section 6): drops from ~`log(113) ≈ 4.7` to under `0.01` within the first few hundred steps. Then stays near zero forever.

2. Test loss curve (Section 6): stays near `4.7` (random-guess level) for thousands of steps, then drops suddenly - over a few hundred steps - to under `0.01`. The grokking transition is sharp, not gradual.

3. When does grokking happen? (Section 6) Somewhere between step 8,000 and 20,000 with our hyperparameters. The exact step varies with the random seed; that's normal. If you're past step 25,000 and test loss is still high, weight decay or learning rate is probably off.

4. Fourier plot (Section 7, stretch): the FFT of `W_E` should be sparse - most frequencies near zero, but a handful (typically 3-6) of "key frequencies" with large magnitudes. Those are the trig functions the model has learnt to use.

---

## 8. How to run it
1. Go to [colab.research.google.com](https://colab.research.google.com).
2. `File → Upload notebook` → upload `grokking.ipynb`.
3. `Runtime → Change runtime type → GPU` (T4 is fine and free).
4. `Runtime → Run all`. The model trains for; the rest takes seconds.

If you want to skip training and just play with the analysis, you could save the trained weights at the end of Section 5 and reload them - we don't bother with that here but it's a one-liner.

---

## 9. Where to go next

Next in the curriculum: [`03-induction-heads/`](../03-induction-heads/) - your first multi-layer transformer, and your first reverse-engineered circuit (two attention heads cooperating across layers). The transformer code from this project carries over almost unchanged.

Other things to try with the model you just built:

- Compute proper progress measures: the paper introduces three of them (restricted loss, excluded loss, Gini coefficient of Fourier components). They smoothly track the slow build-up of the circuit. Plotting them during training instead of post-hoc makes the "hidden progress" story really vivid.
- Reverse-engineer the MLP and the attention head: the embedding is just the start. Neel's full writeup shows that the MLP implements the trig product-to-sum identity, and the attention head reads out the right Fourier component at the `=` position. Full circuit-level interp.
- Try other algorithmic tasks: multiplication mod p, group operations, sorting. Grokking shows up across many algorithmic tasks; the underlying circuits differ.
- Connect to project 1: the Fourier features in the embedding are themselves in superposition - multiple frequencies sharing 128 hidden dimensions. Both projects are looking at the same phenomenon (compressed feature representations) from different angles. Step 5 of this curriculum closes the loop by training an SAE to recover features from a superposed model.

### Reading

- The paper itself: [Progress measures for grokking via mechanistic interpretability](https://arxiv.org/abs/2301.05217).
- Neel Nanda's [blog post version](https://www.alignmentforum.org/posts/N6WM6hs7RQMKDhYjB/a-mechanistic-interpretability-analysis-of-grokking) - more accessible than the paper.
- Neel's [grokking demo Colab](https://colab.research.google.com/github/neelnanda-io/Easy-Transformer/blob/main/Grokking_Demo.ipynb) - the original reference implementation.
- [Power et al. 2022](https://arxiv.org/abs/2201.02177) - the original "grokking" paper that observed the phenomenon.
