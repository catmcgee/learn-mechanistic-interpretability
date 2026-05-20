# Step 5 — Sparse autoencoders: closing the loop on superposition

Step 5 of 6 in the mech interp curriculum. Assumes you've done steps 0-4. (You especially need step 1, which set up the problem this step solves.)

We replicate the core idea of [Sparse Autoencoders Find Highly Interpretable Features in Language Models](https://arxiv.org/abs/2309.08600) (Cunningham, Ewart, Riggs, Huben, Sharkey; 2023) and [Towards Monosemanticity](https://transformer-circuits.pub/2023/monosemantic-features/index.html) (Bricken et al., Anthropic, 2023).

The ONE new idea this step teaches: you can automatically discover the features a model uses, by training a separate "interpreter" model (a sparse autoencoder) on the original model's hidden activations. No hand-crafted prompts, no head-by-head patching — just train an SAE and read the dictionary it learns.

This step closes the loop on the whole curriculum. Step 1 introduced superposition as the central obstacle of mech interp. Step 5 introduces SAEs, the field's current best partial solution.

By the end you'll have:

- Trained a toy model that uses superposition (a slightly bigger version of step 1's setup).
- Trained a sparse autoencoder on the model's hidden activations.
- Verified that the SAE's learnt features recover the original model's ground-truth features — by cosine similarity, not by hand.

---

## Table of contents

1. [The big picture](#1-the-big-picture)
2. [Glossary — terms added in this step](#2-glossary--terms-added-in-this-step)
3. [What is a sparse autoencoder?](#3-what-is-a-sparse-autoencoder)
4. [The experiment in plain English](#4-the-experiment-in-plain-english)
5. [Section-by-section walkthrough of the notebook](#5-section-by-section-walkthrough-of-the-notebook)
6. [What you should see when you run it](#6-what-you-should-see-when-you-run-it)
7. [How to run it](#7-how-to-run-it)
8. [Where this leaves you](#8-where-this-leaves-you)

---

## 1. The big picture

Recap of the whole curriculum so you can see how step 5 fits:

- Step 0: weights are interpretable — you can just look.
- Step 1: but with more features than neurons, the model represents features as non-orthogonal directions (superposition). Neurons stop being interpretable. Problem identified.
- Step 2: real models learn algorithms (e.g. Fourier modular arithmetic) and store the features for those algorithms in superposition.
- Step 3: in transformers, work is done by circuits of attention heads. We found one by hand (induction).
- Step 4: we can verify circuits causally via activation patching. But this is still slow, manual, hypothesis-driven.
- Step 5 (this one): can we automate it? Can we get the features out of a trained model with one big sweep, without knowing in advance what we're looking for? The current best answer is sparse autoencoders.

The pitch for an SAE in one sentence: train a wider autoencoder on the model's hidden activations, with a sparsity penalty, and the directions it learns will be the model's actual features.

---

## 2. Glossary — terms added in this step

- **Sparse autoencoder (SAE)**: an autoencoder whose hidden layer is wider than its input, trained with a penalty that forces only a few hidden units to fire on any given input. The "wider hidden than input" part is essential: if the hidden layer is the same size as the input, the obvious solution is the identity function and no features get found.
- **Dictionary learning**: the general field of "find a sparse representation of data as combinations of basis elements." SAEs are a specific neural-network version. The "dictionary" is the set of basis elements (= the SAE's decoder columns).
- **Encoder / decoder**: the two halves of an autoencoder. Encoder maps input → hidden (the dictionary coefficients). Decoder maps hidden → output (a reconstruction).
- **L1 penalty / L1 loss**: a loss term proportional to the sum of absolute values of the hidden activations. Penalises any nonzero activation, so the optimiser prefers solutions that use as few hidden units as possible.
- **Reconstruction loss**: the standard MSE between the SAE's output and its input. The SAE has to faithfully reproduce the original activation.
- **Feature direction (in an SAE)**: a column of the decoder weight matrix. The SAE represents each input as a sparse positive combination of these directions.
- **Monosemantic feature**: a feature direction that, by inspection, corresponds to one clean concept. The interpretability goal. Project 1 defined this term; step 5 makes it operational.
- **Feature splitting**: when a wider SAE breaks one "feature" from a smaller SAE into several finer ones. A phenomenon you'll see in larger setups; we don't focus on it.
- **Sparsity coefficient (λ)**: the weight on the L1 penalty in the SAE's loss. Higher λ → fewer features active per input but worse reconstruction. Picking this is the main SAE-training hyperparameter headache.

---

## 3. What is a sparse autoencoder?

A sparse autoencoder is a small neural network with three pieces:

```
input x (d_input dims)
       │
       ▼
   ┌──────────┐
   │ encoder  │  W_enc, b_enc  → ReLU
   └──────────┘
       │
       ▼
hidden h (d_sae dims)   ← d_sae > d_input  (the dictionary; many more "features" than the input has dimensions)
       │
       ▼
   ┌──────────┐
   │ decoder  │  W_dec
   └──────────┘
       │
       ▼
output x̂ (d_input dims)
```

Forward pass:

```
h = ReLU(W_enc @ x + b_enc)
x̂ = W_dec @ h + b_dec
```

Loss:

```
L = ||x - x̂||²    (reconstruction — must faithfully decompose the input)
  + λ ||h||₁      (sparsity — penalises the number of features used)
```

The ReLU on `h` is essential — it means each feature can be off (`h_i = 0`) or active to varying degrees (`h_i > 0`), but never negative. Combined with the L1 penalty, the SAE is pushed toward solutions where most features are off and only a few fire on any given input.

After training, each column of `W_dec` is one feature direction in the input activation space. The SAE has learnt a dictionary of `d_sae` directions, and represents each input activation as a sparse, non-negative combination of those directions.

This is what fixes the superposition problem. The original model packs many features into fewer dimensions (project 1). The SAE projects back out into a higher-dimensional space where each feature can have its own dedicated dimension.

![Two-panel diagram. Left: "Base model — features in superposition" — 10 input features collapse into a 5-neuron bottleneck via overlapping coloured arrows, visually crowded; caption: 10 features packed into 5 neurons, each neuron carries pieces of many features. Right: "SAE — features unfolded" — the same 5 neurons feed a 20-feature SAE hidden layer via clean, sparse connections; most SAE features are grey (inactive) with only 1-2 highlighted; caption: SAE fans out the superposed features into a sparse dictionary, now each feature has its own slot. Title: sparse autoencoders recover features from a model trained in superposition.](diagrams/sae-vs-superposition.png)

---

## 4. The experiment in plain English

We want to verify, in a setting where we have ground truth, that an SAE actually finds the real features.

The setup: train a project-1-style toy superposition model. Specifically:

- `n_features = 10` (the ground-truth features)
- `n_hidden = 5` (the bottleneck dimension)
- `sparsity = 0.9` (90% of features off on average)

This model packs 10 features into 5 hidden dimensions. We know the ground-truth feature directions: they're the 10 columns of the model's `W` matrix.

The SAE: train a sparse autoencoder on the model's hidden activations.

- Input: 5-dim (the model's hidden state)
- SAE width: 20 features (overcomplete by 4×)
- Sparsity: L1 coefficient `λ = 0.001`

The test: after the SAE has trained, compute the cosine similarity between each SAE feature direction and each ground-truth feature direction. Build a 20×10 matrix and plot it. If the SAE has recovered the features:

- Each ground-truth feature should be matched by at least one SAE feature with cosine similarity ≈ 1.
- The SAE features not matched to any ground-truth feature are "dead" or redundant — that's normal.

This is the wow moment: the SAE didn't know about the 10 ground-truth features. It only ever saw 5-dim hidden activations from a trained model. With nothing but the L1 penalty as guidance, it recovers the original feature decomposition.

---

## 5. Section-by-section walkthrough of the notebook

The notebook is `sparse_autoencoders.ipynb`.

### Section 1: Setup
Imports, device, seed.

### Section 2: Train a superposition model
A slightly bigger version of step 1's `ToyModel`. We train it briefly at high sparsity so it packs 10 features into 5 hidden dimensions via superposition.

### Section 3: Collect hidden activations
Sample a big batch of sparse inputs, push them through the model's encoder, get a `(n_samples, 5)` tensor of hidden activations. This is what the SAE will be trained on.

### Section 4: The SAE class
Two `nn.Parameter`s for encoder and decoder weights plus biases. Forward pass is ReLU encoder → linear decoder, as described above. We tie decoder columns to have unit norm (a small trick that prevents the L1 penalty from being "cheated" by shrinking hidden activations and growing decoder weights).

### Section 5: Train the SAE
AdamW + cosine schedule. We log reconstruction loss and L0 (the average number of active features per input). Healthy L0 is around the number of features the input is a mixture of — for high-sparsity inputs, that's ~1.

### Section 6: Compare SAE features to ground truth
Compute cosine similarity between every pair of (SAE feature direction, ground-truth feature direction). Plot the 20×10 matrix as a heatmap.

### Section 7: Visualise the matched features
For each ground-truth feature, find its best-matching SAE feature and plot the two directions side by side. They should look nearly identical.

### Section 8: Discussion
What we just did, why it matters, and the realities of using SAEs on real LLMs (it's much messier).

---

## 6. What you should see when you run it

1. Toy model training (Section 2): loss drops to ~0.001 over a few thousand steps. The model has learnt to compress 10 features into 5 dims using superposition.

2. SAE training (Section 5): reconstruction loss drops to near zero; average L0 (number of active SAE features per input) is around 1-2, matching the input sparsity.

3. Cosine similarity heatmap (Section 6): a 20×10 matrix where each ground-truth feature has at least one SAE feature with cosine similarity ≥ 0.95 (a near-perfect match). The matrix is sparse — most cells near zero, a clear "highway" of bright cells.

4. Matched features plot (Section 7): for each of the 10 ground-truth features, an SAE feature that points in the same direction in 5-dim space.

If anything fails to converge, try increasing `n_sae_steps` to 10000 or adjusting `l1_coef`. SAEs are finicky.

---

## 7. How to run it
1. Go to [colab.research.google.com](https://colab.research.google.com).
2. Upload `sparse_autoencoders.ipynb`.
3. GPU is fine but unnecessary — this whole notebook runs on CPU in a few minutes.
4. `Runtime → Run all`.

---

## 8. Where this leaves you

You have now done, by hand, every major move in mech interp through 2026:

- Trained models and read their weights (step 0).
- Discovered superposition in toy models (step 1).
- Reverse-engineered an algorithmic circuit (step 2).
- Found a multi-head circuit in a transformer (step 3).
- Causally verified circuits in a real pre-trained LLM (step 4).
- Automatically recovered features from a trained model (step 5).

That's the toolkit. From here it's about scale, rigour, and applying it to questions that matter.

### What you can't yet do, and what to learn next

- Run SAEs at GPT-2-or-bigger scale. The toy SAE here trains in seconds. Real SAEs on real LLMs take hours, are picky about hyperparameters, and need careful evaluation. Tool: [SAELens](https://github.com/jbloomAus/SAELens), the standard SAE training library.
- Distinguish "real features" from "SAE artefacts." Two different SAEs trained on the same data find slightly different features. Which decomposition is the right one? Active research question.
- Build full circuit-level explanations from SAE features. Connecting features-as-directions to specific circuits is "feature circuit" research. Anthropic's [Circuit Tracing](https://transformer-circuits.pub/2025/attribution-graphs/methods.html) work is the current frontier.
- Read frontier papers. With this background, you can. Start with the Anthropic Transformer Circuits Thread and follow citations.

### Further reading

- [Towards Monosemanticity](https://transformer-circuits.pub/2023/monosemantic-features/index.html) — Anthropic 2023. The first big "yes, SAEs really do find interpretable features on a real LM" paper.
- [Sparse Autoencoders Find Highly Interpretable Features in Language Models](https://arxiv.org/abs/2309.08600) — Cunningham et al. 2023. The academic version.
- [Scaling Monosemanticity](https://transformer-circuits.pub/2024/scaling-monosemanticity/) — Anthropic 2024. SAEs on a production-scale model. Source of the famous "Golden Gate Bridge feature".
- [Anthropic Circuits Updates 2024-2025](https://transformer-circuits.pub/2024/april-update/index.html) — short, frequent posts about ongoing interp work. Read these to stay current.
- [Neel Nanda's "200 Concrete Open Problems"](https://www.alignmentforum.org/posts/LbrPTJ4fmABEdEnLf/200-concrete-open-problems-in-mechanistic-interpretability) — pick one and try it.

### Where to participate

- [AlignmentForum / LessWrong](https://www.alignmentforum.org/) — most public discussion happens here.
- [MATS](https://www.matsprogram.org/) — competitive research fellowship, common pipeline from self-taught to professional interp researcher.
- [ARENA](https://www.arena.education/) — structured curriculum with more advanced exercises if you want to go deeper than these projects.

You're now ~6 months into mech interp by self-study, with a solid foundation. The next step is choosing one of the open problems from the top-level `README.md` and trying a small experiment of your own. That's how everyone in this field actually got started.
