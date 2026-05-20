# Mechanistic Interpretability — a hands-on curriculum

A 6-step open-source learning path from "I've never trained a neural network" to "I can read current mech interp papers" (well, almost). Each step is a self-contained project that replicates a famous experiment, teaches one new big idea, and assumes only what previous steps already covered.

Inspired by Neel Nanda :)

---

## Table of contents

1. [What is interpretability?](#what-is-interpretability)
2. [What is mechanistic interpretability?](#what-is-mechanistic-interpretability)
3. [The curriculum](#the-curriculum)
4. [How each project is structured](#how-each-project-is-structured)
5. [What you'll be able to do at the end](#what-youll-be-able-to-do-at-the-end)
6. [The open problems — what counts as "graduated"](#the-open-problems--what-counts-as-graduated)
7. [Reading list](#reading-list)
8. [Communities](#communities)

---

## What is interpretability?

When you train a neural network, you end up with millions or billions of numbers - weights - that produce intelligent-seeming behaviour. Nobody designed those numbers. Gradient descent did. So a trained model is a bit of a black box: it works, but nobody really knows why or how.

Interpretability is the broad field that tries to fix that. Any method that helps a human understand or predict what a trained model is doing counts. In practice the field splits into several schools:

| School                                               | One-line description                                                                         | Typical methods                                            |
| ---------------------------------------------------- | -------------------------------------------------------------------------------------------- | ---------------------------------------------------------- |
| Behavioural                                          | Study only the model's inputs and outputs. Treat it as a black box.                          | Red-teaming, prompt probing, capability evals              |
| Attribution / saliency                               | For a given output, attribute it back to inputs or neurons.                                  | Integrated gradients, SHAP, attention rollouts             |
| Probing                                              | Train a small separate classifier on hidden activations to test what info is there.          | Linear probes, concept bottleneck models                   |
| Representation engineering / activation steering     | Find directions in activation space that control behaviour, then push the model along them.  | Steering vectors, CAA, RepE                                |
| Mechanistic interp (my favourite!)                   | Reverse-engineer the model's internal algorithm — what circuits exist and what they compute. | Circuit analysis, activation patching, sparse autoencoders |
| Developmental                                        | How the model changes during training. When and why capabilities emerge.                     | Grokking studies, learning dynamics                        |

The boundaries blur in practice. Most real papers combine techniques from a few schools. But the mindset is what's distinctive: behavioural interp asks "what does this model do?", mech interp asks "what algorithm did it learn and where does it live in the weights?"

---

## What is mechanistic interpretability?

Mech interp is the project of opening neural networks up and figuring out, in human-readable terms, what algorithm the weights implement. The dream is a sort of "decompiler" for neural nets.

A few examples of what mech interp has discovered:

- **Induction heads:** in transformers, a specific pair of attention heads in different layers implements an in-context "copy-and-continue" algorithm. This is most of how a language model learns from earlier in its prompt.
- **Indirect object identification (IOI):** in GPT-2 small, researchers traced the exact circuit (a handful of attention heads working together) that resolves which name to predict in sentences like "When John and Mary went to the store, John gave a drink to ___".
- **Sparse autoencoders (SAEs):** trained on real LLM activations, these recover thousands of human-interpretable "features" — directions in activation space corresponding to single concepts like "Golden Gate Bridge" or "Python comment".

Mech interp is the deepest of the interpretability schools — it has its own toolkit and its own mental model, neither of which you can pick up just by knowing regular ML. That's why a dedicated curriculum exists. This one.

Why mech-interp-only here: it's the school most aligned with Neel Nanda's work and Anthropic's interp output. The other schools are mostly applications of techniques you'd already know from regular ML; mech interp needs a dedicated path to build the necessary primitives. Once you have the foundations from this curriculum, the other schools are quick to pick up.

Things you'll bump into anyway, even staying mech-interp-only:

- **Linear probes** — baseline tool; often used to verify a mech-interp claim by checking what info is decodable from an activation.
- **Activation steering** — the action-arm of mech interp. Once you find a feature, steering tests whether it's causally real.
- **Behavioural evals** — how everyone outside research labs measures models. Worth knowing exists.

---

## The curriculum

Each step has its own directory with a teaching `README.md` and a runnable Jupyter notebook. Each step teaches one new big idea on top of what came before.

| #   | Project                                                     | Replicates                    | The ONE new concept it teaches                        |
| --- | ----------------------------------------------------------- | ----------------------------- | ----------------------------------------------------- |
| 0   | [MNIST template weights](00-mnist-templates/)               | Olah-era weight visualisation | Weights are interpretable — you can just look         |
| 1   | [Toy Models of Superposition](01-toy-models-superposition/) | Elhage et al. 2022            | Features ≠ neurons (superposition)                    |
| 2   | [Grokking modular addition](02-grokking-modular-addition/)  | Nanda et al. 2023             | Models learn algorithms, not just templates           |
| 3   | [Induction heads](03-induction-heads/)                      | Olsson et al. 2022            | Real transformers do work via attention-head circuits |
| 4   | [IOI circuit in GPT-2 small](04-ioi-circuit/)               | Wang et al. 2022              | Activation patching — verifying circuits causally     |
| 5   | [Sparse autoencoders](05-sparse-autoencoders/)              | Cunningham et al. 2023        | Dictionary learning — automatically finding features  |

### The narrative arc

- 0 → "weights are interpretable" (the naive view)
- 1 → "but neurons aren't, because of superposition" (complicates 0)
- 2 → "and models learn real algorithms, not just templates" (extends 1)
- 3 → "in transformers, those algorithms live in circuits of attention heads" (scales 2 to real data)
- 4 → "you can verify those circuits causally, even in models you didn't train" (gives you the toolkit)
- 5 → "and you can find features automatically — closing the loop with step 1" (back to where we started, but with a solution)

Each step assumes only what came before. No concept taught twice.

---

## How each project is structured

Every step folder contains:

- A `README.md` that teaches the concepts the project is about. Read this first — it stands alone as a tutorial.
- A notebook (`*.ipynb`) that runs the experiment end-to-end. Built to drop straight into Google Colab. Heavy comments, designed to be a tutorial in its own right.
- Optionally a `diagrams/` folder with explanatory images.

The READMEs share a common shape: what the project is about, glossary of terms, the experiment in plain English, just enough background to follow the code, a section-by-section walkthrough of the notebook, what you should see when you run it, how to run it, and where to go next.

---

## What you'll be able to do at the end

After step 5 you will have:

- Trained your first neural network and looked inside it (step 0).
- Watched a tiny model rediscover a 5-feature geometric solution to a compression problem (step 1).
- Trained a transformer from scratch and watched it suddenly generalise on modular arithmetic, then traced its hidden Fourier algorithm (step 2).
- Found and verified your first attention-head circuit (step 3).
- Reverse-engineered a real pre-trained model using activation patching (step 4).
- Trained a sparse autoencoder and recovered interpretable features automatically (step 5).

That is roughly the toolkit of an entry-level mech interp researcher.

---

## The open problems — what counts as "graduated"

After this curriculum you'll be able to read essentially any mech interp paper. To call yourself "in the conversation," you should be able to articulate the field's actual open problems in your own words. These are what people argue about at conferences:

1. **Are the features SAEs find the "right" features?** SAEs find some basis of interpretable directions, but multiple SAEs trained on the same data find slightly different sets. Is there a privileged, canonical decomposition? Nobody knows.
2. **Polysemanticity at scale.** Features in 70B-parameter models are messier than in toy models. Some of what looks like a single feature may itself be in superposition. Recursive superposition.
3. **Circuit fragility.** Identified circuits often only partially explain behaviour — there are usually backup mechanisms, redundancy, hard-to-attribute pathways. How complete can mech interp ever be on a real model?
4. **Evaluation.** How do you tell a true interpretation from a plausible-sounding-but-wrong one? "Faithfulness" of an interpretation is itself an active research area.
5. **Universality vs idiosyncrasy.** Do all transformers learn the same circuits given the same data, or does each random seed find a different algorithm? Evidence so far: mixed.
6. **Scaling to frontier models.** Most published interp is on GPT-2 small (124M) or similar. Doing it on Claude, GPT-4o, or Llama-405B is much harder.
7. **Does mech interp actually help with safety?** A clean understanding of why a model behaves a certain way should let you predict and control it. So far, the chain "we understand this circuit → therefore the model is safer" is mostly handwavy. The field's open existential question.
8. **Mathematical foundations.** Singular learning theory, computational mechanics, statistical-mechanics-of-deep-learning — several attempts to put mech interp on a theoretical footing. None have fully delivered.

You don't need to solve any of these. But if you can articulate them, you've graduated from learner to participant.

---

## Reading list

If at any point you'd rather just read than build, in roughly this order:

1. Neel Nanda — [Getting Started in Mech Interp](https://www.neelnanda.io/mechanistic-interpretability/getting-started)
2. Anthropic — [Toy Models of Superposition](https://transformer-circuits.pub/2022/toy_model/index.html) (replicated in step 1)
3. Anthropic — [A Mathematical Framework for Transformer Circuits](https://transformer-circuits.pub/2021/framework/index.html)
4. Nanda et al. — [Progress Measures for Grokking](https://arxiv.org/abs/2301.05217) (replicated in step 2)
5. Olsson et al. — [In-Context Learning and Induction Heads](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html) (step 3)
6. Wang et al. — [Interpretability in the Wild: IOI in GPT-2 Small](https://arxiv.org/abs/2211.00593) (step 4)
7. Cunningham et al. — [Sparse Autoencoders Find Highly Interpretable Features](https://arxiv.org/abs/2309.08600) (step 5)
8. Anthropic — [Scaling Monosemanticity](https://transformer-circuits.pub/2024/scaling-monosemanticity/)

---

## Communities

- [AlignmentForum / LessWrong](https://www.alignmentforum.org/) — most public mech interp discussion.
- [Neel Nanda's blog](https://www.neelnanda.io/) — exercises, paper rundowns, strong opinions.
- [Anthropic's Transformer Circuits Thread](https://transformer-circuits.pub/) — official Anthropic interp publications.
- [ARENA](https://www.arena.education/) — structured curriculum with exercises. Run in cohorts but materials are free.
- [MATS](https://www.matsprogram.org/) — competitive research fellowship; common pipeline from self-taught to professional researcher.

---

## How to use this curriculum

Just start at `00-mnist-templates/README.md` and work through each step in order. Each `README.md` tells you what to do, what to expect, and where to go next. Don't skip steps — every one teaches a concept the next one assumes.

Each project takes 1–3 evenings of focused work to really understand (not just run). A few months from start to finish is normal.
