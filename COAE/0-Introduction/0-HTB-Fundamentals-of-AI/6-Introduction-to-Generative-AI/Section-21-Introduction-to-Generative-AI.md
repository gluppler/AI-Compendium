---
tags:
  - type/note
  - theme/generative-models
  - theme/deep-learning
aliases: ["Section 21 - Introduction to Generative AI"]
lead: Generative AI models learn to produce original content — text, images, audio — rather than merely classify or predict.
created: 2026-04-27
modified: 2026-04-28
source: "HackTheBox, Fundamentals of AI, Section 21."
---

`Generative AI` covers models that learn the underlying distribution of a training dataset well enough to draw new samples from it. Discriminative models draw decision boundaries between classes; generative models learn to synthesize examples that could have plausibly come from the training distribution. The outputs can be text, images, audio, video, or code.

## How Generative AI Works

A generative model is trained to capture the statistical structure of its training data. The three-phase cycle:

1. `Training:` The model processes a large corpus and adjusts its parameters to minimize a loss that measures how well it reconstructs or predicts the data. It encodes the dependencies between elements — pixels, tokens, audio frames — into its weights.
2. `Generation:` At inference time, the model samples from its learned distribution. This may involve ancestral sampling, iterative denoising, or decoding from a latent code, depending on the architecture.
3. `Evaluation:` Generated outputs are assessed for quality, diversity, and distributional fidelity. Evaluation can be perceptual (human judges) or metric-based.

## Types of Generative AI Models

- `Generative Adversarial Networks (GANs):` A generator network and a discriminator network train simultaneously in a minimax game. The generator attempts to produce samples the discriminator cannot distinguish from real data; the discriminator attempts to catch fakes. Competitive pressure drives both toward convergence at a high-fidelity generator.
- `Variational Autoencoders (VAEs):` Encode input data into a continuous latent distribution rather than a point, then decode samples from that distribution back into data space. The regularized latent space allows interpolation and controlled generation.
- `Autoregressive Models:` Factor the joint distribution as a product of conditionals and generate one token or pixel at a time: $p(x) = \prod_t p(x_t \mid x_{<t})$. GPT-family language models are the dominant example.
- `Diffusion Models:` Define a forward process that gradually corrupts data with Gaussian noise, then train a neural network to reverse that process step by step. At sampling time, the model starts from pure noise and iteratively denoises to produce a clean sample.

## Important Generative AI Concepts

### Latent Space

The `latent space` is a lower-dimensional representation that captures the essential structure of the data. In a VAE, similar inputs map to nearby points in latent space; sampling nearby points and decoding them produces similar but distinct outputs. The geometry of the latent space directly controls generation diversity and controllability.

### Sampling

`Sampling` draws a point from the learned distribution and maps it to an output. Quality and diversity trade off against each other: a model that concentrates probability mass on high-quality outputs may sacrifice variety, while one that spreads mass broadly may produce incoherent samples.

### Mode Collapse

`Mode collapse` occurs when a generative model produces only a narrow subset of the possible outputs despite the training distribution being diverse. It is a well-known failure mode of GAN training, where the generator discovers a small set of outputs that fool the discriminator and exploits them rather than exploring the full data distribution.

### Overfitting

A generative model that overfits memorizes training examples rather than learning a generalizable distribution. It will regenerate training data near-exactly but fail to produce novel samples. In generative contexts, overfitting limits creativity and risks leaking training data.

### Evaluation Metrics

Quantifying generative quality requires specialized metrics:

- `Inception Score (IS):` Measures both the quality (sharpness of class predictions) and diversity (entropy of marginal class distribution) of generated images using a pre-trained classifier.
- `Fréchet Inception Distance (FID):` Compares the feature-space distribution of generated images to real images using the Fréchet distance between two Gaussians fit to each set. Lower FID indicates greater distributional similarity.
- `BLEU score (for text generation):` Measures n-gram overlap between generated and reference text. A proxy for fluency and lexical accuracy, though insensitive to semantic quality.

---

## Summary

- Generative AI models learn the statistical structure of training data well enough to sample new, plausible examples from the learned distribution.
- The three-phase cycle: training (minimize reconstruction/prediction loss) → generation (sample from learned distribution) → evaluation (quality and diversity metrics).
- Four main architectures: GANs (adversarial generator-discriminator game), VAEs (continuous latent distribution), autoregressive models (factored conditionals), and diffusion models (iterative denoising from noise).
- The latent space is a lower-dimensional representation — its geometry directly controls generation diversity, controllability, and interpolation quality.
- Mode collapse (GAN failure producing narrow output diversity) and overfitting (memorizing training data) are the two primary failure modes.
- Evaluation requires specialized metrics: FID (distributional similarity of images), IS (quality and diversity), BLEU (n-gram overlap for text).

---

## Best Practices

- Choose the architecture based on the output modality and quality requirements: GANs for fast high-resolution image synthesis, diffusion models for best image quality, autoregressive models (GPT family) for text generation.
- Monitor for mode collapse in GAN training — if generated samples lack diversity, inspect the discriminator loss and consider architectural changes or training stabilization techniques.
- Track FID during image model training as a more reliable metric than visual inspection alone — FID captures distributional similarity, not just individual sample sharpness.
- Avoid evaluating generative models on training data — use held-out reference sets to detect overfitting (memorization of training examples).
- When controllable generation is needed (e.g., generating images from text), use conditioning mechanisms (cross-attention, classifier guidance) rather than unconditional models with post-hoc filtering.

---

## Quiz

**Q1:** What is the key difference between a discriminative model and a generative model?
> Discriminative models learn decision boundaries between classes given data. Generative models learn the full joint distribution of the data, allowing them to synthesize new examples that plausibly came from the training distribution.

**Q2:** What is mode collapse in GANs and what causes it?
> Mode collapse occurs when the generator finds a small subset of outputs that consistently fool the discriminator and produces only those, ignoring the full diversity of the training distribution. It is a training instability caused by the adversarial dynamics.

**Q3:** What are autoregressive models and how do they generate outputs?
> Autoregressive models factor the joint distribution as a product of conditionals: `p(x) = Π p(x_t | x_{<t})`. They generate one token or pixel at a time, each conditioned on all previously generated elements. GPT-family LLMs are the dominant example.

**Q4:** What does FID measure and what does a lower score indicate?
> Fréchet Inception Distance (FID) compares the feature-space distributions of generated and real images by computing the Fréchet distance between two Gaussians fit to each set. A lower FID indicates the generated distribution is closer to the real data distribution.

---
# Back Matter

**Source**
- based_on:: [[HTB-COAE-Prep/0-Introduction/0-HTB-Fundamentals-of-AI/0-Module-Information/Module-Description]]

**References**
- see:: [[Section-22-Large-Language-Models]] — LLMs are the dominant text generative architecture
- see:: [[Section-23-Diffusion-Models]] — diffusion models lead for image/audio generation

**Terms**
- generative model, GAN, VAE, discriminator, generator, latent space, generative adversarial network
