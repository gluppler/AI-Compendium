---
tags:
  - type/note
  - theme/generative-models
  - theme/deep-learning
aliases: ["Section 23 - Diffusion Models"]
lead: Diffusion models generate data by learning to reverse a gradual noise-addition process, producing high-quality images and audio.
created: 2026-04-27
modified: 2026-04-28
source: "HackTheBox, Fundamentals of AI, Section 23."
---

`Diffusion models` are a class of generative models that define generation as the reversal of a known noise-injection process. Rather than learning a GAN discriminator or encoding data into a VAE latent, a diffusion model trains a denoising network to undo Gaussian noise added in small, fixed increments. At sampling time, starting from pure noise and repeatedly applying the denoiser yields a clean sample from the learned data distribution.

## How Diffusion Models Work
![[a_cat_in_a_hat.png]]

For conditional generation from a text prompt such as "a cat in a hat," the forward/reverse diffusion operates identically to the unconditional case, but the denoising network receives an additional conditioning signal derived from the text:

1. `Text Encoding:` A pre-trained text encoder (e.g., a transformer or `CLIP`) maps the prompt to a high-dimensional embedding vector that captures its semantic content.
2. `Conditioning the Denoising Process:` The text embedding is injected into the denoising network — typically via cross-attention — at each reverse step, biasing the predicted noise toward removing non-text-consistent structure.
3. `Sampling Process:` Sampling begins from pure Gaussian noise $x_T \sim \mathcal{N}(0, I)$. At each step, the network predicts the noise component given both $x_t$ and the text embedding, and $x_{t-1}$ is computed from that prediction.
4. `Final Image Generation:` After $T$ denoising steps, the resulting $x_0$ is the synthesized image. The iterative process allows gradual refinement from global structure toward fine detail.

### Forward Process: Adding Noise
![[diffusion_first_five.png]]

The forward process defines a fixed Markov chain that corrupts data $x_0$ into Gaussian noise over $T$ steps:

$$x_T = q(x_T \mid x_0)$$

where $x_0$ is the original data and $x_T$ is pure noise. Each transition adds a small amount of Gaussian noise:

$$x_t = q(x_t \mid x_{t-1})$$

with $t$ ranging from 0 to $T$. The step-by-step corruption is designed so that $q(x_T \mid x_0)$ is approximately $\mathcal{N}(0, I)$ for sufficiently large $T$, regardless of the original data distribution.

### Reverse Process: Removing Noise
![[diffusion_final_five.png]]

The reverse process learns to invert the forward chain. A neural network parameterized by $\theta$ approximates:

$$x_{t-1} = p_\theta(x_{t-1} \mid x_t)$$

The network is trained to predict the noise $\varepsilon$ that was added at each forward step. The training loss is mean squared error between true and predicted noise:

$$\mathcal{L} = \mathbb{E}\!\left[\|\varepsilon - \varepsilon_{\text{pred}}\|^2\right]$$

Minimizing this loss over all time steps and data points trains the network to be a generic denoiser across the full noise range.

### Noise Schedule
![[diffusion_mid_five.png]]

The noise schedule $\{\beta_t\}$ specifies the variance of noise added at each forward step. A common linear schedule:

$$\beta_t = \beta_{\min} + \frac{t}{T}(\beta_{\max} - \beta_{\min})$$

where $\beta_t$ is the noise variance at step $t$, and $\beta_{\min}$, $\beta_{\max}$ are the schedule endpoints. The schedule determines how quickly the data is destroyed in the forward direction and how much denoising work each reverse step must do. Cosine and learned schedules often outperform the linear baseline.

### Denoising Network

The denoising network predicts the noise $\hat{\varepsilon}$ at each time step given the noisy input $x_t$. It is typically a deep convolutional U-Net for image data, or a transformer for higher-level representations. The time step $t$ is embedded (e.g., via sinusoidal encoding) and injected into the network so the same weights can handle all noise levels. For text-conditioned models, cross-attention layers inside the network attend to the text embedding at each resolution.

### Training

Training minimizes the denoising loss over randomly sampled data points and time steps:

1. `Initialize the Model:` Start with an initial set of parameters $\theta$ for the denoising network.
2. `Forward Process:` Add noise to the original data using the noise schedule.
3. `Reverse Process:` Train the denoising network to predict the noise at each time step.
4. `Loss Calculation:` Compute the loss between predicted and actual noise.
5. `Parameter Update:` Update model parameters using gradient descent to minimize the loss.
6. `Iterate:` Repeat the process for multiple epochs until the model converges.

### Sampling
![[diffusion_full.png]]

Generation inverts the forward process, starting from $x_T \sim \mathcal{N}(0, I)$:

$$x_0 = p_\theta(x_0 \mid x_T)$$

The sampling loop:

1. `Start with Noise:` Initialize with pure Gaussian noise $x_T$.
2. `Iterative Denoising:` For each time step $t$ from $T$ down to 1, use the denoising network to predict the noise and compute $x_{t-1}$.
3. `Final Sample:` After $T$ steps, $x_0$ is the generated output.

## Data Assumptions

Diffusion models make the following assumptions about the data:

- `Markov Property:` Both forward and reverse processes are Markov chains — each step depends only on the immediately preceding state, not the full history.
- `Static Data Distribution:` The model learns from a fixed training dataset and assumes the distribution it represents does not shift. Distribution shift at inference time degrades sample quality.
- `Smoothness Assumption:` Diffusion models perform best when small perturbations in latent or data space produce small changes in output. Smooth distributions allow the iterative denoising steps to make stable, incremental progress toward a clean sample.

---

## Summary

- Diffusion models generate data by learning to reverse a fixed Markov forward process that gradually corrupts data with Gaussian noise over T steps.
- The forward process is fixed (not learned): each step adds a small amount of Gaussian noise until the data becomes pure noise `x_T ~ N(0, I)`.
- The reverse process is learned: a neural network `p_θ` predicts the noise at each step, trained by minimizing MSE between predicted and true noise.
- At sampling time the model starts from pure Gaussian noise and iteratively denoises over T steps to produce a clean sample.
- The noise schedule `{β_t}` controls how aggressively noise is added at each forward step; cosine and learned schedules often outperform the linear baseline.
- For conditional generation (e.g., text-to-image), a text encoder provides an embedding injected into the denoising network via cross-attention at each reverse step.

---

## Best Practices

- Use cosine or learned noise schedules rather than linear — they produce more uniform training difficulty across time steps and better sample quality.
- Evaluate diffusion models with FID against the training distribution rather than visual inspection alone — small perceptual artifacts may not show up in individual samples.
- For text-conditioned generation, use classifier-free guidance — train both a conditional and unconditional denoiser, then interpolate at sampling time to control the trade-off between sample quality and diversity.
- Keep the number of sampling steps T large enough for clean samples, but use DDIM or other fast samplers to reduce inference-time steps from 1000 to 50–100 without major quality loss.
- Ensure training data and inference distribution are aligned — diffusion models are sensitive to distribution shift, which degrades sample quality.

---

## Quiz

**Q1:** What is the forward diffusion process and what is its endpoint?
> The forward process is a fixed Markov chain that adds small amounts of Gaussian noise at each of T steps, gradually corrupting the original data `x_0`. After sufficiently many steps, the result `x_T` is approximately pure Gaussian noise `N(0, I)`, independent of the original data distribution.

**Q2:** What does the denoising network learn and what is its training loss?
> The denoising network learns to predict the noise `ε` that was added at each forward step given the noisy input `x_t` and the time step `t`. Training minimizes the mean squared error between the predicted noise and the actual noise added: `L = E[‖ε - ε_pred‖²]`.

**Q3:** How does text conditioning work in a text-to-image diffusion model?
> A pre-trained text encoder maps the text prompt to an embedding vector. This embedding is injected into the denoising U-Net via cross-attention layers at each reverse step, biasing the denoiser to remove noise in a direction consistent with the text prompt.

**Q4:** What is the noise schedule and why does it matter?
> The noise schedule `{β_t}` specifies the variance of noise added at each forward step. It controls how quickly data is destroyed in the forward direction and how much work each reverse step must do. A well-designed schedule (cosine or learned) distributes denoising difficulty uniformly across time steps, improving training stability and sample quality.

---
# Back Matter

**Source**
- based_on:: [[HTB-COAE-Prep/0-Introduction/0-HTB-Fundamentals-of-AI/0-Module-Information/Module-Description]]

**References**
- see:: [[Section-22-Large-Language-Models]] — LLMs and diffusion models are the two dominant generative model families
- see:: [[Section-19-Convolutional-Neural-Networks]] — CNNs are used within many diffusion model architectures

**Terms**
- forward diffusion, reverse diffusion, DDPM, noise schedule, denoising, score matching, U-Net
