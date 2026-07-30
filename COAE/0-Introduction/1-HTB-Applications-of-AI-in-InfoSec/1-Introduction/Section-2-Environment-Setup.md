---
tags:
  - type/note
  - theme/machine-learning
aliases: ["Section 2 - Environment Setup"]
lead: Setting up a Miniconda environment for AI/ML experimentation in an infosec context.
created: 2026-04-27
modified: 2026-04-28
source: "HackTheBox, HTB Applications of AI in InfoSec, COAE path."
---

# Section 2 - Environment Setup

A proper environment is required before building any models. This module offers two paths.

## The Playground

The first option is the Playground VM. For those without sufficient hardware to build the models locally, a Virtual Playground Environment is available. This VM is separate from PwnBox and can be spawned from specific sections of the module. Connect to it using your HTB VPN profile or PwnBox. The VM exposes `Jupyter` at `http://<VM-IP>:8888`. Instance time can be extended from the bottom of this section or any Model Evaluation section.

![[jupyter.png]]

**Note:** The Playground environment is sufficient to follow along, but it performs slower than a local setup. A local environment with adequate hardware produces shorter training times, enables experimentation with different parameters, and gives a more productive experience overall.

The second option is setting up an environment on your own system. This requires at least 4 GB of RAM. In most cases, local training will be faster than the playground VM.

---

## Miniconda

`Miniconda` is a minimal installer for the `Anaconda` Python distribution. It provides the `conda` package manager and a base Python environment without pre-installing the full suite of data science libraries that `Anaconda` includes. Packages are installed selectively, keeping the environment lean and purpose-built.

Both `Miniconda` and `Anaconda` use `conda` for package and dependency management. `Miniconda` is the lighter starting point; `Anaconda` ships with a broader set of data science tools pre-installed.

### Why Miniconda?

- `Performance`: `Miniconda` environments use optimized packages that perform well for data science and machine learning tasks.
- `Package Management`: `conda` resolves dependencies across complex library chains, which is especially important for deep learning projects.
- `Environment Isolation`: `conda` environments keep project dependencies separate, preventing conflicts and ensuring reproducibility.

## Installing Miniconda

### Windows

On Windows, `Scoop` (a command-line installer) streamlines the `Miniconda` installation. Open PowerShell and run:

```powershell
C:\> Set-ExecutionPolicy RemoteSigned -scope CurrentUser
C:\> irm get.scoop.sh | iex
```

Add the `extras` bucket, which contains `Miniconda`:

```powershell
C:\> scoop bucket add extras
```

Install `Miniconda`:

```powershell
C:\> scoop install miniconda3
```

Close and reopen PowerShell, then verify the installation:

```powershell
C:\> conda --version
conda 24.9.2
```

### MacOS

`Homebrew` simplifies software installation on macOS. Install it first if it is not already present:

```bash
gluppler@htb[/htb]$ /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

Install `Miniconda` via `Homebrew`:

```bash
gluppler@htb[/htb]$ brew install --cask miniconda
```

Close and reopen the terminal, then verify:

```bash
gluppler@htb[/htb]$ conda --version
conda 24.9.2
```

### Linux

Download the latest installer directly from the official repository, run it silently, and load the `conda` environment into your shell:

```bash
gluppler@htb[/htb]$ wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
gluppler@htb[/htb]$ chmod +x Miniconda3-latest-Linux-x86_64.sh
gluppler@htb[/htb]$ ./Miniconda3-latest-Linux-x86_64.sh -b -u
gluppler@htb[/htb]$ eval "$(/home/$USER/miniconda3/bin/conda shell.$(ps -p $$ -o comm=) hook)"
```

Confirm the installation:

```bash
gluppler@htb[/htb]$ conda --version
conda 24.9.2
```

## Init

The `conda init` command configures your shell to recognize `conda` commands and enables `conda activate` for switching environments. Run it after installing `Miniconda`:

```bash
gluppler@htb[/htb]$ conda init
```

This modifies your shell configuration file (e.g., `.bashrc` or `.zshrc`). Close and reopen the terminal for the changes to take effect.

Then run these commands to complete the setup:

```bash
gluppler@htb[/htb]$ conda config --add channels defaults
gluppler@htb[/htb]$ conda config --add channels conda-forge
gluppler@htb[/htb]$ conda config --add channels nvidia
gluppler@htb[/htb]$ conda config --add channels pytorch
gluppler@htb[/htb]$ conda config --set channel_priority strict
```

The `nvidia` channel is only needed on systems with an NVIDIA GPU.

## Deactivating Base

After installation, the `base` environment activates automatically on every new terminal, showing a `(base)` prefix in the prompt:

```bash
(base) $
```

To disable automatic activation:

```bash
gluppler@htb[/htb]$ conda config --set auto_activate_base false
```

This modifies the `condarc` configuration file. New terminals will no longer show the `(base)` prefix.

## Managing Virtual Environments

Virtual environments are isolated spaces where project-specific packages can be installed without affecting other projects or the global Python installation. They are essential for AI work because:

- `Dependency Isolation`: Each project maintains its own set of packages, even when versions conflict across projects.
- `Clean Project Structure`: All dependencies live inside the environment, keeping the project directory organized.
- `Reproducibility`: The same environment can be recreated on a different system with identical dependencies.
- `System Stability`: Conflicts with the global Python installation are avoided.

Create a new environment named `ai` with Python 3.11:

```bash
gluppler@htb[/htb]$ conda create -n ai python=3.11
```

### Activating the Environment

Activate the `ai` environment:

```bash
gluppler@htb[/htb]$ conda activate ai
```

The terminal prompt changes to show `(ai)`, confirming the environment is active. Packages installed via `conda` or `pip` now go into this environment.

Deactivate when done:

```bash
gluppler@htb[/htb]$ conda deactivate
```

## Essential Setup

With the `ai` environment active, install the core packages needed for this module. `conda` covers most requirements; `pip` handles the remainder.

```bash
gluppler@htb[/htb]$ conda install -y numpy scipy pandas scikit-learn matplotlib seaborn transformers datasets tokenizers accelerate evaluate optimum huggingface_hub nltk category_encoders
gluppler@htb[/htb]$ conda install -y pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia
gluppler@htb[/htb]$ pip install requests requests_toolbelt
```

## Updates

To update all `conda`-managed packages in the current environment:

```bash
gluppler@htb[/htb]$ conda update --all
```

This does not update packages installed via `pip`. Manage `pip` packages separately. Mixing `pip` and `conda` installations can introduce dependency conflicts, so track which packages came from each source.

---

## Summary

- Two setup paths exist: the Playground VM (accessible via HTB VPN or PwnBox) and a local Miniconda environment.
- Miniconda provides a minimal, lean Python environment managed by `conda`, distinct from the full Anaconda distribution.
- `conda` resolves complex dependency chains and provides environment isolation critical for reproducibility.
- The `ai` environment is created with Python 3.11 and populated with core ML packages via `conda` and `pip`.
- The `nvidia` channel is only needed on systems with an NVIDIA GPU.
- Mixing `pip` and `conda` packages can introduce conflicts; track which packages came from each source.

---

## Best Practices

- Use `conda create -n ai python=3.11` to create an isolated project environment rather than polluting the base environment.
- Run `conda config --set auto_activate_base false` to prevent the base environment from auto-activating on every terminal.
- Set channel priority to `strict` (`conda config --set channel_priority strict`) to avoid dependency resolution surprises.
- Always activate the target environment before installing packages to ensure they land in the correct environment.
- Track which packages were installed via `conda` vs. `pip` to avoid dependency conflicts when updating.
- Verify installation with `conda --version` after each setup step before proceeding.

---

## Quiz

**Q1:** What is the difference between Miniconda and Anaconda?
> Miniconda is a minimal installer that includes only `conda` and a base Python environment; Anaconda ships with a broad set of data science tools pre-installed.

**Q2:** Why should `conda` environments be used for AI projects rather than the global Python installation?
> Environments isolate project dependencies, prevent version conflicts across projects, and ensure reproducibility on different systems.

**Q3:** What command creates a new conda environment named `ai` with Python 3.11?
> `conda create -n ai python=3.11`

---

# Back Matter

**Source**
- based_on:: [[HTB-COAE-Prep/0-Introduction/0-HTB-Fundamentals-of-AI/0-Module-Information/Module-Description]]

**References**
- see:: [[HTB-COAE-Prep/0-Introduction/1-HTB-Applications-of-AI-in-InfoSec/1-Introduction/Section-1-Introduction]] — outlines the module context this environment supports
- see:: [[Section-3-JupyterLab]] — the IDE launched inside the environment configured here
- see:: [[Section-4-Python-Libraries-for-AI]] — the libraries installed into the conda environment

**Terms**
- Miniconda, Anaconda, conda, virtual environment, environment isolation, package manager, conda channels, pip, Scoop, Homebrew, dependency management
