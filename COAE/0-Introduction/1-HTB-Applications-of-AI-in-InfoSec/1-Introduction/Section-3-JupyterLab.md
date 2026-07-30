---
tags:
  - type/note
  - theme/machine-learning
aliases: ["Section 3 - JupyterLab"]
lead: Using JupyterLab as an interactive development platform for AI model prototyping.
created: 2026-04-27
modified: 2026-04-28
source: "HackTheBox, HTB Applications of AI in InfoSec, COAE path."
---

# Section 3 - JupyterLab

`JupyterLab` is a web-based interactive development environment for code, data, and visualization. Its cell-based execution model makes it a standard tool for data scientists and machine learning practitioners.

## Why JupyterLab?

- `Interactive Environment`: Code runs in individual cells, supporting iterative development and quick experimentation.
- `Data Exploration and Visualization`: Integrates directly with libraries like `matplotlib` and `seaborn` for in-notebook visualizations.
- `Documentation and Sharing`: Supports markdown and LaTeX, allowing rich documentation alongside code.

Install `JupyterLab` using `conda` from within the `ai` environment:

```bash
gluppler@htb[/htb]$ conda install -y jupyter jupyterlab notebook ipykernel
```

Launch `JupyterLab`:

```bash
gluppler@htb[/htb]$ jupyter lab
```

This opens the `JupyterLab` interface in a new browser tab.

## Using JupyterLab

![[jupyter_dark.png]]

A notebook is `JupyterLab`'s primary document type, combining code, text, and visualizations in a single file. Notebooks consist of cells:

- `Code cells`: Execute code in Python, R, Julia, or other supported kernels.
- `Markdown cells`: Render formatted text, equations, and images using markdown syntax.
- `Raw cells`: Unrendered plain text.

To create a new notebook, click the "Python 3" icon under "Notebook" in the Launcher. A notebook opens with a single empty code cell.

![[jupyter_new.png]]

Enter Python code into a code cell and press `Shift + Enter` to execute it:

```python
print("Hello, JupyterLab!")
```

Output appears directly below the cell.

![[jupyter_hello_output.png]]

`Jupyter` notebooks use a stateful environment where variables, functions, and imports defined in one cell persist for all later cells as long as the kernel is running. This differs from a stateless model where each execution is isolated.

The stateful nature requires care with execution order. Running cells out of order can produce unexpected results because earlier variable assignments may still be in effect.

For example, a cell that sets:

```python
x = 1
```

makes `x` available in a later cell:

```python
print(x)  # prints 1 because x was defined previously
```

If the first cell is updated to `x = 2` and re-run before the `print(x)` cell, the output changes accordingly.

Add new cells with the "+" button in the toolbar. Select the cell type (code or markdown) from the dropdown in the toolbar.

`JupyterLab` integrates with `pandas`, `matplotlib`, and `seaborn` for data exploration. The following example creates a sample DataFrame and plots it:

![[jupyter_simple_plot.png]]

```python
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Create a sample DataFrame
data = pd.DataFrame({
    "column1": np.random.rand(50),
    "column2": np.random.rand(50) * 10
})

# Display the first few rows
print(data.head())

# Create a scatter plot
plt.scatter(data["column1"], data["column2"])
plt.xlabel("Column 1")
plt.ylabel("Column 2")
plt.title("Scatter Plot")
plt.show()
```

Save the notebook with the save icon in the toolbar or `Ctrl + S`. Rename the notebook by right-clicking the tab or its entry in the file browser.

## Restarting the Kernel

The kernel is the separate process that executes code and maintains the notebook's state. Restarting the kernel clears all variables, functions, and imported modules from memory, providing a clean slate without closing `JupyterLab`.

To restart the kernel:

1. Open the `Kernel` menu in the top toolbar.
2. Select `Restart Kernel` to reset the environment while preserving cell outputs, or `Restart Kernel and Clear All Outputs` to also remove all previous output.

After restarting, re-run cells containing imports, variable definitions, and computations to restore the environment state.

For a full reference, see the [JupyterLab Documentation](https://jupyterlab.readthedocs.io/en/latest/getting_started/overview.html).

---

## Summary

- JupyterLab is a web-based, cell-based IDE that supports interactive development, data exploration, and rich documentation.
- Notebooks consist of code cells, markdown cells, and raw cells; Shift+Enter executes a code cell.
- The kernel maintains a stateful environment — variables and imports persist across cells as long as the kernel runs.
- Running cells out of order can produce unexpected results because earlier variable assignments remain in effect.
- Restarting the kernel clears all in-memory state without closing JupyterLab; re-run all setup cells afterward.
- JupyterLab integrates natively with pandas, matplotlib, and seaborn for in-notebook visualization.

---

## Best Practices

- Restart the kernel and run all cells from top to bottom before submitting or sharing a notebook to ensure reproducibility.
- Name and save notebooks meaningfully; use the file browser to keep datasets and notebooks organized together.
- Use markdown cells to document intent and results alongside code — this makes notebooks usable as lab notes.
- Keep cell order logical: imports first, then data loading, then processing, then modeling and evaluation.
- Use `Restart Kernel and Clear All Outputs` when sharing notebooks to remove stale outputs that might mislead readers.

---

## Quiz

**Q1:** What does it mean that JupyterLab uses a "stateful environment"?
> Variables, functions, and imports defined in one cell persist and are available in all later cells as long as the kernel is running.

**Q2:** What are the three cell types in a JupyterLab notebook?
> Code cells (execute code), markdown cells (render formatted text and equations), and raw cells (unrendered plain text).

**Q3:** What happens when you restart the kernel in JupyterLab?
> All variables, functions, and imported modules are cleared from memory; the notebook's outputs are preserved unless you also select "Clear All Outputs."

---

# Back Matter

**Source**
- based_on:: [[HTB-COAE-Prep/0-Introduction/0-HTB-Fundamentals-of-AI/0-Module-Information/Module-Description]]

**References**
- see:: [[Section-2-Environment-Setup]] — environment where JupyterLab is installed and launched
- see:: [[Section-4-Python-Libraries-for-AI]] — libraries used within notebooks for model development
- see:: [[Section-5-Datasets]] — first section that uses notebook-based code to load and inspect data

**Terms**
- JupyterLab, notebook, kernel, code cell, markdown cell, stateful environment, stateless model, pandas, matplotlib, seaborn, ipykernel, Shift+Enter
