---
tags:
  - type/note
  - theme/machine-learning
aliases: ["Section 4 - Python Libraries for AI"]
lead: Core Python libraries for AI/ML — Scikit-learn and PyTorch for model training and evaluation.
created: 2026-04-27
modified: 2026-04-28
source: "HackTheBox, HTB Applications of AI in InfoSec, COAE path."
---

# Section 4 - Python Libraries for AI

Python's library ecosystem makes it the dominant language for AI development. This section covers two central libraries: `Scikit-learn` for classical machine learning and `PyTorch` for deep learning. The goal is familiarity with their purpose, structure, and common APIs; the official documentation is the authoritative reference for full details. The code snippets here are illustrative; they do not need to be executed.

## Scikit-learn

`Scikit-learn` is a comprehensive machine learning library built on `NumPy`, `SciPy`, and `Matplotlib`. It provides a wide range of algorithms and a consistent API across all of them.

- `Supervised Learning`: Linear Regression, Logistic Regression, Support Vector Machines, Decision Trees, Naive Bayes, Random Forests, Gradient Boosting.
- `Unsupervised Learning`: K-Means and DBSCAN clustering, PCA and t-SNE dimensionality reduction.
- `Model Selection and Evaluation`: Hyperparameter tuning, cross-validation, and performance metrics.
- `Data Preprocessing`: Feature scaling, missing value handling, and categorical encoding.

### Data Preprocessing

Feature scaling ensures all features contribute proportionally to the learning process. `Scikit-learn` provides several scalers:

- `StandardScaler`: Removes the mean and scales to unit variance.
- `MinMaxScaler`: Scales features to a fixed range, typically 0 to 1.
- `RobustScaler`: Scales using statistics that are robust to outliers.

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

Categorical features must be converted to numeric form before most algorithms can use them:

- `OneHotEncoder`: Creates a binary column for each category.
- `LabelEncoder`: Assigns a unique integer to each category.

```python
from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder()
X_encoded = encoder.fit_transform(X)
```

Missing values are common in real-world data. `Scikit-learn` provides imputation strategies:

- `SimpleImputer`: Replaces missing values using a fixed strategy (mean, median, most frequent).
- `KNNImputer`: Imputes using the k-nearest neighbors algorithm.

```python
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)
```

### Model Selection and Evaluation

Splitting data into training and test sets measures how well a model generalizes to unseen data:

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
```

Cross-validation provides a more robust estimate by training and testing on multiple data folds:

```python
from sklearn.model_selection import cross_val_score

scores = cross_val_score(model, X, y, cv=5)
```

Common evaluation metrics:

- `accuracy_score`: For classification tasks.
- `mean_squared_error`: For regression tasks.
- `precision_score`, `recall_score`, `f1_score`: For classification with imbalanced classes.

```python
from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, y_pred)
```

### Model Training and Prediction

All `Scikit-learn` models follow the same API pattern. Instantiate the model with desired hyperparameters:

```python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(C=1.0)
```

Train on the training data using `fit()`:

```python
model.fit(X_train, y_train)
```

Generate predictions on new data using `predict()`:

```python
y_pred = model.predict(X_test)
```

## PyTorch

`PyTorch` is an open-source machine learning library developed by Facebook's AI Research lab. It provides a flexible framework for building and deploying deep learning models.

### Key Features

- `Deep Learning`: Supports building complex neural networks with arbitrary architectures.
- `Dynamic Computational Graphs`: The computation graph is constructed on the fly during the forward pass, enabling flexible and debuggable model design.
- `GPU Support`: Models and tensors can be moved to GPU for accelerated computation.
- `TorchVision Integration`: `TorchVision` provides datasets, pre-trained models, and image transformation utilities.
- `Automatic Differentiation`: `autograd` computes gradients automatically, simplifying backpropagation.

### Dynamic Computational Graphs and Tensors

`PyTorch` builds its computational graph dynamically during the forward pass, which makes non-linear and experimental architectures straightforward to implement.

`Tensors` are multi-dimensional arrays and the core data structure in `PyTorch`. They behave like `NumPy` arrays but can run on GPU:

```python
import torch

# Creating a tensor
x = torch.tensor([1.0, 2.0, 3.0])

# Move to GPU if available
if torch.cuda.is_available():
    x = x.to('cuda')
```

### Building Models with PyTorch

`PyTorch`'s `torch.nn` module provides layers and building blocks for neural networks.

The `Sequential` API stacks layers in order:

```python
import torch.nn as nn

model = nn.Sequential(
    nn.Linear(784, 128),
    nn.ReLU(),
    nn.Linear(128, 10),
    nn.Softmax(dim=1)
)
```

Subclassing `nn.Module` supports more complex architectures with shared layers, multiple inputs, or non-linear topologies:

```python
import torch.nn as nn

class CustomModel(nn.Module):
    def __init__(self):
        super(CustomModel, self).__init__()
        self.layer1 = nn.Linear(784, 128)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(128, 10)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        x = self.softmax(x)
        return x

model = CustomModel()
```

### Training and Evaluation

Optimizers adjust model parameters to minimize the loss function. Common options include `Adam`, `SGD`, and `RMSprop`:

```python
import torch.optim as optim

optimizer = optim.Adam(model.parameters(), lr=0.001)
```

Loss functions measure the gap between predictions and targets:

- `CrossEntropyLoss`: For multi-class classification.
- `BCEWithLogitsLoss`: For binary classification.
- `MSELoss`: For regression.

```python
import torch.nn as nn

loss_fn = nn.CrossEntropyLoss()
```

Custom metrics can be defined as plain functions:

```python
def accuracy(output, target):
    _, predicted = torch.max(output, 1)
    correct = (predicted == target).sum().item()
    return correct / len(target)
```

The training loop runs forward passes, computes loss, and updates parameters via backpropagation:

```python
import torch

epochs = 10
num_batches = 100

for epoch in range(epochs):
    for batch in range(num_batches):
        # Get batch of data
        x_batch, y_batch = get_batch(batch)

        # Forward pass
        y_pred = model(x_batch)

        # Calculate loss
        loss = loss_fn(y_pred, y_batch)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Print progress every 10 batches
        if batch % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Batch [{batch+1}/{num_batches}], Loss: {loss.item():.4f}')
```

### Data Loading and Preprocessing

`Dataset` and `DataLoader` handle data loading and batching:

```python
from torch.utils.data import Dataset, DataLoader

class CustomDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# Example usage
dataset = CustomDataset(data, labels)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
```

### Model Saving and Loading

Save and reload model weights for inference or continued training:

```python
# Save model
torch.save(model.state_dict(), 'model.pth')

# Load model
model = CustomModel()
model.load_state_dict(torch.load('model.pth'))
model.eval()  # Set to evaluation mode
```

---

## Summary

- Scikit-learn provides a consistent API across supervised learning, unsupervised learning, preprocessing, and model evaluation.
- All Scikit-learn models follow fit/predict/transform patterns, making it easy to swap algorithms with minimal code changes.
- PyTorch uses dynamic computational graphs built during the forward pass, enabling flexible and debuggable architectures.
- Tensors are PyTorch's core data structure — they behave like NumPy arrays but can be moved to GPU for acceleration.
- `torch.nn.Module` subclassing supports complex architectures; the `Sequential` API is sufficient for simple stacked layers.
- `Dataset` and `DataLoader` abstract data batching and parallel loading, decoupling data pipeline from model code.

---

## Best Practices

- Always call `scaler.fit_transform(X_train)` on training data only, then use `scaler.transform(X_test)` to prevent data leakage.
- Use `train_test_split` with a fixed `random_state` to ensure reproducible splits across runs.
- Call `optimizer.zero_grad()` before each backward pass to prevent gradient accumulation from previous batches.
- Set the model to `model.eval()` during inference and wrap it in `torch.no_grad()` to skip gradient computation and reduce memory use.
- Save model weights with `torch.save(model.state_dict(), path)` rather than pickling the entire model for portability.
- Use `cross_val_score` rather than a single train/test split for small datasets to get a more reliable performance estimate.

---

## Quiz

**Q1:** What is the difference between `StandardScaler` and `MinMaxScaler` in Scikit-learn?
> `StandardScaler` removes the mean and scales to unit variance; `MinMaxScaler` scales features to a fixed range (typically 0 to 1).

**Q2:** What is the purpose of calling `optimizer.zero_grad()` in the PyTorch training loop?
> It clears the gradients accumulated from the previous batch so they do not add up incorrectly during the next backward pass.

**Q3:** What is the role of `DataLoader` in PyTorch?
> It wraps a `Dataset` to provide automatic batching, shuffling, and parallel data loading during training and inference.

---

# Back Matter

**Source**
- based_on:: [[HTB-COAE-Prep/0-Introduction/0-HTB-Fundamentals-of-AI/0-Module-Information/Module-Description]]

**References**
- see:: [[Section-6-Data-Preprocessing]] — applies Scikit-learn preprocessing tools introduced here
- see:: [[Section-7-Data-Transformation]] — uses OneHotEncoder and train_test_split covered in this section
- see:: [[AI-ML-Neural-Network-Foundations]] — theoretical grounding for the neural network concepts behind PyTorch

**Terms**
- Scikit-learn, PyTorch, tensors, dynamic computational graph, autograd, loss function, optimizer, StandardScaler, OneHotEncoder, DataLoader, cross-validation, accuracy_score, CrossEntropyLoss
