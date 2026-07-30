---
tags:
  - type/note
  - theme/deep-learning
aliases: ["Section 19 - Convolutional Neural Networks"]
lead: CNNs use convolutional and pooling layers to learn spatial feature hierarchies from grid-like data such as images.
created: 2026-04-27
modified: 2026-04-28
source: "HackTheBox, Fundamentals of AI, Section 19."
---

`Convolutional Neural Networks` (`CNNs`) are architectures optimized for grid-structured data. A convolutional layer slides a small learnable filter across the spatial dimensions of the input, computing a dot product at each position to produce a `feature map`. Because the same filter weights are applied everywhere (weight sharing), the network detects a feature regardless of where it appears in the input — a property called `translation equivariance`. Multiple filters per layer detect different feature types simultaneously.

A standard CNN pipeline stacks three layer types:

- `Convolutional Layers:` Apply learned filters to extract local features — edges, corners, textures. The output is a set of feature maps, one per filter.
- `Pooling Layers:` Downsample feature maps spatially, reducing computation and providing a degree of translation invariance. `Max pooling` retains the strongest activation in each window; `average pooling` retains the mean.
- `Fully Connected Layers:` After the convolutional stack flattens the feature maps into a vector, dense layers perform the final high-level reasoning and produce class scores or regression outputs.

## Feature Maps and Hierarchical Feature Learning
![[04 - Convolutional Neural Networks_0.png]]

Feature maps produced by different filters capture different visual attributes. The network refines filter weights during training, turning them into increasingly specific detectors. The hierarchy that emerges is consistent across architectures:

- `Initial Layers:` Learn low-level detectors — vertical edges, horizontal edges, color blobs.
- `Intermediate Layers:` Combine those primitives into corners, curves, and simple textures.
- `Deeper Layers:` Assemble part-level structures — wheels, faces, characters — from the intermediate representations.

The handwritten digit "7" illustrates this progression well.

![[cnn_7.png]]

The first convolutional layer responds to the sharp intensity transitions at the digit's borders — its output highlights the outer edges of the stroke.

![[cnn_layer_1.png]]

The second convolutional layer attends to the interior structure: the continuous lines and diagonal stroke that define the numeral's geometry rather than its boundary.

![[cnn_layer_2.png]]

Each successive layer compresses the spatial resolution while expanding the semantic content, allowing deep CNNs to map raw pixel values to abstract object identities.

## Image Recognition
![[04 - Convolutional Neural Networks_1.png]]

A CNN classifying animals processes an image through the following stages:

1. `Input Layer:` Receives a 3D tensor of shape (height, width, channels).
2. `Convolutional Layers:`
    - `Layer 1:` Detects low-level features like edges and simple textures.
    - `Layer 2:` Combines these features to detect more complex patterns, such as corners and curves.
    - `Layer 3:` Recognizes higher-level structures like shapes and object parts.
3. `Pooling Layers:`
    - Reduce the spatial dimensions of the feature maps, making the network less computationally expensive and more robust to small translations in the input image.
4. `Fully Connected Layers:`
    - Flatten the output from the final pooling layer.
    - Perform high-level reasoning and make predictions based on the extracted features, such as classifying the image as a cat, dog, or bird.

## Data Assumptions for a CNN

### Grid-Like Data Structure

CNNs assume the input is a spatial grid. The canonical cases:

- `Images:` 2D grids of pixel values with height, width, and channel dimensions.
- `Videos:` 3D grids extending images with a time axis.

The grid structure makes localized convolution operations both meaningful and efficient.

### Spatial Hierarchy of Features

CNNs exploit the assumption that features are compositional — simple local patterns (edges) combine into complex global ones (objects). The convolutional stack encodes this directly: each layer operates on the outputs of the layer before it, building representations layer by layer.

### Feature Locality

Nearby pixels are highly correlated; pixels far apart often are not. Convolutional filters have a small `receptive field` and operate only on their local neighborhood, making them efficient detectors of locally coherent patterns. Receptive fields grow with depth as each layer aggregates over a larger region of the original input.

### Feature Stationarity

A vertical edge at the left of an image and a vertical edge at the right are the same feature in different locations. Weight sharing encodes this stationarity: one filter detects a feature type everywhere, rather than learning a separate detector per position. This dramatically reduces the parameter count compared to a fully connected layer.

### Sufficient Data and Normalization

- `Sufficient data:` CNNs generalize poorly when training data is scarce, falling back to memorizing training examples. Large labeled datasets or transfer learning from pre-trained models mitigate this.
- `Normalized input:` Scaling pixel values to $[0,1]$ or $[-1,1]$ stabilizes training by preventing large input magnitudes from producing large gradients that destabilize weight updates.

---

## Summary

- CNNs are designed for grid-structured data (images, video) using learnable convolutional filters that slide across the input to produce feature maps.
- Weight sharing (same filter applied at every spatial position) enables translation equivariance and dramatically reduces parameter count vs. fully connected layers.
- A standard CNN pipeline: convolutional layers (local feature extraction) → pooling layers (spatial downsampling) → fully connected layers (high-level reasoning and output).
- Feature hierarchies emerge through depth: early layers detect edges and textures; intermediate layers detect shapes; deeper layers recognize object parts.
- Max pooling retains the strongest activation in each window; average pooling retains the mean — both reduce spatial resolution and provide translation invariance.
- CNNs assume spatial hierarchy of features, feature locality, feature stationarity, sufficient labeled data, and normalized inputs.

---

## Best Practices

- Normalize pixel values to [0,1] or [-1,1] before training — raw pixel magnitudes (0–255) produce large gradients that destabilize early training.
- Use transfer learning from pre-trained CNN models (e.g., ResNet, VGG) when training data is scarce — fine-tuning requires far fewer labeled examples than training from scratch.
- Apply data augmentation (random crops, flips, color jitter) to artificially increase training set diversity and improve generalization.
- Add pooling layers after convolutional layers to reduce spatial dimensions and computation; avoid pooling too aggressively early as it discards spatial detail needed in later layers.
- Use max pooling as the default — it retains dominant features and produces sparser, more robust representations than average pooling.

---

## Quiz

**Q1:** What is weight sharing in a CNN and why is it important?
> Weight sharing means the same filter weights are used at every spatial position in the input. It gives the network translation equivariance (the same feature is detected wherever it appears) and reduces parameter count dramatically compared to fully connected layers.

**Q2:** What is the purpose of pooling layers in a CNN?
> Pooling layers downsample the spatial dimensions of feature maps, reducing computational cost and providing a degree of translation invariance. Max pooling retains the strongest activation in each window; average pooling retains the mean.

**Q3:** Describe the hierarchical feature learning that emerges in a deep CNN.
> Initial layers detect low-level patterns (edges, color blobs). Intermediate layers combine these into corners, curves, and simple textures. Deeper layers assemble part-level structures (faces, wheels) from intermediate representations. Each layer builds on the previous.

**Q4:** What data assumptions does a CNN make that distinguish it from a fully connected network?
> CNNs assume the input is a spatial grid, that features are local (nearby pixels are correlated), compositional (simple local patterns combine into complex global ones), and stationary (the same feature type can appear anywhere). Fully connected networks make none of these assumptions.

---
# Back Matter

**Source**
- based_on:: [[HTB-COAE-Prep/0-Introduction/0-HTB-Fundamentals-of-AI/0-Module-Information/Module-Description]]

**References**
- see:: [[Section-18-Neural-Networks]] — CNNs extend standard fully-connected networks
- see:: [[Section-21-Introduction-to-Generative-AI]] — CNNs underpin many generative model architectures

**Terms**
- convolution, filter, feature map, pooling, stride, padding, receptive field, max pooling, flattening
