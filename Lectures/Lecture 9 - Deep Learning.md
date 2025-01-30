---
tags:
  - Marinus
  - Lectures
Created: 2025-01-06
---
I missed the first half an hour.

## Optimiers

- optimizers help normalise the learning rate for different parameters. because of course none of them have the same learning rate

## ADAptive Moments (ADAM) 

-  yapped with Emily about her new computer. very impressive machine

## Dropout [Srivastava et al., 2014]

- **Problem**: Neural networks tend to overfit due to "co-adaption" between neurons, where neuron activations become correlated, reducing generalization ability.
- **Solution**: Introduce noise into the network during training to break co-adaption.
### Mechanism
- Each output from a layer is randomly set to zero with a probability $p$.
- A "mask" sampled from a Binomial distribution determines which outputs are dropped.
### Key Details
- Typical values for $p$ range from $[0.1, 0.5]$.
- $p$ is a **hyper-parameter** that must be tuned manually.
- This is not applied during inference
### Dropout at Inference Time
- At testing/inference:
  - No random dropping is performed.
  - Instead, activations of each layer are scaled by $p$.
  - This compensates for the increased expected value caused by random dropping during training.
### Dropout in Convolutional Layers
- Two main options:
  1. Drop individual feature map pixels.
  2. Drop entire feature maps (channels).
### Observations and Effects
- **Training vs. Validation/Test Loss**:
  - Dropout introduces noise during training.
  - Common consequence: training loss is higher than test/validation loss due to added noise in the training process.
  - val loss is also generalisation loss and since dropout decreases some generalisation 

## Batch Normalization [Ioffe and Szegedy, 2015]

- **Problem**: 
  - Machine learning models require normalized inputs and outputs for optimal training.
  - Without normalization, training may fail or be sub-optimal.
  - This issue also applies to inner activations.

- **Solution**: 
  - Google researchers proposed normalizing inputs to layers.
  - Benefits:
    - Faster training (fewer iterations required).
    - Introduces a small degree of regularization.

### Batch Normalization Calculation
For each feature, component-wise:
$$
x = \frac{x - \mathbb{E}[x]}{\sqrt{\text{Var}(x)}} \tag{3}
$$
$$
y = \gamma x + \beta \tag{4}
$$

- This process combines **mean-std normalization** with **linear scaling**.

### Batch Normalization Details [Ioffe and Szegedy, 2015]

- **Batch-wise Computation**:
  - Mean $\mathbb{E}[x]$ and Variance $\text{Var}(x)$ are computed for each batch during training.
  - At test time, fixed population statistics (computed over the entire training set) are used.
  - $\gamma$ and $\beta$ are trainable scaling factors optimized using gradient descent.

- **Effects**:
  - **Reduces "Co-Variate Shift"**:
    - Co-variate shift refers to variations in parameters of a layer due to changes in preceding layers (similar to the butterfly effect).
    - Batch Normalization (BN) minimizes this issue, stabilizing the training process.
  - **Regularization**:
    - Degrees of freedom are constrained as parameters fulfill normalization criteria.
    - This regularizes the model, improving generalization.

## Multi-Task Learning

- **Overview**:
  - Commonly used with neural networks for architectures with multiple outputs.
  - Typically includes a shared trunk of weights to encode common/shared knowledge.

- **Loss Function**:
  - A single loss function is needed for optimization.
  - Most common approach: linearly combine the loss for each task $\ell_i$.
  - Requires labels $y^i$ for each task.

$$
\ell(x, y, \theta) = \sum_{i} w_i \ell_i(f_i(x, \theta_i), y^i) \tag{5}
$$

- **Components**:
  - $f_i(x, \theta_i)$: Output head of the $i$-th task.
  - $\theta_i$: Weights for the $i$-th task head.
  - $w_i$: Weight factor for the $i$-th task loss.

# Architectures for Images

## Network Design

- **Design Process**:
  - Requires selecting:
    - Layers and their connections.
    - Hyper-parameters for each layer:
      - Examples: Number of neurons, filter size (for convolutional layers).
    - Processing steps between layers.

- **Challenges**:
  - Designing optimal architectures manually is difficult.
  - Researchers rely on:
    - Pre-established architectures.
    - Variation of key hyper-parameters:
      - Width (number of neurons).
      - Depth (number of layers).

- **Automated Optimization**:
  - Hyper-parameters can be tuned with algorithms.
  - Techniques include:
    - **Automatic Machine Learning (AutoML)**.
    - **Neural Architecture Search (NAS)**.

## VGG [Simonyan and Zisserman, 2014]

- **Overview**:
  - Built on AlexNet with modifications for increased depth.
  - Replaces larger filters with two $3 \times 3$ filters to emulate $5 \times 5$ filters.
  - Only $3 \times 3$ filters are used across all layers.

- **Configurations**:
  - VGG has 16-19 layers depending on the configuration.

- **Architecture**:
  - Five stacks of 2-3 convolutional layers.
  - Each stack is followed by $2 \times 2$ max-pooling.
  - Ends with three dense (fully connected) layers:
    - Sizes: 4096, 4096, 1000.
  - Filters in each stack double compared to the previous stack, starting with 64 filters.

- **Performance**:
  - Parameter count: 133-144 million.
  - Achieved a **top-5 error** of **6.8%** on ImageNet using an ensemble of two networks.

## Inception [Szegedy et al., 2014]

- **Overview**:
  - Developed by Google.
  - Key principle: Perform convolutions at multiple scales and concatenate their outputs.
  - These operations are grouped into an "Inception module," which can be stacked to form a network architecture.

### Inception Module Variants
1. **Naïve Version**:
   - Combines $1 \times 1$, $3 \times 3$, $5 \times 5$ convolutions, and $3 \times 3$ max-pooling.
2. **With Dimension Reduction**:
   - Uses $1 \times 1$ convolutions to reduce dimensionality (channels) before larger convolutions, minimizing computational cost.

### Network Architecture
- 22 layers with 9 Inception modules.
- **Intermediate Classifiers**:
  - Added to mitigate vanishing gradients.
- **Performance**:
  - Parameter count: 5 million.
  - Achieved **top-5 error** of **6.67%** on ImageNet.

### Additional Details
- **Dimensionality Reduction**:
  - $1 \times 1$ convolutions reduce the number of input channels.
  - The number of filters is less than the input channel count.

## ResNet [He et al., 2016]

- **Problem**:
  - Increasing network depth (e.g., GoogleNet, VGG) often leads to performance degradation.
  - This is attributed to the **vanishing gradient problem**, where gradients diminish as they propagate backward through layers.

- **Proposed Solution**:
  - Introduced a method to propagate gradients effectively in very deep networks.
  - Allows deeper networks to perform better by addressing gradient flow issues.

### Observations
- **Performance Comparison**:
  - On CIFAR-10:
    - Deeper networks (e.g., 56 layers) without this solution underperform shallower networks.
  - On ImageNet:
    - A 34-layer network with this method outperforms an 18-layer network.
  - Networks without this method see reduced accuracy as depth increases.

## Residual Block Design [He et al., 2016]

- **Key Feature**: 
  - Residual blocks contain **skip connections** that propagate gradients through the layer.
  - If a layer computes $F(x)$, adding the skip connection results in the output:
    $$
    F(x) + x
    $$
    ![[Pasted image 20250106101906.png]]
### Constraints
- The output of $F(x)$ must have the **same dimensionality** as $x$ to enable addition.
- If only spatial dimensions match (but not channels), **concatenation** can be used instead of addition.

### Purpose
- The primary goal is to pass gradients forward efficiently via skip connections.
- This design alleviates the **vanishing gradient problem**, enabling stable training of very deep networks.

## Summary of Image Architectures

### Key Takeaways
- **AlexNet and VGG**:
  - Share similar designs but differ in:
    - **Depth**.
    - **Filter sizes** (e.g., $11 \times 11$ and $5 \times 5$ in AlexNet vs. $3 \times 3$ in VGG).

- **Inception**:
  - Utilizes **multi-scale convolutions**:
    - Combines multiple convolutional operations and concatenates their outputs.

- **ResNet**:
  - Employs **residual blocks**:
    - Enables much deeper networks by facilitating gradient flow.
    - Allows easier optimization and training.

- **Broader Applications**:
  - Residual connections from ResNet are widely used in other architectures:
    - Examples: **DenseNet**, **Transformers**, **U-Net**, etc.

## Some Architectural Patterns

### Common Patterns in Large-Scale Networks
- In most large-scale networks for ImageNet, the number of filters increases with depth:
  - **AlexNet**: Filters = 96, 256, 384, 512.
  - **VGG**: Filters = 64, 128, 256, 512, 512.
  - **ResNet**: Filters = 64, 128, 256, 512.
- Observations:
  - Filter counts often follow powers of two.
  - Filter sizes are usually odd numbers.

### Max-Pooling and Performance
- **Issue**: Excessive Max-Pooling can reduce performance.
- **Design Practice**:
  - Designers often group several convolutional modules together before applying Max-Pooling.
  - VGG exemplifies this design pattern.
- **Impact of Pooling**:
  - Pooling determines the minimum input image size required.

# Architectures for Sequences

## Basic RNN Architectures

![[Pasted image 20250106102423.png]]

### Key Architectures
- **One-to-One**:
  - A simple feed-forward neural network.
  - No timesteps or sequential data are used.

- **One-to-Many**:
  - Single input timestep, multiple output timesteps.
  - Typically implemented by:
    - Feeding back the output of one timestep as input to the next timestep.
    - Example: Auto-regressive models.

- **Many-to-One**:
  - Multiple input timesteps, single output timestep.
  - Implemented as a standard RNN:
    - Predicts an output for each timestep.
    - Discards intermediate outputs, keeping only the output of the last timestep.

- **Many-to-Many**:
  - Multiple input timesteps, multiple output timesteps.
  - Two cases:
    1. The number of input and output timesteps is the same.
    2. The number of output timesteps is variable.
      - The model learns when to stop by outputting a special marker token.

### Handling Variable Output Timesteps
- The network:
  - Consumes all input timesteps first.
  - Then produces output timesteps.
- Input for each additional timestep is the output of the previous timestep.

### Encoder-Decoder Architectures
- Typically used for variable-sized inputs and outputs.

## Bi-Directional RNNs
![[Pasted image 20250106102722.png]]
- **Normal RNNs**:
  - Are **causal**: They only consider information from the past to produce a prediction.

- **Need for Bi-Directional RNNs**:
  - Some applications require looking at the entire sequence (past and future) before making a prediction.
  - Examples: **Machine translation**, **speech recognition**.

- **How Bi-Directional RNNs Work**:
  - Consist of two sets of RNN cells:
    1. One processes the sequence in the forward direction (from past to present).
    2. The other processes the sequence in the backward direction (from future to past).
  - The outputs from the forward and backward passes are combined to make the final prediction.
### Key Features
- Both past and future context are utilized for better predictions.
- Commonly used in tasks requiring a global understanding of sequences.


## Sequence-to-Sequence Models

- **Overview**:
  - A meta-architecture designed for problems involving:
    - An **input sequence**.
    - A predicted **output sequence**.
  - Commonly implemented with RNNs.

### Encoder-Decoder Model
- **Purpose**:
  - Encodes all information from the input sequence into a **hidden state vector**.
  - The decoder uses the hidden state to generate the output sequence.

- **Key Mechanism**:
  - The **encoder**:
    - Consumes the input sequence fully and produces a hidden state vector.
    - Typically uses the hidden state of the last sequence element.
  - The **decoder**:
    - Takes the hidden state and its own previous output to generate the sequence.
    - Continues until a **stop token** is produced.
    - Operates in an **autoregressive** manner (predicts one token at a time).

### Use Case
- Effective for tasks like machine translation, text summarization, and speech recognition.
![[Pasted image 20250106102948.png]]

## Sequence-to-Sequence Models (Encoder-Decoder)

### Architecture
- **Encoder**:
  - Processes the input sequence $x_1, x_2, x_3, \dots$ and generates hidden states $h_1, h_2, h_3, \dots$.
  - The final hidden state (encoder vector) represents the summarized information from the entire input sequence.
  - Intermediate encoder outputs are ignored.

- **Decoder**:
  - Takes the encoder vector as its initial hidden state.
  - Predicts the output sequence $y_1, y_2, \dots$ in an **autoregressive** manner:
    - Each predicted output is fed back as input for subsequent predictions.

### Key Notes
- Each RNN node represents an invocation of the encoder or decoder RNN.
- The encoder vector is critical for transferring information between the encoder and decoder.
- In the decoder:
  - Outputs are used as inputs for subsequent RNN activations.
## Issues with Sequence-to-Sequence (Seq2Seq) Models

- **General Challenges**:
  - Seq2Seq models are effective but face difficulties with **long-term dependencies**.

- **Specific Limitations**:
  - In tasks like **Neural Machine Translation**:
    - All input sequence information must be stored in a **single hidden state vector**.
  - Even with high-dimensional hidden state vectors:
    - They generally cannot capture all the required information.
    - This results in loss of context, particularly for long sequences.

- **Empirical Evidence**:
  - These limitations have been validated through experimental studies.

## The Concept of Attention

- **Definition**:
  - Attention is a fundamental concept in machine learning.
  - It allows the model to decide **where to look** in the input, focusing on the most relevant parts.
  - Reduces processing complexity while ensuring the most important inputs or features are utilized.

- **Mechanism**:
  - An attention model typically has two components:
    1. A mechanism that predicts **"where to look"** in the input.
    2. A method to **combine this focus** with the input or extracted features.

- **Introduction to Soft-Attention**:
  - First explored by **Bahdanau, Cho, and Bengio (2014)**.
  - Applied to **Seq2Seq models** to enhance their performance by addressing long-term dependency issues.


I didnt end up taking notes of the Soft-attention and Multi-head Attention slides. reference back to these later
