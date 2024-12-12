---
tags:
  - _FirstPass
  - Marinus
  - Lectures
  - LLM
Date: 2024-12-12
---
"its only 70 slides"

- LLMs are neural networks. wow

# what are neural networks

- neural networks name does not come from the brain
- allow for very non-linear behaviours
- SVM is a one layer neural network with a specific activation something and something

## Perceptron 

- we take the sign of the result
- one of hte first brain inspired in the 60s 
- initialise weights randomly 
- for each datapoint in dataset you update update weights iteratively with this equation
	1. Initialize all weights $w_i$ with random values in a small range (e.g., $[-0.5, 0.5]$).

2. For each data point $(x_i, y_i)$ in the training set:
   - Compute current output:
     $$
     \hat{y}_j = \text{sgn}(\mathbf{x} \cdot \mathbf{w} + b)
     $$
   - Update each weight $w_k$:
     $$
     w_k(n+1) = w_k(n) + \alpha (\hat{y}_i - y_i) x_{ik}
     $$

- People found this experimentally

![[Pasted image 20241212151403.png]]

## Multi-layer preceptron

- think of it being one layer 
$f(x) = \sigma(\mathbf{w} \cdot \mathbf{x} + b)$
- then multiple layers
$h_0 = x$
$h_1 = \sigma(\mathbf{W}_1^\top h_0 + \mathbf{b}_1)$
...
$h_L = \sigma(\mathbf{W}_L^\top h_{L-1} + \mathbf{b}_L)$

- blocks of layers, stacks multiple layers above each other to have more inputs.
- each layer has its own weights also called a FeedForward neural netowrk4
- Layer should do some non-linear equations
> [!def] Layer

> [!def] Activations

> [!def] Activation Functions

- model without activation function is a lienar functions. So the activation fucntion is very important
- the output of one layer is the input of another layer
	- hence the stacking 

# Neurons 

- perceptron is also called a neuron 
- single perceptron computes this: $\sigma(\mathbf{w} \cdot \mathbf{x} + b)$
- this gives you a vector in the network.

## Perceptron layer (Dense Layer)

- d dimensional features to m dimensional feature
- $f(x) = \sigma(\mathbf{W} \mathbf{x} + \mathbf{b})$
- Covers theorem more dimensions better
- m x d + 1
- input $x$
- output $f(x)$

## Dense Layer Parameters

- number of neurons:
	- Hyper parameter to be tuned
- Activation function:
	- Defines the non-linear behaviour of the network

## Tasks with MLPs
- Last layer in the network generally defines the task to be done:
- Classification:
	- Dense layer with $M = C$ neurons and softmax activation with cross-entropy loss
		- stack as many layers as you want
	- $M$ corresponding to the number of classes
- Regression:
	- Dense Layer with $M$ Neurons (same dimensionality of target value) a linear or sigmoid activation, and a mean squared error loss
	- if target is normalised to \[1, 0\] then sigmoid function is preferred
		- predict age between 1 and 0. age between 0 and 100
	- if not, use linear activation (can have negative values)

## Number of parameters

- $M * (D+1$) 
	- weights of matrix is $M*D$
	- bias has $M$ elements
- some more notes

## MLP structure
![[Pasted image 20241212152909.png]]

## Learning process

- first large vectors
- flat vectors
We use gradient descent and its variants. Considering $\mathbf{w} = [\mathbf{W}_1, \mathbf{b}_1, \mathbf{W}_2, \mathbf{b}_2, ..., \mathbf{W}_L, \mathbf{b}_L]$:

$$
\mathbf{w}_{m+1} = \mathbf{w}_m - \alpha \frac{\partial \ell}{\partial \mathbf{w}}
$$
- can train LLM from scratch using this

- Optimisers: ways to change the gradient 

## Stochastic Gradient Descent

Gradient descent requires computing the loss over all training samples:

$$
\frac{\partial \ell}{\partial \mathbf{w}} = \sum_{i=1}^{N} \frac{\partial \ell(f(x_i), y_i)}{\partial \mathbf{w}}
$$

Stochastic Gradient Descent (SGD) simplifies this by using non-overlapping batches:

$$
\frac{\partial \ell}{\partial \mathbf{w}} = \sum_{i \in \text{batch}} \frac{\partial \ell(f(x_i), y_i)}{\partial \mathbf{w}}
$$

This iterates over the training set in batches of size $B$.
 - split dataset into batched and then calculate the loss over batches
 - computationally feasible
 - still works really well even if its an approximation
 - batching is not in the books for today

parallise forward pass over the batches. utilises the gpus parallel processing advantage

- if someone uses batches with gradient descent, they are using stochastic descent 

- **SGD** introduces noise into learning but is generally tolerable.
- Loss and gradient are computed on a subset of data called a **batch**.
- The size of this subset is called the **batch size** ($B$).

- **Types**:
  - $B = 1$: Stochastic Gradient Descent (SGD).
	  - one sample at a time. not ideal
	  - more samples better but more computation
  - $B > 1$: Mini-Batch Gradient Descent (MGD), often still referred to as SGD.

- **Batch size ($B$)** affects memory/compute requirements and the gradient's quality/noise.
	- has to be correlated to ram and gpu power

- A complete pass over the dataset is called an **epoch**.

### Backpropagation
- classical way of training
- dont need to go though this. very complex. im practive we use automatic computations. python boi
- In neural networks (e.g., MLP), the gradient is computed using the **chain rule**.
- Backpropagation applies the chain rule to calculate gradients layer by layer.

**Chain Rule for $y = f(g(x))$:**
$$
\frac{\partial y}{\partial x} = \frac{\partial f}{\partial g} \frac{\partial g}{\partial x}
$$

**For a neural network:**
- $h_0 = x$
- $h_1 = \sigma(\mathbf{W}_1^\top h_0 + \mathbf{b}_1)$
- ...
- $h_L = \sigma(\mathbf{W}_L^\top h_{L-1} + \mathbf{b}_L)$
- $\ell = \text{loss}(h_L, y)$

Gradient computation:
$$
\frac{\partial \ell}{\partial \mathbf{w}} = \frac{\partial h_L}{\partial \mathbf{h}_L} \frac{\partial h_L}{\partial h_{L-1}} \dots \frac{\partial h_1}{\partial \mathbf{w}}
$$

### Example of Backpropagation

- Considering $h_2 = \sigma(u_2)$ and $u_2 = \mathbf{W}_2 h_1 + \mathbf{b}_2$:
$$
\frac{\partial h_2}{\partial h_1} = \frac{\partial h_2}{\partial u_2} \frac{\partial u_2}{\partial h_1}
$$
$$
= \frac{\partial \sigma(u_2)}{\partial u_2} \mathbf{W}_2
$$
- Key terms:
  - $\frac{\partial \sigma(u_2)}{\partial u_2}$: Gradient of the activation function.
  - $\frac{\partial u_2}{\partial h_1} = \mathbf{W}_2$.

- Similar derivations can be done for other layers, but a simpler framework will be introduced later.

# Terminology

- Forward pass
- Backward pass 
	- pass where you compute the gradient

### Automatic Differentiation - Autograd

- **Gradient computation** is cumbersome manually, but modern libraries automate this.

#### Two Modes:
1. **Forward Mode**:
   - Traverses the chain rule from inside to outside.
   - Example: TensorFlow (graph mode).
   - Requires building a computational graph; can be more optimal.

2. **Reverse Mode**:
   - Traverses the chain rule from outside to inside.
	   - when you apply the formula you can go from the LHS to the RHS
   - Example: PyTorch, TensorFlow (eager mode).
   - More straightforward for implementation.
- !we now use automatic gradient descent.

- Each operation in the computational graph must have gradient operations implemented.

### Weight Initialization and Symmetry

- Weights need initial values to use gradient descent.
- Commonly initialized using small random values (e.g., Gaussian or Uniform distribution in range $[-0.5, 0.5]$).
- Proper initialization methods help optimize training.

#### Breaking Symmetry: (symmetry breaking)
- If all weights are initialized to zero:
  - Gradients remain zero.
  - Training cannot proceed.
- **Random initialization** prevents this, breaking symmetry and enabling gradient computation.
- layers dont have the same weights. they need to have different weights. and need to start randomly at different weights
### Tuning the Learning Rate
- if you move too much or too fast you go past and the model "explodes"
- The **learning rate** and the number of **training epochs** are crucial for successful training.
- **Learning Rate**:
	- Should be set to produce a consistent decrease in the loss value across batches.
- **Number of Epochs**:
	- Should ensure that the loss converges to a small value.
	- Too few epochs can result in **underfitting**.
![[Pasted image 20241212154451.png]]
- "Climbing the mountain down" when your learning rate is too high haha
- "needs to be a bit smaller so that you can go into the hole"
- cannot predict the right learning rate, you need to figure that out yourself

## Problems
>[!note] In exam!!!
- **High Learning Rate**:
	- Loss diverges to infinity or NaN.
	- May cause loss to oscillate instead of decreasing.
- **Low Learning Rate**:
	- Loss decreases slowly.
	- Training becomes unnecessarily long.
- **Good Learning Rate**:
	- Loss decreases quickly.
	- Converges within a reasonable number of epochs.
## Solutions

- **High Learning Rate**:
  - Divide the current learning rate by 10 until the loss decreases consistently.
- **Low Learning Rate**:
  - Gradually increase the learning rate by factors of 2–5.
- **Good Learning Rate**:
  - Keep this learning rate.
### Activation Functions

| Name            | Range       | Function                                 |
|------------------|-------------|-----------------------------------------|
| **Linear**       | $[-\infty, \infty]$ | $g(x) = x$                           |
| **Sigmoid**      | $[0, 1]$    | $g(x) = (1 + e^{-x})^{-1}$             |
| **Hyperbolic Tangent** | $[-1, 1]$ | $g(x) = \frac{e^{2x} - 1}{e^{2x} + 1}$ |
| **ReLU**         | $[0, \infty]$ | $g(x) = \max(0, x)$                    |
| **Softplus**     | $[0, \infty]$ | $g(x) = \ln(1 + e^x)$                 |
| **Softmax**      | $[0, 1]^n$  | $g(x) = \frac{e^{x}}{\sum_k e^{x_k}}$   |
| **Leaky ReLU**   | $[-\infty, \infty]$ | $g(x) = \max(0.01x, x)$             |
| **Parametric ReLU** | $[0, \infty]$ | $g(x) = \max(\alpha x, x)$          |
| **Swish**        | $[-1, \infty]$ | $g(x) = x \cdot \text{sigmoid}(x)$    |

#### Notes:
- **ReLU**: A good default for hidden layers.
- **Output layers**:
  - Use **Softmax** for classification.
  - Use **Linear**, **Softplus**, or **Sigmoid** for regression.
- sigmoid saturates
	- derivative of constant value is 0
- the dead something problem. couldnt catch what he said.
- based on the task you need the right activation and loss functions
- 

### Issues with Training

1. **Vanishing Gradients**:
   - Saturating activations (e.g., sigmoid, tanh) have near-zero gradients at extreme inputs.
   - This results in very small gradients, slowing or halting training in hidden layers.
   - **Solution**: Use non-saturating activations, like ReLU.

2. **Non-Convex Loss**:
   - The loss function is non-convex due to the non-linear nature of neural networks.
   - convexity: 
	   - in math handout
	   - means shape is a smile
	   - easy to do gradient descent 
	   - not convext, weird curve, can get stuck in a local minimum
	   - lose this in non-linear models
   - Challenges include:
	   - Flat regions
	   - Saddle points
	   - Multiple local and global optima

1. **Potential Overfitting**:
   - The large number of parameters ($P$) in neural networks increases the risk of overfitting.
   - **Solution**:
     - Careful network design.
     - Use regularization techniques.

2. **Hyper-Parameters**:
   - Neural networks require tuning of many hyper-parameters:
     - **Model structure**: Number of layers, neurons.
     - **Training process**: Batch size, learning rate.
   - Proper tuning is crucial for achieving good performance.
   - he has given us intuitions for how to do this. rules and learning rate
   - number of nurons doesnt have rules really. need to look in the next lecture at existing architectures of what works

# Convolutional Neural Networks
- if you can flatten any data into a vector you can do anything
### Using MLPs with Image Inputs

- An $M$-neuron layer connected to a $W \times H$ image requires $M \times W \times H$ weights.
- **Issues**:
  - Too many weights to learn, making the model inefficient.
  - No built-in **translation equivariance** in the network design.
- **Conclusion**: This approach is far from ideal for image inputs.

### Convolutional Neural Networks (CNNs)

- **Key Features**:
  - CNNs constrain weights to better handle specialized input data like images.
  - Replace matrix multiplication in fully connected ANNs with the **convolution operator**.

- **Convolution Operation**:
  - Uses kernels (filters) to process the input image.
  - Example: A kernel applied to an image extracts specific features, creating a convolved image.

- **Advantages**:
  - Reduces the number of parameters.
  - Introduces translation equivariance, making CNNs more suitable for image data.
![[Pasted image 20241212161336.png]]
- Edges are features of the images

![[Pasted image 20241212161402.png]]
- something about slideing and weights not being specific but rather shared.
- weight sharing
- make assumptions about the data
	- works better

- **Yann LeCun's Idea (1980s)**: 1989
  - Connect a neuron only to a **spatial neighbourhood** of the image and slide it across the image.
  - Enables learning the same weights regardless of their location in the image (**less weights to learn**).
  - Represented as **convolution**, where the convolution filter contains the neuron weights.

- **Subsampling**:
  - To reduce the information, subsample the outputs after convolution.

### Convolution Operation

- The convolution operation calculates the output at $(x, y)$ using a kernel and input:

$$
\text{out}(x, y) = \sum_i \sum_j \text{input}(x + i, y + j) \cdot \text{kernel}(i, j)
$$

- **Coordinates**:
  - $i, j$: Pixel coordinates within the kernel dimensions (width and height).
  - $x, y$: Pixel coordinates over the image or feature map dimensions.


- first is kernel, what you slide over the images. back prop minimises weights

![[Pasted image 20241212161636.png]]
![[Pasted image 20241212161721.png]]
- shift pixels you will get a similar complexions

### Idea of Convolutional Networks

- Use **learnable filter kernels** (parameters or weights).
- Build multiple 2D **feature maps** (outputs of convolution).
- Each feature map in one layer connects to each feature map in the previous layer.
- Different filter kernels are used for each pair of feature maps.

#### Key Advantages:
- **Exploitation of Image Structure**:
	- ]better for images
	- Compared to fully connected layers, convolutional networks utilize the **spatial order** of pixels.
- **Benefits**:
	- Reduces the number of connections and weights.
	- Gains some **rotation** and **illumination invariance**.


### Convolutional Layers

#### **Purpose**:
- Dimensionality reduction
- Feature learning
- Weight sharing

#### **Forward Pass**:
$$
Z^j = \sum_{i=0}^{l-1} \mathbf{W}^{ij} * \mathbf{X}^i + b^{ij}, \quad Y^j = g(Z^j)
$$
#### **Components**:
- $\mathbf{X}^i$: $i$-th feature map in the input.
- $Y^j$: $j$-th feature map in the output.
- $\mathbf{W}^{ij}$: Filter kernel between feature maps $i$ and $j$.
- $b^{ij}$: Bias between feature maps $i$ and $j$.

#### **Notes**:
- Filter kernels are typically moved by **one pixel per step** (stride of 1), meaning they **overlap**.
- same as perceptron but the weights are different plus activation function g


### Pooling Layers

#### **Purpose**:
- Dimensionality reduction.
- Local translation invariance.
#### **Key Features**:
- **Max-Pooling**:
  - Example: A 2x2 filter with a stride of 2.
  - Retains the maximum value in each region.
- **Functions**:
  - Can use **max pooling** or **average pooling**.
- **Global Pooling**:
  - Performs a single operation (max/average) over the whole feature map.
  - Produces a $1 \times 1$ spatial output.
#### **Benefits**:
- Summarizes **low-level features**.
- Reduces **feature dimensions** and the number of parameters.

- Images are large and lots of info
- how do you summarise
- feature maps 
- divide 
![[Pasted image 20241212162025.png]]

if you do this, you get this:![[Pasted image 20241212162108.png]]

Reduces computation

### Building Convolutional Networks
- convolutions are meant to extract features
1. **Feature Extraction**:
   - Stacks of **convolution** and **max-pooling layers** are used to extract hierarchical features.
   - Ends with **flattening** to produce a large feature vector.
2. **Task (Classification or Regression)**:
   - Dense layers are stacked to process the feature vector from previous layers.
   - These layers learn to perform the final task.
3. **Training**:
   - The entire CNN is trained using **stochastic gradient descent**.
   - Weights for both feature extraction (convolutional layers) and task (dense layers) are learned end-to-end.
![[Pasted image 20241212162224.png]]
- adapts to model you are training. 
- will learn different features for that task specifically
- more interperatable 
- 50% black box
![[Pasted image 20241212162237.png]]
### Input-Output Shapes

| Layer               | Input Shape       | Output Shape                        | Remarks                                                                                               |
| ------------------- | ----------------- | ----------------------------------- | ----------------------------------------------------------------------------------------------------- |
| **Fully Connected** | $(f,)$            | $(n,)$                              | Operates on the last dimension of the input.                                                          |
| **Convolutional**   | $(w_i, h_i, c_i)$ | $(w_o, h_o, c_o)$                   | Output size: $x_o = \frac{x - f + 2p + 1}{s}$ borders, padding, set for model to apply to whole image |
| **Pooling**         | $(w_i, h_i, c)$   | $(\frac{w_i}{p}, \frac{h_i}{p}, c)$ | $p$ is the pooling factor (e.g., $p=2$).                                                              |
| **Global Pooling**  | $(w_i, h_i, c)$   | $(1, 1, c)$                         | Consumes spatial dimensions.                                                                          |
#### Notation:
- $f$: Input feature dimensionality or filter size.
- $c$: Channels.
- $n$: Number of neurons.
- $p$: Padding factor.
- $s$: Stride.
- For maintaining convolutional dimension, set $p = \frac{f - 1}{2}$ with $s = 1$.

- can also be seen as a kernel model

# Recurrent Neural Networks

- CNN and other are called feed forward networks
- RNN "memory"
### Recurrent Neural Networks (RNNs)

An RNN differs from a feed-forward network in three key ways:

1. **Memory**:
   - RNNs maintain an internal state that persists between timesteps.
2. **Temporal/Sequence Information**:
   - The internal state ($h$) captures temporal or sequential (information) patterns from data.
3. **Variable-Sized Inputs/Outputs**:
   - RNNs handle variable-sized inputs and produce variable-sized outputs, often using an encoder-decoder architecture.

### RNN Equations

1. **Hidden State**:
   $$
   h_t = \sigma_h(\mathbf{W}_{hh} h_{t-1} + \mathbf{W}_{xh} x_t + \mathbf{b}_h)
   $$
2. **Output**:
   $$
   y_t = \sigma_y(\mathbf{W}_{hy} h_t + \mathbf{b}_y)
   $$
- All functions of time
- the next value is determined by the previous value --> Recurrent
- hidden states need activation function that prevents it from going to infinity 
#### Notes:
- **Weight Matrices**:
  - $\mathbf{W}_{hh}$: Hidden-to-hidden weights.
  - $\mathbf{W}_{xh}$: Input-to-hidden weights.
  - $\mathbf{W}_{hy}$: Hidden-to-output weights.
- **Bias Vectors**:
  - $\mathbf{b}_h$: Bias for the hidden state.
  - $\mathbf{b}_y$: Bias for the output.
- $\sigma_h$ and $\sigma_y$: Activation functions for the hidden state and output, respectively.

### Symbols and Their Meanings

| Symbol | Meaning                      |
|--------|------------------------------|
| $t$    | Timesteps, temporal dimension |
| $x$    | Input values                 |
| $y$    | Output values                |
| $h$    | Hidden state                 |
| $W$    | Weight matrices              |
| $b$    | Bias vectors                 |
| $\cdot$| Component-wise product       |

### Key Feature of RNNs: Weight Sharing

- **Weight/Bias Sharing**:
  - RNNs share weight and bias matrices across timesteps.
  - Ensures that these weights/biases are the same for each timestep, making the operation **sequence-invariant**.

- **Advantages**:
  - Reduces the number of parameters needed for learning.
  - Forces the RNN to learn **timestep-invariant features**.
  - Similar to how CNNs achieve **translation invariance**.

- Less weights are better because you control the variance

### Hidden State in RNNs

- The dimensionality of the hidden state ($h_t$) **does not need to match** the input dimensionality.
- Similar to feedforward NNs, RNNs can project the input into a higher-dimensional space, which includes the hidden state.
- This enables:
  - Capturing sequence features in the hidden state.
  - Modeling both **long-term** and **short-term** dependencies in the sequence.

### RNN Hyper-Parameters

- **Key Hyper-Parameter**:
  - Number of units/neurons in the recurrent layer, which determines the dimensionality of the hidden state.
- **Activations**:
  - $\sigma_h$: Typically sigmoid or tanh, chosen to prevent uncontrolled growth of the hidden state.
  - $\sigma_y$: Configurable by the user.
- **Output Configurations**:
  - Return a **sequence of outputs** ($y_t$ for all timesteps).
  - Return the **output of the last timestep** ($y_n$).
- **Stateful RNNs**:
	- means they have a state, on or off over time
	- Retain the hidden state across batches of inputs.
	- Hidden state can be manually reset (e.g., to zero).
	- starts at zero
- bouded to \[0,1\]

### Fundamental Issues in RNNs

1. **Vanishing and Exploding Gradients**:
   - Recurrent application of weight matrices can cause gradients to:
     - **Vanish** (approach zero) or
     - **Explode** (grow uncontrollably).
   - This issue is more likely for long sequences.

2. **Long-Term Dependencies**:
   - RNNs struggle to model dependencies across long sequences.
   - The hidden state ($h_t$) must encode all information from the sequence, which limits its effectiveness for capturing long-range relationships.
	   - fixed size not infiniatly dimensional.
	   - need to forget something for more information

- more weights more complexity

### Vanishing or Exploding Gradients in RNNs

- RNNs involve repetitive multiplication by a weight matrix $\mathbf{W}$.
- Using eigen decomposition $\mathbf{W} = \mathbf{Q} \Lambda \mathbf{Q}^\top$, the state $h_t$ can be approximated as:
  $$
  h_t = \mathbf{Q} \Lambda^t \mathbf{Q}^\top
  $$
  lambda to the power of T
  
#### Key Insights:
- **Eigenvalues in $\Lambda$**:
  - If eigenvalues $< 1$, they **converge to zero**, causing vanishing gradients.
  - If eigenvalues $> 1$, they **explode to infinity**, causing exploding gradients.
	- 0.01 they will explode
	- if set to 1 then regularisation
- **Implications**:
  - These phenomena occur in long sequences due to the repeated application of the kernel matrix $\mathbf{W}$.
  - Results in instability during training.

### Tricks for exploding gradients

> [!def] Element-Wise Clipping
> - Clip any component of the gradient $\mathbf{g}$ with absolute value larger than $v$.
> - For elements where $g_i > v$, set $g_i = v$.

> [!def] Norm Clipping
> - Clip the norm of the gradient. If $\|\mathbf{g}\| > v$, set:
  $$
  \mathbf{g} = \frac{v \mathbf{g}}{\|\mathbf{g}\|}
  $$
### **Notes**
- The clip parameter $v$ must be chosen carefully.
- Can be determined through monitoring gradients during training and using trial and error.

![[Pasted image 20241212163929.png]]

### Long-Short Term Memory (LSTM) [Hochreiter and Schmidhuber, 1997]

- **Developers**: Sepp Hochreiter and Jürgen Schmidhuber (1997).
	- Jurgen jokes he invented everything with different names ha ha
- **Advantages**:
  - Advanced RNN cell with a reduced risk of **vanishing or exploding gradients**.
  - Better at modelling **long-term dependencies** within sequences.
- **How It Works**:
  - Uses **gated connections** to control the flow of information.
  - Splits the state $h_t$:
    - Part for **output prediction**.
    - Part for learning **features from the sequence**.

### Long-Short Term Memory (LSTM) Gates

An LSTM computes three gates:

1. **Forget Gate ($g_f$)**:
   $$
   g_f = \sigma(\mathbf{W}_{hf} h_{t-1} + \mathbf{W}_{xf} x_t + b_f)
   $$
2. **Input Gate ($g_i$)**:
   $$
   g_i = \sigma(\mathbf{W}_{hi} h_{t-1} + \mathbf{W}_{xi} x_t + b_i)
   $$
3. **Output Gate ($g_o$)**:
   $$
   g_o = \sigma(\mathbf{W}_{ho} h_{t-1} + \mathbf{W}_{xo} x_t + b_o)
   $$
#### Gate Functionality:
- **Sigmoid Activation ($\sigma$)**:
  - Each gate uses a sigmoid function to produce values between 0 and 1.
  - They behave like a switch: **0** (off) or **1** (on).

### LSTM Cell State

- The hidden state $h_t$ is divided into two parts:
  - $h_t$: Predicts the output.
  - $C_t$: **Cell state**, models cross-timestep dependencies.

1. **Cell State Proposal ($\hat{C}$)**:
   $$
   \hat{C} = \tanh(\mathbf{W}_{hc} h_{t-1} + \mathbf{W}_{xc} x_t + b_c)
   $$
2. **Final Cell State ($C_t$)**:
   $$
   C_t = g_f \cdot C_{t-1} + g_i \cdot \hat{C}
   $$
#### Functionality:
- **Forget Gate ($g_f$)**: Determines how much of the previous cell state is used.
- **Input Gate ($g_i$)**: Controls how the proposal is added to the final cell state.
- **Tanh Activation**: Produces values in $[-1, 1]$, intuitively adding or removing information in $\hat{C}$.
- conceptually, either pos or neg, either add or remove info from hidden state c

### LSTM Output/Hidden State

The output/hidden state is computed as:

$$
h_t = g_o \cdot \sigma_y(C_t)
$$

- **Output Gate ($g_o$)**: Controls which parts of the cell state $C_t$ are used as output.
- **$\sigma_y$**: Activation function for the output, configurable by the user.

#### Advantages of LSTM:
- LSTMs are more complex than regular RNNs, but this complexity helps:
  - Better model **long-term dependencies**.
  - Reduced issues with **vanishing** and **exploding gradients**.
![[Pasted image 20241212164403.png]]

### Summary of Neural Networks

1. **Multi-Layer Perceptrons (MLPs)**:
   - For general data, especially flattened data like **tabular data** or **raw features**.
2. **Convolutional Networks (CNNs)**:
   - For **image data** and **matrix-data**, where local correlations between neighboring elements exist.
3. **Recurrent Networks (RNNs)**:
   - For **sequence data** like **text**, **audio**, **time series**, etc.
4. **Combination of Networks**:
   - These types of networks can be combined, such as building **convolutional RNNs**, etc.

### What You Must Learn

This lecture covers a lot about three types of neural networks. Key topics to learn:

1. **Types of Neural Networks**:
   - Their basic structure.
   - Concepts of **forward pass**, **backward pass**, **autograd**, **weight sharing**, and **types of layers**.
2. **Training**:
   - Training with **SGD**.
   - Tuning **learning rates**.
   - Setting **batch size**.
3. **Issues and Applications**:
   - Issues specific to each type of neural network.
   - Types of data they can be applied to.

# Take Home messages

### Take Home Messages

- **Neural Networks** are state-of-the-art for tasks in **NLP**, **Computer Vision**, and more.
- They are built with interconnected layers that either **learn features** or **perform tasks**. Different layers are suitable for various types of data and features.
- **Training Neural Networks**:
  - They are trained using **SGD**.
  - Since the loss is non-convex, there is no guarantee of obtaining a **global optimum**.
- **Backpropagation** is used to compute gradients analytically, exploiting the **compositional nature** of neural networks.
