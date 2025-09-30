# Neural Network Forward and Backward Propagation Equations

! Note this was generated with some AI. !

This document contains all the mathematical equations for the neural network implementation in `example.py`, including standard layers, batch normalization, dropout, and various optimization algorithms.

## Table of Contents
- [Matrix Shapes](#matrix-shapes)
- [Forward Propagation](#forward-propagation)
- [Backward Propagation](#backward-propagation)
- [Activation Functions](#activation-functions)
- [Loss Functions](#loss-functions)
- [Optimization Algorithms](#optimization-algorithms)

## Matrix Shapes

Understanding matrix dimensions is crucial for implementing neural networks correctly. Here are the key dimensions used throughout:

### Basic Dimensions
- $m$: batch size (number of training examples)
- $n^{[l]}$: number of units (neurons) in layer $l$
- $n^{[0]}$: input feature dimension
- $L$: total number of layers (excluding input layer)

### Parameter Shapes
| Parameter | Shape | Description |
|-----------|-------|--------------|
| $W^{[l]}$ | $(n^{[l]}, n^{[l-1]})$ | Weight matrix for layer $l$ |
| $b^{[l]}$ | $(n^{[l]}, 1)$ | Bias vector for layer $l$ |
| $\gamma^{[l]}$ | $(n^{[l]}, 1)$ | Batch norm scale parameter |
| $\beta^{[l]}$ | $(n^{[l]}, 1)$ | Batch norm shift parameter |

### Activation and Gradient Shapes
| Variable | Shape | Description |
|----------|-------|--------------|
| $X = A^{[0]}$ | $(n^{[0]}, m)$ | Input features |
| $A^{[l]}$ | $(n^{[l]}, m)$ | Activations after layer $l$ |
| $Z^{[l]}$ | $(n^{[l]}, m)$ | Linear outputs before activation |
| $Y$ | $(n^{[L]}, m)$ | True labels (one-hot for multi-class) |
| $dW^{[l]}$ | $(n^{[l]}, n^{[l-1]})$ | Gradient w.r.t. weights |
| $db^{[l]}$ | $(n^{[l]}, 1)$ | Gradient w.r.t. bias |
| $dA^{[l]}$ | $(n^{[l]}, m)$ | Gradient w.r.t. activations |
| $dZ^{[l]}$ | $(n^{[l]}, m)$ | Gradient w.r.t. linear outputs |

### Batch Normalization Shapes
| Variable | Shape | Description |
|----------|-------|--------------|
| $\mu^{[l]}$ | $(n^{[l]}, 1)$ | Batch mean |
| $\sigma^{2[l]}$ | $(n^{[l]}, 1)$ | Batch variance |
| $Z_{\text{norm}}^{[l]}$ | $(n^{[l]}, m)$ | Normalized values |
| $d\gamma^{[l]}$ | $(n^{[l]}, 1)$ | Gradient w.r.t. scale |
| $d\beta^{[l]}$ | $(n^{[l]}, 1)$ | Gradient w.r.t. shift |

## Forward Propagation

### Standard Layer (without Batch Normalization)

For layer $l$:

$$Z^{[l]} = W^{[l]} A^{[l-1]} + b^{[l]}$$
**Shapes:** $(n^{[l]}, m) = (n^{[l]}, n^{[l-1]}) \times (n^{[l-1]}, m) + (n^{[l]}, 1)$

$$A^{[l]} = g^{[l]}(Z^{[l]})$$
**Shapes:** $(n^{[l]}, m) = g((n^{[l]}, m))$

where:
- $W^{[l]}$ is the weight matrix for layer $l$ with shape $(n^{[l]}, n^{[l-1]})$
- $b^{[l]}$ is the bias vector for layer $l$ with shape $(n^{[l]}, 1)$
- $g^{[l]}$ is the activation function for layer $l$
- $A^{[0]} = X$ (input features) with shape $(n^{[0]}, m)$

### Layer with Batch Normalization

1. **Linear transformation:**
   $$Z_{\text{raw}}^{[l]} = W^{[l]} A^{[l-1]} + b^{[l]}$$
   **Shapes:** $(n^{[l]}, m) = (n^{[l]}, n^{[l-1]}) \times (n^{[l-1]}, m) + (n^{[l]}, 1)$

2. **Batch statistics (training mode):**
   $$\mu^{[l]} = \frac{1}{m} \sum_{i=1}^{m} Z_{\text{raw},i}^{[l]}$$
   **Shapes:** $(n^{[l]}, 1) = \text{mean}((n^{[l]}, m), \text{axis}=1)$
   
   $$\sigma^{2[l]} = \frac{1}{m} \sum_{i=1}^{m} (Z_{\text{raw},i}^{[l]} - \mu^{[l]})^2$$
   **Shapes:** $(n^{[l]}, 1) = \text{var}((n^{[l]}, m), \text{axis}=1)$

3. **Normalization:**
   $$Z_{\text{norm}}^{[l]} = \frac{Z_{\text{raw}}^{[l]} - \mu^{[l]}}{\sqrt{\sigma^{2[l]} + \epsilon}}$$
   **Shapes:** $(n^{[l]}, m) = \frac{(n^{[l]}, m) - (n^{[l]}, 1)}{\sqrt{(n^{[l]}, 1) + \text{scalar}}}$

4. **Scale and shift:**
   $$Z_{\text{tilde}}^{[l]} = \gamma^{[l]} Z_{\text{norm}}^{[l]} + \beta^{[l]}$$
   **Shapes:** $(n^{[l]}, m) = (n^{[l]}, 1) \odot (n^{[l]}, m) + (n^{[l]}, 1)$

5. **Activation:**
   $$A^{[l]} = g^{[l]}(Z_{\text{tilde}}^{[l]})$$
   **Shapes:** $(n^{[l]}, m) = g((n^{[l]}, m))$

6. **Running statistics update (training mode):**
   $$\text{running\_mean}^{[l]} = \beta_{\text{momentum}} \cdot \text{running\_mean}^{[l]} + (1 - \beta_{\text{momentum}}) \cdot \mu^{[l]}$$
   **Shapes:** $(n^{[l]}, 1) = \text{scalar} \times (n^{[l]}, 1) + \text{scalar} \times (n^{[l]}, 1)$
   
   $$\text{running\_var}^{[l]} = \beta_{\text{momentum}} \cdot \text{running\_var}^{[l]} + (1 - \beta_{\text{momentum}}) \cdot \sigma^{2[l]}$$
   **Shapes:** $(n^{[l]}, 1) = \text{scalar} \times (n^{[l]}, 1) + \text{scalar} \times (n^{[l]}, 1)$

### Inference Mode (Batch Normalization)

During inference, use running statistics:
1. **Normalization:**
   $$Z_{\text{norm}}^{[l]} = \frac{Z_{\text{raw}}^{[l]} - \text{running\_mean}^{[l]}}{\sqrt{\text{running\_var}^{[l]} + \epsilon}}$$
   **Shapes:** $(n^{[l]}, m) = \frac{(n^{[l]}, m) - (n^{[l]}, 1)}{\sqrt{(n^{[l]}, 1) + \text{scalar}}}$

2. **Scale and shift:**
   $$Z_{\text{tilde}}^{[l]} = \gamma^{[l]} Z_{\text{norm}}^{[l]} + \beta^{[l]}$$

### Dropout

Applied to hidden layers during training:
$$A^{[l]} = A^{[l]} \odot \frac{D^{[l]}}{\text{keep\_prob}}$$
**Shapes:** $(n^{[l]}, m) = (n^{[l]}, m) \odot \frac{(n^{[l]}, m)}{\text{scalar}}$

where $D^{[l]}$ is a binary mask with probability `keep_prob` of being 1.

## Backward Propagation

### Output Layer

For the final layer $L$:
$$dZ^{[L]} = A^{[L]} - Y$$
**Shapes:** $(n^{[L]}, m) = (n^{[L]}, m) - (n^{[L]}, m)$

### Hidden Layers (without Batch Normalization)

For layers $l = L-1, L-2, \ldots, 1$:

1. **Gradient w.r.t. activation:**
   $$dA^{[l]} = W^{[l+1]T} dZ^{[l+1]}$$
   **Shapes:** $(n^{[l]}, m) = (n^{[l]}, n^{[l+1]}) \times (n^{[l+1]}, m)$

2. **Apply dropout gradient:**
   $$dA^{[l]} = dA^{[l]} \odot D^{[l]}$$
   **Shapes:** $(n^{[l]}, m) = (n^{[l]}, m) \odot (n^{[l]}, m)$
   (if dropout was applied)

3. **Gradient w.r.t. pre-activation:**
   $$dZ^{[l]} = dA^{[l]} \odot g'^{[l]}(Z^{[l]})$$
   **Shapes:** $(n^{[l]}, m) = (n^{[l]}, m) \odot (n^{[l]}, m)$

### Hidden Layers (with Batch Normalization)

1. **Gradient w.r.t. activation:**
   $$dA^{[l]} = W^{[l+1]T} dZ^{[l+1]}$$
   **Shapes:** $(n^{[l]}, m) = (n^{[l]}, n^{[l+1]}) \times (n^{[l+1]}, m)$

2. **Apply dropout gradient:**
   $$dA^{[l]} = dA^{[l]} \odot D^{[l]}$$
   (if dropout was applied)
   **Shapes:** $(n^{[l]}, m) = (n^{[l]}, m) \odot (n^{[l]}, m)$

3. **Activation derivative:**
   $$dZ_{\text{tilde}}^{[l]} = dA^{[l]} \odot g'^{[l]}(Z_{\text{tilde}}^{[l]})$$
   **Shapes:** $(n^{[l]}, m) = (n^{[l]}, m) \odot (n^{[l]}, m)$

4. **Batch normalization gradients:**
   
   **Scale and bias parameter gradients:**
   $$d\gamma^{[l]} = \frac{1}{m} \sum_{i=1}^{m} dZ_{\text{tilde},i}^{[l]} \odot Z_{\text{norm},i}^{[l]}$$
   **Shapes:** $(n^{[l]}, 1) = \frac{1}{m} \sum_{\text{axis}=1} [(n^{[l]}, m) \odot (n^{[l]}, m)]$
   
   $$d\beta^{[l]} = \frac{1}{m} \sum_{i=1}^{m} dZ_{\text{tilde},i}^{[l]}$$
   **Shapes:** $(n^{[l]}, 1) = \frac{1}{m} \sum_{\text{axis}=1} (n^{[l]}, m)$

   **Gradient w.r.t. normalized input:**
   $$dZ_{\text{norm}}^{[l]} = dZ_{\text{tilde}}^{[l]} \odot \gamma^{[l]}$$
   **Shapes:** $(n^{[l]}, m) = (n^{[l]}, m) \odot (n^{[l]}, 1)$

   **Gradient w.r.t. variance:**
   $$d\sigma^{2[l]} = \sum_{i=1}^{m} dZ_{\text{norm},i}^{[l]} \odot (Z_{\text{raw},i}^{[l]} - \mu^{[l]}) \odot \left(-\frac{1}{2}\right) (\sigma^{2[l]} + \epsilon)^{-3/2}$$
   **Shapes:** $(n^{[l]}, 1) = \sum_{\text{axis}=1} [(n^{[l]}, m) \odot (n^{[l]}, m) \odot (n^{[l]}, 1)]$

   **Gradient w.r.t. mean:**
   $$d\mu^{[l]} = \sum_{i=1}^{m} dZ_{\text{norm},i}^{[l]} \odot \frac{-1}{\sqrt{\sigma^{2[l]} + \epsilon}} + d\sigma^{2[l]} \frac{-2}{m} \sum_{i=1}^{m} (Z_{\text{raw},i}^{[l]} - \mu^{[l]})$$
   **Shapes:** $(n^{[l]}, 1) = \sum_{\text{axis}=1} [(n^{[l]}, m) \odot (n^{[l]}, 1)] + (n^{[l]}, 1) \odot (n^{[l]}, 1)$

   **Final gradient w.r.t. input:**
   $$dZ_{\text{raw}}^{[l]} = dZ_{\text{norm}}^{[l]} \odot \frac{1}{\sqrt{\sigma^{2[l]} + \epsilon}} + d\sigma^{2[l]} \odot \frac{2(Z_{\text{raw}}^{[l]} - \mu^{[l]})}{m} + \frac{d\mu^{[l]}}{m}$$
   **Shapes:** $(n^{[l]}, m) = (n^{[l]}, m) \odot (n^{[l]}, 1) + (n^{[l]}, 1) \odot (n^{[l]}, m) + (n^{[l]}, 1)$

### Weight and Bias Gradients

For all layers:
$$dW^{[l]} = \frac{1}{m} dZ^{[l]} A^{[l-1]T} + \frac{\lambda}{m} W^{[l]}$$
**Shapes:** $(n^{[l]}, n^{[l-1]}) = \frac{1}{m} (n^{[l]}, m) \times (m, n^{[l-1]}) + (n^{[l]}, n^{[l-1]})$

$$db^{[l]} = \frac{1}{m} \sum_{i=1}^{m} dZ_i^{[l]}$$
**Shapes:** $(n^{[l]}, 1) = \frac{1}{m} \sum_{\text{axis}=1} (n^{[l]}, m)$

where $\lambda$ is the L2 regularization parameter.

## Activation Functions

### ReLU
$$g(z) = \max(0, z)$$
$$g'(z) = \begin{cases} 1 & \text{if } z > 0 \\ 0 & \text{if } z \leq 0 \end{cases}$$

### Leaky ReLU
$$g(z) = \begin{cases} z & \text{if } z > 0 \\ \alpha z & \text{if } z \leq 0 \end{cases}$$
$$g'(z) = \begin{cases} 1 & \text{if } z > 0 \\ \alpha & \text{if } z \leq 0 \end{cases}$$

where $\alpha$ is the leaky constant (default: 0.01).

### Sigmoid
$$g(z) = \frac{1}{1 + e^{-z}}$$
$$g'(z) = g(z)(1 - g(z))$$

### Softmax
For multi-class classification:
$$g_i(z) = \frac{e^{z_i - \max(z)}}{\sum_{j=1}^{C} e^{z_j - \max(z)}}$$

where $C$ is the number of classes and we subtract $\max(z)$ for numerical stability.

## Loss Functions

### Binary Cross-Entropy (Log Loss)
$$J = -\frac{1}{m} \sum_{i=1}^{m} \left[ y^{(i)} \log(a^{(i)}) + (1-y^{(i)}) \log(1-a^{(i)}) \right]$$

### Categorical Cross-Entropy (Multi-class)
$$J = -\frac{1}{m} \sum_{i=1}^{m} \sum_{j=1}^{C} y_j^{(i)} \log(a_j^{(i)})$$

### L2 Regularization
Added to both loss functions:
$$J_{\text{reg}} = J + \frac{\lambda}{2m} \sum_{l=1}^{L} \|W^{[l]}\|_F^2$$

where $\|W^{[l]}\|_F^2$ is the Frobenius norm squared.

## Optimization Algorithms

### Gradient Descent
$$W^{[l]} := W^{[l]} - \alpha \cdot dW^{[l]}$$
$$b^{[l]} := b^{[l]} - \alpha \cdot db^{[l]}$$

### Momentum
$$v_{dW}^{[l]} = \beta v_{dW}^{[l]} + (1-\beta) dW^{[l]}$$
$$v_{db}^{[l]} = \beta v_{db}^{[l]} + (1-\beta) db^{[l]}$$

$$W^{[l]} := W^{[l]} - \alpha v_{dW}^{[l]}$$
$$b^{[l]} := b^{[l]} - \alpha v_{db}^{[l]}$$

### RMSProp
$$s_{dW}^{[l]} = \beta s_{dW}^{[l]} + (1-\beta) (dW^{[l]})^2$$
$$s_{db}^{[l]} = \beta s_{db}^{[l]} + (1-\beta) (db^{[l]})^2$$

$$W^{[l]} := W^{[l]} - \frac{\alpha}{\sqrt{s_{dW}^{[l]} + \epsilon}} dW^{[l]}$$
$$b^{[l]} := b^{[l]} - \frac{\alpha}{\sqrt{s_{db}^{[l]} + \epsilon}} db^{[l]}$$

### ADAM
$$v_{dW}^{[l]} = \beta_1 v_{dW}^{[l]} + (1-\beta_1) dW^{[l]}$$
$$s_{dW}^{[l]} = \beta_2 s_{dW}^{[l]} + (1-\beta_2) (dW^{[l]})^2$$

**Bias correction:**

$${v_{dW}^{[l]}}(\text{corrected}) = \frac{v_{dW}^{[l]}}{1-\beta_1^t}$$
$${s_{dW}^{[l]}}(\text{corrected}) = \frac{s_{dW}^{[l]}}{1-\beta_2^t}$$

**Parameter update:**
$$W^{[l]} := W^{[l]} - \frac{\alpha}{\sqrt{s_{dW}^{[l]}(\text{corrected}) + \epsilon}} v_{dW}^{[l]}(\text{corrected})$$

### Learning Rate Decay
$$\alpha_t = \frac{\alpha_0}{1 + \text{decay\_rate} \cdot t}$$

where $t$ is the training iteration number.

## Weight Initialization

### He Initialization (for ReLU)
$$W^{[l]} \sim \mathcal{N}\left(0, \sqrt{\frac{2}{n^{[l-1]}}}\right)$$

where $n^{[l-1]}$ is the number of units in layer $l-1$.

## Notation

- $m$: batch size (number of training examples)
- $L$: number of layers
- $n^{[l]}$: number of units in layer $l$
- $\alpha$: learning rate
- $\lambda$: regularization parameter
- $\epsilon$: small constant to avoid division by zero (typically $10^{-8}$)
- $\beta$, $\beta_1$, $\beta_2$: momentum parameters
- $\gamma^{[l]}$, $\beta^{[l]}$: batch normalization scale and shift parameters
