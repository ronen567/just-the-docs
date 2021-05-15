---
layout: default
title: Batch and Mini Batch
nav_order: 11
---
# Batch and Minibatch

The  fitting algorithm executed during the Training phase is most commonly carried by the Gradient Descent optimization algorithm and its variations, presented in previous posts.


## Stochastic Gradient Descent

Here's the basic formula of Gradient Descent:

### Eq. 1 Gradient Descent

\\(w=w-\alpha \cdot \frac{\partial L(w)}{\partial w}\\)

Where \\(L(w)\\) is a Loss function, which expresses the prediction's accuracy calculated for a every Training data example.

Eq. 1 expresses a single iteration of the optimization algorithm, at which the Gradient Descent update is calculated per each Training data example. This Gradient Descent variant, which runs and update iteration per a single training example is caleed **Stochastic Gradient Descent** abrivated to **SGD**. Why Stochastic? Because the gradient is calculated per a single sample and not for the entire m examples of the data set. As a result of that, the gradient convergence iterative process is expected to be noisy. Still, SGD proves to fastly converge. However that, current High Processing Computing (HPC) devices, e.g. GPUs, are tailored for vectorized computations. Otherwise they are not vey efficient. As an alternative to SGD, Batch algoritm, AKA Deterministic Methods (as oposed to Stochastic) are more efficient with vectorized computation machines. 

## Batch and Minibatch Gradient Desent Computation

When speaking about Batch methods, there's a distinction between the **Batch** and **Minibatch** methods.
**Batch** computation relates to running the Gradient Descent update once for the entire m training examples. Accordingly, Eq. 1 changes to Eq. 2, where the gradient with respect to w is taken over the Cost function, denoted by J(w), which is an average of the loss function, as expressed in Eq. 3.

### Eq. 2: Batch Gradient Descent
\\(w=w-\alpha \cdot \frac{\partial J(w)}{\partial w}\\)

Where \\(J(w)\\) is a Cost function which expresses the averaged prediction's accuracy over the entire tarining m samples batch.

### Eq. 3: Cost Function

\\(J(w)=\frac{1}{m}\sum_{i=1}^{m}L(w)
\\)

The gradient claculated over the cost function rather than the loss function give is more accurate, less volunarable to noise. Still, the updates are slow, taken once every calculations of m samples. 

**Minibatch** is a compromise solution for the tradeoff between the SGD faster updates, and Batch algorithms accurate gradient estimaton. Normally, selected batch size would be a power of 2 number in the range of 32-512, which fits vectorized processors.


Some considerations on chosing batch size (:
- Larger batches provide a mode accurate estimate of the gradient but with less than a linear return (Goodfellow et al, ch. 8). Let's explain that: Generaly speaking, Standard error gives the accuracy of a variable's mean, by measuring it's variability. Eq. 4 shows the standard error of the Cost function, which is, by definition, the average of m loss functions.


-The standard error of the mean loss, calculated over m loss values. is not decreased by a factor of \\(\frac{1}{m}\\) wrt the error of a none-batched sample, but only by a factor of \\(\frac{1}{\sqrt{m}}\\).

so accordingly, a gradient based on averaging the loss of 100 samples requires 100 times more computations, but the standard error reduces by a factor of 10 only. 
Let's see that:

### Eq. 4: Standard error of the mean loss

\\(SE(\hat{\mu}_m )=\sqrt{var[\frac{1}{m}\sum_{i=1}^{m}L^{(i)}]}=\frac{1}{m} \cdot \sqrt{m} \sigma=\frac{1}{\sqrt{m}} \cdot \sigma\\)

- Multicore architecture, e.g. GPUs, are more underutilized by extremely small batches. This motivatess using some minimum batch size, otherwise there is no much gain in processing minibatches wrt SGD.
- Amount of required memory grows with batch size, so batch size is limitted by memory requirements.
- Some Hardware vectorized processors achieve better performance gain with power of 2 batch sizes.
- Small batches can offer regularization effect, perheps due to noise they add to the learning process.









