---
layout: default
title: Batch and Mini Batch
nav_order: 11
---

The  fitting algorithm executed during the Training phase is most commonly carried by the Gradient Descent optimization algorithm and its variations, presented in previous posts.

Here's the basic formula of Gradient Descent:

### Eq. 1 Gradient Descent

\\(w=w-\alpha \cdot \frac{\partial L(w)}{\partial w}\\)

\\( \texttt{Where } L(w) \texttt{ is a Loss function which expresses the prediction's accuracy calculated for a every Training data example }\\)

Eq. 1 expresses a single iteration of the optimization algorithm, at which the Gradient Descent update is calculated per each Training data example. This Gradient Descent variant, which runs and update iteration per a single training example is caleed **Stochastic Gradient Descent** abrivated to SGD. Why Stochastic? Because the gradient is calculated per a single sample and not for the entire m examples of the data set. As a result of that, the gradient convergence iterative process is expected to be noisy. Still, SGD proves to fastly converge. However that, current High Processing Computing (HPC) devices, e.g. GPUs, are tailored for vectorized computations. Otherwise they are not vey efficient. As an alternative to SGD, Batch algoritm, AKA Deterministic Methods (as oposed to Stochastic)  are more efficient with vectorized computation machines. When speaking of Batch methods we distinct between the **Batch** methods, at which the training process and gradient is computed over the entire set of training examples, and **Minibatch** which batch size is a fraction of the entire training set. Minibatch is a compromise solution for the tradeoff between the SGD faster updates and Batch algorithms accurate gradient estimaton. Normally, selected batch size would be a power of 2 number in the range of 32-512, 

The 


**Batch** computation relates to running the Gradient Descent update once for the entire m training examples. Accordingly, Eq. 1 changes to Eq. 2, where the gradient with respect to w is taken over the Cost function, denoted by J(w), which is an average of the loss function, as expressed in Eq. 3.

Some considerations on chosing batch size:
- The processors optimized vector size


### Eq. 2: Batch Gradient Descent
\\(w=w-\alpha \cdot \frac{\partial J(w)}{\partial w}\\)

\\(\texttt{Where } J(w) \texttt{ is a Cost function which expresses the averaged prediction's accuracy over the entire tarining m samples batch}\\)

### Eq. 3: Cost Function

\\(J(w)=\frac{1}{m}\sum_{i=1}^{m}L(w)
\\)

So, it is o
















Loss function is replaced by a cost function, which is the 

are Accordingly, a vectorized computation is commonly preffered. dcomputation lacking the vectorized-block computation
As an alternative for SGD, 

coefficient value w is updated per each Training data example: The  Loss function is calculated  gradient with respect to w, multiplied by the learning rate \\(\alpha\\). The Loss function is calcu




