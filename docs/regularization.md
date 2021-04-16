# Regularization

Overfiting is one of the most common prediction problems in deep learning systems. An overfitted model fits well for the training sequence, but shows bad performance for validation data not seen in the training set. This behavior is a result of fitting to the model to the details of a particular set of training data. Reasons for overfitting may be a too complex prediction model (e.g. too many nodes or layers), and a too short training sequence. 

Figure 1 illustrates underfitting rightfitting and overfitting of a 1D model. Figure 1a shows underfitting. Common reason for underfitting is a too simple model. Anyway, this post is about a set of algorithms used to avoid overfitting, named Regularization.

Figure 1a: Underfitting

![](../assets/images/regularization/underfitting.png)
 
Figure 1b: Goodfitting

![](../assets/images/regularization/rightfitting.png)
  
Figure 1c: Overfitting

![](../assets/images/regularization/overfitting.png)



Regularization

As mentioned above, Regularization applies to a set of algorithms which aim to prevent overfitting, by either simplify the system or increase the size of training system. The regularization algorithms presented here are:

- L1 and L2 Regularization
- Dropout
- Early Stopping
- Data Augmentation

- L1 and L2 Regularization

L1 and L2 Regularizations are both similar methods, which aim to prevent overfitting by effectively simplify the system. We'll see that.

In L1 and L2 Regularizations, the cost function is incremented by a regularization coefficient as shown by Eq. 1 and Eq. 2. BTW, to be more precise, L2 Regularization is actually L2 Squared Regularization.

### Eq. 1: Norm Regularization

#### Eq. 1a: L2 Regularization

![](../assets/images/regularization/l2-regularization.svg)

#### Eq. 1b: L1 Regularization

![](../assets/images/regularization/l1-regularization.svg)


The regularaizd cost function effects the Gradient Descent formula for weights coefficents iterative calculation:

\\(\mathbf{w=w-\alpha \cdot\frac{\partial C}{\partial w}}\\)

Starting with L2 Regularization, let's plug the L2 regulated cost function to the Gradient Descent formula and get:

![](../assets/images/regularization/l2-gradient-descent-1.svg)


Let's calculate the gradient of the L2 norm. Noticing that:

![](../assets/images/regularization/l2-derivative.svg)


The gradient is accordingly:

![](../assets/images/regularization/l2-derivative.svg)


Plugging the gradient back to the Gradient Descent equation we get:

\\(\mathbf{w=w-\alpha (\cdot \frac{\partial C}{\partial w} + \lambda \frac{1}{2} \triangledown_w \left \| w \right \|_2^2)=w(1-\lambda)-\alpha \frac{\partial d }{\partial w}C}\\)

![](../assets/images/regularization/l2-gradient-descent.svg)


So we reached the formula, as expressed in Eq. 2:

### Eq. 2: Gradient Descent with L2 Regularization

![](../assets/images/regularization/l2-gradient.svg)



Now let's plug the L1 regulated cost function to the Gradient Descent formula and get:


![](../assets/images/regularization/l1-gradient.svg)


Noticing that the gradient of the L1 norm is:

![](../assets/images/regularization/l1-derivative.svg)

Which can be expressed as the weights signs:


![](../assets/images/regularization/l1-gradient-sign.svg)


Plugging the gradient back to the Gradient Descent equation we get:

![](../assets/images/regularization/l1-gradient-descent.svg)


So we reached the formula, as expressed in Eq. 3:

### Eq. 3: Gradient Descent with L1 Regularization

\\(\mathbf{\mathbf{w=w-\alpha \cdot \lambda \cdot sign(w)-\alpha \cdot \frac{\partial d }{\partial w}C}}\\)




