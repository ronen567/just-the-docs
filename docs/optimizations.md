---
layout: default
title: Optimization Algorithms
nav_order: 4
---

## Introduction


**Gradient Descent** is a recursive algorithm, which finds the minimum of a function. The minimum is located by striding in the oposite direction of the function's gradient, with a step size according to the gradient size, as expressed in Eq. 1.

## Eq. 1: Gradient Descent Equation

\\(X(t) = X(t-1)-\alpha \cdot \bigtriangledown f(X)) \\)
**Where**:
**t** is the iteration index
\\(X = {x_j} j =1:n\\) the n dimensional parameter searched for.
\\(\alpha\\) is constant known as the learning rate.

The recursive algorithm goes as follows:

0. Init \\(X={x_j}\\), j=1:n to an arbitrary value. (A bad values selection may delay convergance. Check if so by trial and error.)
1. Calculate gradient of f(X)
2. Calculate new X(t) using Eq. 1
3. Continue to step 1 if \\(\bigtriangledown f(X) > \epsilon \\)


**Gradient Descent in Deep Neural Networks*** 
In the context Deep Neural Networks (DNNs), Gradient Descent is the most popular optimization algorithm, used to find the optimized set of weight coefficients which minimizes the cost function J, (i.e. a function which expresses the error between the expected DNN output and the model's predicted output). Figure 2 depicts a schematic diagram of DNNs Training phase functionality. 

![Training-Phase](../assets/images/gd_optimizations/Training-Phase.png)



**Illustrative Examples**

Figure 1 illustrates gradient descent convergence for a single variable function \\(f(x) = (x-w_1)^2\\). In this single variable example, the gradient degenerates to:
\\(x_{t} = x_{t-1}-\alpha \cdot \frac{d}{dx}f(x) \\)

![gradient decent example](../assets/images/gd_optimizations/sgd_1d_intro.gif)

Figure 2 is a contour graph which illustrates gradient descent convergence for a 2 variable function of the type \\(f(x) = a \cdot (x-w_1)^2 + b \cdot (x-w_1)^2\\)

**Figure 2: Gradient Descent - Easy Convergence

![gradient decent example](../assets/images/gd_optimizations/2d.gif)


Now look at Figure 3, which is similar to Figure 2, except that the gradient is steepper in w2 direction. The resultant step size in the w2 direction at the begining is too large, which leads to some overshoots, but the algorithm converges eventually.

**Figure 3: Gradient Descent - Slight Oscilations

![gradient decent example](../assets/images/gd_optimizations/2d_contour_sgd_asymetric.gif) 

Figure 4 however, presents an even steepper gradient in w2 direction. Now we get oscilations in w2 direction, which never converge. 

**Figure 4: Gradient Descent - Oscilations
![gradient decent example](../assets/images/gd_optimizations/2d_contour_sgd_oscilations.gif)

A smaller learning rate would solve this, make the algorithm smoothly converge. A smaller learning rate would have slowed down convergance in all scenarios. This is a tradeoff. Chosing alpha is one of the chalenges of Gradient Descent, and it normally requires some trial and error iterations to find a suitable value. Overshoot is a one of the Gradient Descent performance issues, between them are:
- ***Overshooting***: As depicted by Figures 3 and 4, high curvutures may lead to overshooting. Overshooting is a result of moving too fast along the gradient direction, while it changes signs. 

- ***Local Minimun trap***: Getting trapped in a local minimum, not reaching the global minima.

- ***Oscillations***: this phenomenon can occure not only when gradient changes significantly in high curvuturs as depicted by Figure 4, but also when no matter the direction it  navigating in a plateau, where gradient is negligible but still may have slight differences which lead to oscliations

This post reviews some of the most popular Gradient Descent variants which aim to answer Gradient Descent issues.

This algorithms reviewd in this post are:

**Momentum**
**Nesterov momentum**
**Adagrad**
**Adadelta**
**RMSprop**

**Adam**
**Adamax**
**FTR**
**NAdam**

The next paragraphs describe the principles of the various Gradient Descent algoritms. Following that are graphical illustrations of these algorithms when applied on a 3 "cost functions" presented above, i.e. "easy convergence, "slightly oscilated" and "oscilated".

## Momentum

A smaller learning rate coefficient at gradient change of sign regions, could improve Gradient Descent's performance in overshoot and oscilation scenarios. In other scenarios, where gradient's direction does not change, a larger learning rate would have increase convergance rate. The Momentum algorithm aims to achieve that by adding another term to the Gradient Descent correction factor, denoted by \\(v \\). Here's the momentum formula:

## Eq. 2: Momentum

## Eq. 2a
\\( v =\beta \cdot v - \alpha \cdot \bigtriangledown_w f(w) \\)
## Eq. 2b
\\( w = w+v \\)

The hyperparameter values are \\( \alpha \epsilon(0,1) \\) and \\( \beta \\) is typically 0.9.
As Eq. 2 shows, the updated value w, is dependent not only on the recent gradient, but also on \\(v\\), an exponentially decaying moving average of past gradients.
So, update step size will be increased if sign of \\( v \\) is same as the current gradient's, and decreased otherwise, i.e. when gradient has changed direction with regard to averaged gradient direction.

The reason for naming it momentum, is the analogy to Newtonian motion model: \\(v(t) = v(t-1) + a \cdot \Delta T,\;Delta T=1\\), where the velocity \\(v \\) at time t, equals to the sum of velocity at \\( t-1 )\\ and accelaration term. In Eq 2, the averaged step size is analogous to velocity,while the gradient is analogous to the acceleration. In the Newtonian phisics (mechanics),the momentum is the product of velocity and mass (denoted by m), so assume m=1.

**Nesterov momentum**

Nesterov momentum algorithm is a variation of the momentum, but with a slight difference: rather than \\(\beta(w(t)-w(t-1)) \\), it is now \\(\beta(w(t+1)-w(t)) \\), i.e. it uses the Gradeint Descent value calculated at (t+1). Accordingly, the new value is calculated in 2 steps:

## Eq. 3: Nesterov momentum
### 3.a
\\(v(t+1)=/beta \cdot v - \alpha \cdot \bigtriangledown f(w+\beta v(t)) \\)
### 3.b
\\(w=w+ \beta \cdot v + v(t+1) \\)


**Adagrad**
"Adaptive Subgradient Methods for Onlie Learning and Stochastic Optimization, Journal Of Machine Learning Reaserch 12 (2011), Duchi et. al.

The name Adagrad derived from **Adaptive Gradient** Algorithm. The idea is to modify the learning rate, based on past gradients. Still, a "Global Learning Rate" value should be selected. The Gradient Descent step update formula now becomes :

### Eq. 4: Adagrad

\\(w_{t+1}=w_t-\frac{\alpha}{\epsilon + \sqrt{G_{t}} \odot g(t) \\)

Where:
\\(\alpha \\) is the "Global Learning Rate".
\\(g(t)=\bigtriangledown_w J(w_t) \\)
-\\(odot\\) stands for "elementwise multiplication".
-G_{t} is a diagonal matrix, where the (i,i) element is the square of the ith gradient of f(w), i.e. \\(\bigtriangledown_w_i f(w) \\). 
-\\(G_{t,(i,i)}=\sum_{}^{t} \bigtriangledown_w_{i} f(w)) \\)
-\\(\epsilon \\) is a small value used to maintain stability, commonly set to \\(10^{-7} \\).

Adagrad gives lower learning rates to parameters with higher gradients, and higher rates, thus faster convergance to smoth pathes. Still, since Avagard accumulates squared gradients from the begining of the training, the the adaptive learning rate coefficient can excessively decrease as the training continues.


**AdaDelta**

ADADELTA: An Adaptive Learning Rate Method, Zeiler

AdaDelta's idea was derived from AdaGrad, which parameter updating term is (See Eq. 4):

\\(\Delta{w_{t}}}=-\frac{\alpha}{\epsilon + \sqrt{G_{t}} \odot g(t) \\)

Where
\\(g_t = \bigtriangledown f(w_t) \\)

AdaDelta aims to improve the 2 drawbacks of that updating term: 1. the continual decay of learning rate. 2. the need to select a global learning rate.

To improve the first drawback, Avagard's denominator is replaced by an exponentially decaying average of squared gradients \\(E(g^2) \\) :

\\(E(g^2)_t=\gamma E(g^2)_{t-1}+(1-\gamma)g^2_{t} \\)

where \\(\gamma\\) is a constant controlling the decay of the gradients average.
and \\(g^2_{t} = g_{t} \odot g_{t} \\) , i.e. an elementwise square. 

The term required in the denominator is the square root of this quantity, which is denoted as the RMS (Root 
Square) of previous squared gradients, up to time t, so:

\\(RMS(g^2)_t=\sqrt {E(g^2)_{t} + \epsilon} \\)

Where \\(\epsilon}\\) is a small value used to maintain stability, commonly set to \\(10^{-7} \\). So that's for improving the decaying learning rate issue.


To improve the second drawback, i.e. avoid the need to determine a global learning rate, the numerator is taken as an exponentially decaying average of the past parameters updates:

\\(E(\Delta{w}^2)_{t-1}=\gamma E(g^2)_{t-2}+(1-\gamma)\Delta{w^2}_{t-1} \\)

And same as with the denominator, the square root of the avarage is taken for the numerator:

\\(RMS(\Delta{w}^2)_{t-1}=\sqrt {E(\Delta{w}^2)_{t-1} + \epsilon} \\)

Integrating all the components the updating term formula becomes:

\\(\Delta{w}_t=-\frac{RMS(\Delta{w}^2)_{t-1}}{RMS(g^2)_t}\cdot\bigtriangledown f(w_t) \\)


Finally we have it all:

### Eq. 5: AdaDelta

\\(w_{t+1}=w_t-\Delta w_t = w_t-\frac{RMS(\Delta{w}^2)_{t-1}}{RMS(g^2)_t} \cdot g_t \\)

Where
\\(g_t = \bigtriangledown f(w_t) \\)
and \\(g^2_{t} = g_{t} \odot g_{t} \\) , i.e. an elementwise square. 


**RMSprop**

RMSprop was presented in a Coursera course lecture.

The name RMSprop derived from RMS (**R**oot **M**ean **S**quare) + **Prop**agation.
Like AdaDelta, RMSprop is an improvement of AdaGrad. It aims to solve AdaGrad drwaback regarding the continual decay of learning rate. It does so by replacing the denominator of AdaGrad (Eq. 4), by an exponentially decaying average of squared gradients \\(E(g^2) \\), exactly as done by AdaDelta. Unlike AdaDelta, RMSprop leaves AdaGrad's global learning rate coefficient in place, so the updating formula becomes:

### Eq. 6: RMSprop

\\(w_{t+1}= w_t-\frac{\alpha}{RMS(g^2)_t}\cdot g_t \\)

Where
\\(g_t = \bigtriangledown f(w_t) \\)
and \\(g^2_{t} = g_{t} \odot g_{t} \\) , i.e. an elementwise square. 


Recommended values for the global learning rate \\(\alpha \\) and the decay constant \\(\gamma \\) hyperparameters are 0.001 and 0.9 respectively.



## Adam

ADAM: A METHOD FOR STOCHASTIC OPTIMIZATION, ICLR 2015, Kingma and Ba

The name Adam derived from Adaptive Moment Estimation. The algorithm was designed to combine the advantages of AdaGrad and RmsProp. It incorporates exponential decay moving averages of both past gradients, aka moment (aka first raw moment), denoted by \\m_t(\\) , and of squared gradients, (aka second raw moment or uncentered variance), denoted  \\v_t \\) . Adam also incorporates initialization bias correction, to compensate the moments' bias to zero at early iterations. Let's see all that.

Here is the moment estimate at time t, calculated as an exponantial decay moving average of past gradients.

### Eq. 7.a: moment estimate 

\\(m_t=\beta_1 \cdot m_{t-1} + (1-\beta_1) \cdot g_t \\)

where:
\\( g_t = \bigtriangledown f(w_t) \\)
\\(\beta_1 \epsilon [0,1) \\) is a hyper parameter which controls the exponential decay rate of the moving average.

Here's the second raw moment estimate at time t, calculated as an exponantial decay moving average of past squared gradients.

### Eq. 7.b: Second raw moment estimate

\\(v_t=\beta_2 \cdot v_{t-1} + (1-\beta_2) \cdot g_t^2 \\)

\\( g_t = g_t \odot g_t\\) is an elementwise square.
\\(\beta_2 \epsilon \left [0,1 \right ) \\) is a hyper parameter which controls the exponential decay rate of the moving average.

Now let's consider the initialization bias issue and its correction: moving averages \((m_t\)) and \((v_t\)) vectors are initialized to all 0s. That leads Eq.7 and Eq. 8 to bias towards zero, especially in the first iteration steps. To compensate that, Adam sets bias correction to the first and second raw  moments and second raw moment estimates.

### Eq. 7.c: Bias corrected moment estimate 

\\(\hat{m}_t=\frac{m_t}{1-\beta_1^t} \\)

Where:
With \\(\beta_1^t\\) we denote \\(\beta_1\\) to the power of t 

\\(1-\beta_1^t} \\) is small for small values of t, which leads to increasing \\(\hat{v}_t \\) value for initial steps.

### Eq. 7.d Bias corrected second raw moment estimate 

\\(\hat{v}_t=\frac{v_t}{1-\beta_2^t} \\)

Where:

With \\(\beta_2^t\\) we denote \\(\beta_2\\) to the power of t 

\\(1-\beta_2^t} \\) is small for small values of t, which leads to increasing \\(\hat{v}_t \\) value for initial steps.


Finally Adam's update forula is:


### Eq. 7.e: Adam  

\\(w_{t+1}= w_t-\frac{\alpha \cdot \hat{m}_t}{\sqrt(\hat{v}_t)+\epsilon} \\)

Where proposed hyperparameter values are:
\\(\alpha=0.001 \\)

\\(\beta_1=0.9 \\)

\\(\beta_2=0.999 \\)

\\(\epsilon=10^{-8} \\)


#### Inspection  of updates bounderies and behavior.



Notes on the Algorithm  Update Rule
 Let's examine the bounderies of update step size.
 
 



## Adamax

ADAM: A METHOD FOR STOCHASTIC OPTIMIZATION, ICLR 2015, Kingma and Ba


A variant of Adam, proposed in same paper, suggests to replace \\( v_t \\) , the second raw moment, by \\( u_t \\):

### Eq. 8.a: Adam  

\\( u_t = max(\beta_2 \cdot  u_{t-1}, \left | g_t \right | \\)

Bias correction is not needed for \\( u_t \\) anymore. The numerator is same as in Adam. Consequently, the Aadamax update formula is:

### Eq. 8.b: Adamax

\\( w_{t+1}= w_t-\frac{\alpha \cdot {m}_t}{(1-\beta_1^t) \cdot u_t} \\)


Where proposed hyperparameter values are:

\\(\alpha=0.002 \\)

\\(\beta_1=0.9 \\)

\\(\beta_2=0.999 \\)








coursera:
Mini Batch


Training with batch of all m examples - 1 update (step of gradient descent) after each cycle
Mini Batch - 1000 examples...runs faster for big training set


Referances:
- Deep Learning by Ian Goodfellow, Yoshua Bengio, and Aaron Courville 
