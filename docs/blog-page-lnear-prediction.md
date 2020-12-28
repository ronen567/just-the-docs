## Linear Regression and Gradient Descent


So, here’s the problem:
Context is supervised learning. In the  training phase, we use a set of labeled input data samples,  build a model function h(x), which should be used to predict the output based on features i.e.

\[\hat{y}=h(x)\]

Where \[\hat{y}, x\] are the estimated value of y and the feature data respectively.

Let’s illustrate that in an example:
Table 1 lists 3 features according to which house prices should be predicted. (Note - obviously those 3 features are not enough to predict a house prices, but let’s keep it like so for the sake of example’s simplicity).  The training set consists of 4 examples, each  labeled with the known corresponding price



Table 3:  House price prediction - Labeled data

\[x1\]
Number of bedrooms
$$x_{2}$$
Zip code
$$x_{3}$$
floor
$$y$$
Price $
4
32321
2
17900
2
54322
3
21000
2$$x_{1}$$
43243
4
25000
3
63422
1
26000


Following, is table 2, which consists of regular data, for which the price should be estimated, based on features.


Table 4: How Price Prediction - Regular Data


\[x1\]
Number of bedrooms
$$x_{2}$$
Zip code
$$x_{3}$$
floor
$$y$$
Price $
3
34321
3
?
5
64322
5
?
1$$x_{1}$$
23241
4
?
4
83422
2
?



So, our challenge is to do the following:
Determine a prediction model
Train the model with the labeled set of samples, so that it can later can estimate tabel 4’s data set.


Let’s start with the first challenge, and determine the model. In this post, for the sake of simplicity, we chose the Linear Regression model. Note that the most common prediction model used is current algorithms is the Logistic Regression. We will get to it in  later posts. But Linear Regression is a good point to start with in order to get a  foundation for the other algorithms. Still, you may skip this post and jump directly to Logistic Regression.

So, Linear Regression is a model which approximates y, as a linear function of the input x, as expressed in Eq 1.

Eq. 1: Linear Prediction 

\[\hat{y}=b+\sum_{j=1}^{n}w_jx_j\]


Now when the model is selected, the challenge is to train it. The goal of the training is to determine the set of predictor's coefficients \[b, w_j, j=1:n\], where in our example n=3.
Note that I deliberately keep the notation {b}, rather then {w_0} for the bias coefficient, i.e the coefficient which does not multiply a feature sample. I plan to keep that notation from now on, as it is common in many papers and books.
Rest of this post is presents the mathematical development of the predictor’s coefficients formula.
















Eq 1 expresses  an n dimensional input model. A multi dimensional problem is hard to sketch graphically, not to say impossible. For the sake of simplicity, and without losing generality, let’s reduce the dimensionality of features to 1. Now it will be easier to sketch and illustrate the problem and the solution.


So now, Eq. 1 reduces to a 1D linear prediction:

\[ \hat{y}^i ={ b^i}+w_1{x^i}\]

Figure 1 presents a line approximation to scattered points


Figure: Line Approximation 




Now we need to find the predictor’s coefficients \[b, w_1\] which minimize the cost function:

The cost function is chosen to be the some of the squared errors:

\[J(w,b)=\frac{1}{m}\sum_{i=1}^{m}(\hat{y}^i-y^i)^2\]

BTW, optionally other cost functions could be chosen, e.g.

\[J(w,b)=\frac{1}{m}\sum_{i=1}^{m}\left | \hat{y}^i-y^i \right |\]

But by summing the quadric errors, the cost  increases more, the larger error is.


In  our reduced dimensions case, the is as a quadratic equation with 2 unknown parameters: \[b\] and \[Jw_1\]:

Eq. 6:
\[J(w,b)=\frac{1}{m}\sum_{i=1}^{m}(b^i+w_1x^i-y^i)^2\]


Plotting Eq 6, will result in a surface, such as  illustrated in figure 7:

Figure 7:  Plot illustration of  \[J(b,w)\]::\[J(b,w)\]::

![Figure 7](./assets/images/assets/images/linear_prediction_approximation surface.png)


So we are looking for (b, w1) which minimizes the cost function. If you studied calculus sometimes in your life, you probably know that the minima of J(b, w1) is the point were the partial derivatives  wrt b and w1 are 0. And if you don’t know that, never mind, you may just believe it’s so.









So let’s derivate:
\[\frac{\partial J(b,w)}{\partial  b}=\frac{\partial(\frac{1}{2m}\sum_{i=1}^{m}(b+w_1x^i-y^i)^2)}{\partial  b}\]

\[=\frac{1}{2m}*\sum_{i=1}^{m}\frac{\partial(b+w_1x^i-y^i)^2}{\partial  b}\]
\[= \frac{1}{m}\sum_{i=1}^{m}(b+w_1x^i-y^i)= \frac{1}{m}\sum_{i=1}^{m}(h(x^i)-y^i)\]


\[\frac{\partial J(b,w)}{\partial  w1}=\frac{\partial(\frac{1}{2m}\sum_{i=1}^{m}(b+w_1x^i-y^i)^2)}{\partial  w1}\]

\[=\frac{1}{2m}*\sum_{i=1}^{m}\frac{\partial(b+w_1x^i-y^i)^2}{\partial  w1}\]
\[= \frac{1}{m}\sum_{i=1}^{m}(b+w_1x^i-y^i)*x^i= \frac{1}{m}\sum_{i=1}^{m}(h(x^i)-y^i)*x^i\]







the point where \[J(b,w)\] is minimal
To find the cTo minima of the cost function, is the point where derivative of order one is zero. 



dimensions, the cost function equals to the sum of squared distances between the example points labels, aka training data points, to the approximated values  $$en the example points labels, aka training data points, to the approximated values  $$$$ the example points labels, aka training data points, to the approximated values  $$values  $$$$lues  $$\hat{y^i}.$$

================================

This can be plot using a 2 dimensions lgraph. Here’s such a graph:


Let’s plot 
 The n=1



 model, illustrated in the figure below.




So, the training data set consists of m labeled examples, each denoted as $$x^{i}, i=1 to m$$ is the input data and it is labeled by $$y^{i} $$, the corresponding label, is the expected decision of the predictor.$$(x^{i}, y^{i})$$, where $$x^{i}, i=1 to m$$ is the input data and it is labeled by $$y^{i} $$, the corresponding label, is the expected decision of the predictor.

So, for simplicity, we start with a 1D set of input points X, and can present it on a 2D graph, as shown in figure 3.

Figure 3: Linear Approximation







Looking at Figure 3, we can see that a linear approximation fits the training examples points well.
The line equation is $$^{\hat{y}}=wx+b$$, so now we need to calculate w and b that give the best approximation. We do that by defining a cost function J(w,b). Then we find w and b which minimize the cost function.
 We chose it to be the some of the squared errors:

$$J(w,b)=\frac{1}{m}\sum_{i=1}^{m}(\hat{y}^i-y^i)^2$$

Aim is to find w and b, such that minimize the cost function:
\\ \min_{w,b}J(w,b)

$$J(w,b)$$ is a 2 quadric function of 2 parameters. The 2D Graph below illustrates the surface defined by an equation of this type:





To find the best line equation, we should select a cost function, which minimizing 





Figure 1 describes the  Supervised Learning system, be it a regression or classification prediction.


Figure 1: Supervised Learning





The system runs in 3 stages, as presented in Figure 2.
Training - At which the predictor coefficient are calculated
Testing - At which the predictor performance is evaluated,
Prediction - The predictor calculates output Y based on input X

Figure 2: Supervised Learning - 3 stages




Here we will delve to the calculation of predictor’s coefficient.






