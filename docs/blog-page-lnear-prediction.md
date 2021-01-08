## Supervised Learning, Linear Regression and Gradient Descent


In this post we'll present a simple supervised learning problem example. Following that, we'll present a simple solution for the calculation of the Machine Learning's predictor's coeficients.


Let’s begin with an example problem: It is required make a Machine Learning Predictor which determines house pricing, based on 3 features:
1. Number of bedrooms
2. Zip code
3. Floor
as listed in Table 1. 

Note - obviously those 3 features are not enough to predict houses prices, but for the sake of simplicity, let's assume they are.

Table 1, consists of 4 training labeled data entries, aka examples, each labeled with the known corresponding price.



Table 1:  House price prediction - Labeled data


|`x1`-Number of bedrooms|`x3`- Zip code    |`x3`-Floor|`y`- Price|
|:----------------------|:------------------|:----------|:---------|
| 4                     |32321              | 2        | 179000   |
| 2                     |54322              | 3        | 210000   |
| 2                     |43243              | 4        | 250000   |
| 3                     |63422              | 1        | 260000   |



Following now, is table 2, which consists of regular features data- 4 entries. The house price should be estimated for those, based on data features.


Table 2: House Price Prediction - Regular Data

| x1 -Number of bedrooms| x3 - Zip code     | x3 -Floor| y - Price|
|:----------------------|:------------------|:---------|:---------|
| 3                     |34321              | 3        |  ?       |
| 5                     |64322              | 5        |  ?       |
| 1                     |23241              | 4        |  ?       |
| 4                     |83422              | 2        |  ?       |


Our challenge is the following:

1. Select a prediction model
2. Train the model with the labeled examples, (table 1), so that this model will later be able to estimate normal data (Table 2).

### Select a prediction model
Let’s start with the first challenge, and determine the model. 
In this post, for the sake of simplicity, we chose the "Linear Regression" model. Note that however the currently most common prediction model used is "Logistic Regression", examining the simpler "Linear Regression" first, will be a foundation for the other more complicated algorithms. Still, you may skip this post and jump directly to "Logistic Regression" which will be presented in the post which follows.

With that said, let's start!

Linear Regression is a model which approximates y by $$/hat(y}$$, which is a linear function of the input x,  as expressed in Eq 1.

Eq. 1: Linear Prediction 

$$y\approx \hat{y}=b+\sum_{j=1}^{n}w_jx_j$$

Where:
$${x_j}$$ are the input's features, e.g. columns of tables 1 and 2.
$$n$$ is the dimension of the input features.
$$ b and {w_j}$$ are the estimator coefficients.


Now having the Linear Regression selected, the challenge is find the coefficients' values. This should be done during the Trainingphase.

Note that I deliberately keep the notation of the free coefficient, (aka bias), as {b}, rather then {w_0}. Reason - this is a common notation in many papers and books.
Rest of this post presents the mathematical development, which leads to finding formula for the predictor's coeficients values.



Eq 1 expresses  an n dimensional input model. A multi dimensional problem is hard to sketch graphically, not to say impossible. For the sake of simplicity, and without losing generality, let’s reduce the dimensionality of features to 1. Now it will be easier to sketch and illustrate the problem and the solution.

For the sake of simplicity, let's reduce the dimension $$n$$ of the predictor's Eq. 1 to $$n=1$$, making it a 2 dimensional problem. Results would be valid to any n, but with 2-3 dimensions, we can presnt the data graphically.

So Eq. 1 reduces to Eq. 2.  

Eq. 2: Line Approximation

$$ \hat{y} ={ b}+w_1{x}$$

Figure 1, illustrates the line approximity: We are looking for the line offset $$b$$ and slope $$w_1$$ which optimize the approximation of the line to the dara point. To make that optimization, we will define a cost function which expresses the distance between the model and the points it aims to model, and chose the coefficients which minimize that cost.

Figure 3: Line Approximation 

![Linear Approximation](../assets/images/linearApproximation.jpg)



The cost, denoted by $$J(w,b)$$ is presented in Eq. 3, where the cost function is chosen to be the sum of squared euclidiean distances between the line and data points, aka squared errors. 

Eq. 4: Cost function -Squared Errors

$$J(w,b)=\frac{1}{m}\sum_{i=1}^{m}(\hat{y}^i-y^i)^2$$

Figure 4 illustrates graphically the euclidiean distance between approximate and actual value $$d=\hat{y}-y$$.

Figure 5: Approximation Distance
![Approximation Distance](../assets/images/approximation/point_to_aprox_line_dist.png)



BTW, optionally other cost functions could be chosen, e.g.minimizing the absolute difference between coordinates, see Eq. 5.

Eq. 5: Cost function using absolute coordinates differences

$$J(w,b)=\frac{1}{m}\sum_{i=1}^{m}\left | \hat{y}^i-y^i \right |$$


Cost expression as in Eq. 5, is dependent on squared distances, thus increases more as error grows, and is commonly used.


Let's substutute  \hat{y}^i by b^i+w_1x^i in Eq. 4, and get Eq. 6:

Eq. 6: Cost function for line approximation

$$J(w,b)=\frac{1}{m}\sum_{i=1}^{m}(b^i+w_1x^i-y^i)^2$$


Plotting Eq 6, which is a quadric equation, will result in a surface, such as  illustrated in figure 7:

Figure 7:  Plot illustration of  $$J(b,w)$$:


![Approximation Surface](../assets/images/approximationSurface.jpg)



####Question####: Which are the coefficients {b, w1} which minimize Eq.6? 
####Answer####: Take a look at Figure 7! The cost function minimum is at the bottom of the curve. We need {b, w1} which correspond to this point.
####Question####: How should this point be found?
####Answer####: We'll calculate the partial derivatives of $$J(b,w)$$ as expressed in Eq. 6, set derivatives to 0, and solve for b and w1.
Here:


Eq. 7 Cost function derivative wrt b: 
$$\\\frac{\partial J(b,w)}{\partial  b}=\\\frac{\partial(\frac{1}{2m}\sum_{i=1}^{m}(b+w_1x^i-y^i)^2)}{\partial b}=$$

$$\\\frac{1}{2m}*\sum_{i=1}^{m}\frac{\partial(b+w_1x^i-y^i)^2}{\partial b}=$$
$$\\\frac{1}{m}\sum_{i=1}^{m}(b+w_1x^i-y^i)= \frac{1}{m}\sum_{i=1}^{m}(h(x^i)-y^i)$$

Eq. 8 Cost function derivative wrt w1: 

$$\\\frac{\partial J(b,w)}{\partial w1}=\\\frac{\partial(\frac{1}{2m}\sum_{i=1}^{m}(b+w_1x^i-y^i)^2)}{\partial  w1}=$$

$$\\\frac{1}{2m}*\sum_{i=1}^{m}\frac{\partial(b+w_1x^i-y^i)^2}{\partial  w1}=$$
$$\\\frac{1}{m}\sum_{i=1}^{m}(b+w_1x^i-y^i)*x^i=\\\frac{1}{m}\sum_{i=1}^{m}(h(x^i)-y^i)*x^i$$




the point where $$J(b,w)$$ is minimal
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
$$$$\\ \min_{w,b}J(w,b)$$

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






