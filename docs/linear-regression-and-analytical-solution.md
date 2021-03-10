---
layout: default
nav_order: 2
title: Linear Regression Analytical Solution
---
# Linear Regression with the Analytical Solution

Consider the following (very simplified problem): It is needed to predict homes' price, based on 6 features: their address, floor, number of rooms, number of bethrooms, area, and schools. How should we do that? To perform the task we need to set a prediction algorithm, and train it to predict homes prices.
Sounds simple right? This process is detailed in this post.


## Recap - Regression Supervised Predictive model  
The prediction model which should be implemented for this problem is the Regression Supervised Predictive model. A Regression predictor is the predictor which predicts continous values shuch as home prices. The supervised model, runs through 3 phases,as depicted in Figure 1 (a-c). 

Figure 1 presents a system with n-dimensional input (AKA n features), denoted by\\(x_1-x_n\\), and the predictor's corresponding n+1 coefficients, denoted by \\(b\\) - the bias coefficient, and \\(w_1-w_n\\). 

Here's a brief description of the 3 phases:

***Training Phase*** - during the Training phase, the predictor's coefficients are calculated, as depicted by Figure 1-a. The Training data consists of m examples, denoted by superscripts 1-m. The optimizer calculates the predictor's coeffcients, by minimizing a cost function, which expresses the error between the expected value \\(y\\) and the predcited/modeled value \\(\hat{y}\\). 
***Validation Phase*** - during the Validation phase, the predictor's performance is validated, using a cost function as depicted in Figure 1-b. In case results are not acceptable, the Training phase should be revisited. The input dataset consists of k sets om labeled input data.
***Normal Data Phase*** - During the Data phase, the system predicts the output for the input data.


### Figure 1: Supervise Learning Model

### Figure 1-a: Training Phase.  
****The Training input Dataset consists of m labeled examples, each of n features.****


![Supervise Learning Outlines](../assets/images/supervised/ML-training-network.svg)

### Figure 1-b: Validation Phase
****The Validation input Dataset consists of k labeled examples, each of n features.****


![Supervise Learning Outlines](../assets/images/supervised/ML-validation-network.svg)


### Figure 1-c: Normal Data Phase
****The Normal Data input Dataset consists n features.****


![Supervise Learning Outlines](../assets/images/supervised/ML-data-network.svg)


## Linear Prediction


The first step towards solving the problem, should be the selection of the prediction model. This post presents the **Linear Predictor** aka **Linear Regression**

At the name Linear Predictor implies, the predicted output is modeled by a linear combination of the input dataset, as shown in Eq. 1. 
According to our notations, \\(\hat{y}\\) represents the the estimated value of  \\(y\\), \\({x_j}\\) represent the input data set, where j=1:n, and  \\({b, w_j}\\), i=1:n are the predictor's coeffcients. The estimation residual, AKA error is denoted by e. 


### Eq. 1: Linear Prediction 

$$y = \hat{y} + e = b+\sum_{j=1}^{n}w_jx_j +e $$

Where b stands for bias, and w for weight.


Let's return to our homes prices prediction example, but for our convinience, simplify it for now, reduce the number of input features to n=1. This reduction will simplify the comutation illustration and graphical presentation of the problem, but we will not use generality and develope thesolution for any n.

With that in mind, watch Table 1, which holds a set of 5 example data points, each relates to an appartment which is charectarized by its floor, and  labeled by its price. Based on those examples, we should set a model which will predict apartments price, given its floor. (Obviously the floor  by itself is not sufficient for a valid apartment price prediction, but still, we take that for the sake of example's simplicity, but without losing generality - we will show the solution for any n. Besides the too-small number of features,the number of examples, m=5, is obviously too small for a good prediction, but again, it's all for the sake of the example's simplicity.


### Table 1:  Apartments price prediction - single feature


|Floor    |Price (M $)|
|:--------|:----------|
|0        | 0.4       |
|1        | 0.7       |
|2        | 1.1       |
|3        | 1.4       |
|4        | 1.65      |        


For Table 1's single feature data, the predictor equation of Eq. 1, reduces to the equation specified in Eq 2A. Pluging the Table 1's data to Eq. 2a, results in Eq. 2b.

### Eq. 2a: Linear Prediction with n=1 

$$y = \hat{y} + e=b+w_1x_1 + e$$


Inserting table 1's data to Eq. 2a gives the following set of equations:

### Eq. 2b:
$$b+w_1*0+\epsilon^{(1)}=0.4
$$

$$
b+w_1*1+\epsilon^{(2)}=0.7
$$

$$
b+w_1*2+\epsilon^{(3)}=1.1
$$

$$
b+w_1*3+\epsilon^{(4)}=1.4
$$

$$
b+w_1*4+\epsilon^{(5)}=1.65
$$

From the above equations the coefficients \\(b,w_1\\) can be found. We will get to solving the equations later in this post.



Figure 1 presents the Table 1's 5 data points, along with a line regression estimator graph.  

### Figure 1: Linear Prediction - 1D

![Linear Predictiopn 1d](../assets/images/linear-regression/linear_1d_prediction.png)


According to the graph sketched in Figure 1, the linear model indeed seems to fit in as an estimator for the data points. To prove a validitiy of a model though,  one needs much more than 5 example data points. Needles to repeat here on the reasons for our seletion.

Have not yet explained how the fitting line was calculated, butlet's examine a higher order estimator, with n=2. So accordingly, Let's have now 2 features - floor number and number of bedrooms, as shown in Table 2.


### Table 2:  Apartments price prediction, with n=2

|Floor|Bedrooms | Price (M $)|
|:----|:--------|:-----------|
| 0   | 5       | 1.5        |
| 1   | 4       | 0.5        |
| 2   | 5       | 1          |
| 3   | 6       | 1.2        |
| 4   | 6       | 1.5        |
| 5   | 4       | 0.5        |
| 6   | 5       | 0.6        |
| 7   | 3       | 0.9        |
| 8   | 4       | 0.7        |
| 9   | 3       | 0.7        |
| 10  | 5       | 1.3        |



Still now, these 2 features are not enough for a valid prediction of apartments pricess, nor is the number of the example points. But that doesn't matter at this point. Anyway, with n<=2, it is still possible to present the datapoints and the predictor graphically, as shown in Figure 2.
Figure 2 presents the listed above dataset examples, (see x marker points), and based on those points, it presents a linear predictor points, which where  calculated with the predictor expression of Eq. 3, which is same as Eq. 1, but with n=2.

As before,  i=1:m, where m=11 is the number of examples. Nore that the Linear Predictor's calculated \\(\hat{y^{(i)}}\\)  points, are located on the colored presented surface of Fig 2.

### Eq. 3:

$$
y=\hat{y}^{(i)}+\epsilon^{(i)}=b+\sum_{j=1}^{2}w_j{x}^{(i)}_j+\epsilon^{(i)}
$$


Let's insert table 2's data to Eq. 3 gives the following set of equations:


$$b+w{_1}*0+w{_2}*5+\epsilon^{(1)}=1.5$$
$$b+w{_1}*1+w{_2}*4+\epsilon^{(2)}=0.5$$
$$b+w{_1}*2+w{_2}*5+\epsilon^{(3)}=1.0$$
$$b+w{_1}*3+w{_2}*6+\epsilon^{(4)}=1.2$$
$$b+w{_1}*4+w{_2}*6+\epsilon^{(5)}=1.5$$
$$b+w{_1}*5+w{_2}*4+\epsilon^{(6)}=0.5$$
$$b+w{_1}*6+w{_2}*5+\epsilon^{(7)}=0.6$$
$$b+w{_1}*7+w{_2}*3+\epsilon^{(8)}=0.9$$
$$b+w{_1}*8+w{_2}*4+\epsilon^{(9)}=0.7$$
$$b+w{_1}*9+w{_2}*3+\epsilon^{(10)}=0.7$$
$$b+w{_1}*10+w{_2}*5+\epsilon^{(11)}=1.3$$



From the above 11 equations, it is possible to calculate the coefficients \\( {b,w_1, w_2} \\). We will get to solving the equations in the next paragraph of this post.



### Figure 2: Linear Prediction - 2D

![Linear Predictiopn 2d](../assets/images/linear-regression/linear_prediction_2d.gif)


## Calculation of Predictor's Coeffcients
We can present Eq.1 in a matrix form \\(y=wx\\),  where  y, w and x are listed below. Considering that the equations are linear, isolution is straight forward:

1. Construct the y vector, which holds the true results of m trainaing examples.
2. Construct the x vector wich holds the related input features of m training examples, each example's input has n features.
The detailed solution follows in the next paragraph.


$$\y=\begin{bmatrix}
y^{(1)}
\\\\
y^{(2)}
\\\\
.
\\\\
.
\\\\
.
\\\\
y^{(m)}
\end{bmatrix}$$

$$b=\begin{bmatrix}
b\\\\\ 
w_1\\\\\
.\\\\
.\\\\\
. \\\\\
w_n
\end{bmatrix}$$


and 

$$x=\begin{bmatrix}
1 & x_1^{(1)} & x_2^{(1)} & x_3^{(1)} & . & . & x_n^{(1)}\\\\\
1 & x_1^{(2)} & x_2^{(2)} & x_3^{(2)} & . & . & x_n^{(2)}\\\\\
.& .& .& .& \\\\\
.& .& .& .& \\\\\
.& .& .& .& \\\\\
1 & x_1^{(m)} & x_2^{(m)} & x_3^{(m)} & . & . & x_n^{(m)} 
\end{bmatrix}$$


## Analytical Solution for Linear Predictor Coefficents

Given Eq. 1 we aim to find the predictor's coefficents \\( {b, w_i}\\),  j=1:n which will lead to the best predictor \\(\hat{y}\\) for y.

For our calculation, we need a set of labeled examples \\( {x_1...x_n, y} \\) See Eq. 4, where the superscript which runs from 1 to m is the exaple index, and the subscript is the parameter index, which runs fro, 1 to n. 

### Eq. 4: Linear Predictor, n dimensional input, m examples 

$$y^{(1)}=b+w_1x_1^{(1)}+w_2x_2^{(1)}+w_3x_3^{(1)}+....w_nx_n^{(1)}+\epsilon^1$$

$$y^{(2)}=b+w_1x_1^{(2)}+w_2x_2^{(2)}+w_3x_3^{(2)}+....w_nx_n^{(2)}+\epsilon^2$$
$$.$$
$$.$$
$$.$$
$$y^{(m)}=b+w_1x_1^{(m)}+w_2x_2^{(m)}+w_3x_3^{(m)}+....w_nx_n^{(m)}+\epsilon^2$$



Let's present Eq. 4 in a matrix format:

### Eq. 5: 

$$\begin{bmatrix}
y^{(1)}
\\\\
y^{(2)}
\\\\
.
\\\\
.
\\\\
.
\\\\
y^{(m)}
\end{bmatrix}= \begin{bmatrix}
1 & x_1^{(1)} & x_2^{(1)} & x_3^{(1)} & . & . & x_n^{(1)}\\\\\
1 & x_1^{(2)} & x_2^{(2)} & x_3^{(2)} & . & . & x_n^{(2)}\\\\\
.& .& .& .& \\\\\
.& .& .& .& \\\\\
.& .& .& .& \\\\\
1 & x_1^{(m)} & x_2^{(m)} & x_3^{(m)} & . & . & x_n^{(m)} 
\end{bmatrix} \begin{bmatrix}
b\\\\\ 
w_1\\\\\
.\\\\
.\\\\\
. \\\\\
w_n
\end{bmatrix}+\begin{bmatrix}
\epsilon^{(1)}\\\\ 
\epsilon^{(2)}\\\\  
.\\\\\
.\\\\ 
. \\\\\ 
\epsilon^{(n)}
\end{bmatrix}
$$

And in a more compact format:

### Eq. 6: 

$$
Y=XW+\epsilon
$$

And dropping  \\(\epsilon\\), leaving aside the prediction residual we get:

Eq. 6: 

$$
Y=XW
$$

Bow let's do some basic well-known Linear Algebra gimnastics:

Matrix X dimensions are mXn, where m >> n, i.e. m, the number of examples, should be be much greater than n, the input's dimensions. 
Accordingly, considering X is not square, it is not invertible. Still, if X is Full Rank, i.e. it's columns are linear independent, then \\(X^TX\\) is invertible.
So multiply each side of Eq. 6 by  \\(X^T\\):

### Eq. 7: 

$$
X^TY=X^TXW
$$

Multiply each side of Eq. 7 by \\((X^TX)^{-1}\\) :

### Eq. 8: 

$$
(X^TX)^{-1}X^TY=(X^TX)^{-1}X^TXW
$$


Since   \\((X^TX)^{-1}X^TX\\)=I   Eq. 8 reduces to:

### Eq. 9: 

$$
\mathbf{(X^TX)^{-1}X^TY=W}
$$

#### And that's it! We have the solution for the predictor's coefficents.

To sum up the solution, we 
Sample some training data and run it through the network to make predictions.
Measure the loss between the predictions and the true values.
Finally, adjust the weights in a direction that makes the loss smaller.



## Ilustrating the Solution by Calculating the 1D Predictor

Let's illustrate Eq. 9 on the 1D predictor listed above. Let's plug Eq.2 into the components of Eq. 9, as shown in Eq. 10.

### Eq. 10: 

$$Y=\begin{bmatrix}
0.4\\\ 
0.7\\\\
1.1\\\\\
1.4\\\\\
1.65
\end{bmatrix}$$

$$ X=
\begin{bmatrix}
1 & 0\\\ 
 1 & 1\\\\
 1 & 2\\\\
 1 & 3\\\\
 1 &4
\end{bmatrix}$$

$$
w=\begin{bmatrix}
b\\\\\
w_1
\end{bmatrix}
$$

$$
X^TY=\begin{bmatrix}
0 &1&2&3&4 \\\\\
1&1&1&1&1
\end{bmatrix}
\begin{bmatrix}
0.4\\\ 
0.7\\\\
1.1\\\\\
1.4\\\\\
1.65
\end{bmatrix}=
\begin{bmatrix}
5.25  \\\ 
13.7 
\end{bmatrix}
$$

$$
X^TX
=\begin{bmatrix}
0 &1&2&3&4 \\\\\
1&1&1&1&1
\end{bmatrix}
\begin{bmatrix}
1 & 0\\\ 
 1 & 1\\\\
 1 & 2\\\\
 1 & 3\\\\
 1 &4
 \end{bmatrix}=
\begin{bmatrix}
5 &10 \\\ 
10 & 30
\end{bmatrix}
$$


Inverse of a 2X2 matrix is given by:

$$
\begin{bmatrix}
a &b \\\\\
c & d
\end{bmatrix}^{-1}=\frac{1}{\begin{vmatrix}
d & -b \\\\\
-c & a
\end{vmatrix}}*\begin{bmatrix}
d &-b \\\\\
-c & a
\end{bmatrix}
$$

Where the denominator is a determinant, so:

$$\begin{vmatrix}
d & -b \\\\\ 
-c & a
\end{vmatrix} = da-bc$$

So: 

$$
(X^TX)^{-1}=\begin{bmatrix}
0.6 &-0.2 \\\\\\
-0.2 & 0.1
\end{bmatrix}
$$





We have now all the building block to complete the calculation:


$$
w=\begin{bmatrix}
b\\\\\
w_1
\end{bmatrix}=(X^TX)^{-1}(X^TY)=
\begin{bmatrix}
0.6 &-0.2 \\\\\\
-0.2 & 0.1
\end{bmatrix}
\begin{bmatrix}
5.25\\\\\
13.7
\end{bmatrix}=
\begin{bmatrix}
0.41\\\\\
0.32
\end{bmatrix}
$$



The predictor's coefficients are \\(b=0.41\\) and \\(w_1=0.32\\), which define the regression line:
***\\(\hat{y}=0.41+0.32x\\)***. 

The line predictor sketched in Figure 1.


***To conclude***, Eq. 9 provides the analytical solution, which determines the prediction's coefficients \\W=( b, w_j...w_n \\), based on the set of labeled data \\( X=({x_i}^{(2)}...{x_n}^{(i)}), Y=y^{(i)} \\) where i is the example's index, running from 1 to m.

Here's Eq. 9 again:
$$
(X^TX)^{-1}X^TY=W 
$$
where
$$
W = \begin{bmatrix}
b\\\\\
w_1\\\\\
w_2\\\\\
.\\\\\
.\\\\\
w_n
\end{bmatrix}
$$


The drawbacks of the analyical solution is the need to inverse natrix \\(X^TX\\), which dimension are (n+1)*(n+1). So when the number of features is large, say overa few hundrerds or so, depending available computation power, the analytical solution might be too expensive. Alternatively, the solution may be calculated using Gradient Descent,  an optimized iterative solution. Next post introduces the Gradient Descent algorithm, and illustrates calculation Linear Predictor's coefficents.




