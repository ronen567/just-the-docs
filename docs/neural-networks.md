# Deep Learning

## Introdcution

This post introduces Deep Learning, which is a branch of Machine Learning, using similar building blocks but in a denser architecture which can achieves better performance in complicated problems. Let's show that.


Figure 1 depicts the the scheme of Logistic Regression prediction model. Observing the scheme from its left side, the input data set consists of n elements \\(x_1-x_n\\), multiplied by n weights \\(w_1-w_n\\). The weighted inputs are sumed, together with the bias input b. Then this sum is activated by a sigmoid, which is a non linear function, used for binary decision. The sigmoid output, a, is here the predicted decision value , \\(\hat{y}\)). The weights and bias values are determined in the Training phase. 


### Figure 1: Logistic Regression Data Network

![Supervise Learning Outlines](../assets/images/neural-networks/logistic-regression-network.svg)

## Neural Networks

Figure resembles a Nueron, at least graphically-wise, where the data input lines resemble the dendrites, by which the neuron receives input from other cells.
Deep Learning algorithm are driven by a dense network, structured with many such Neurons, as depicted in Figure 2.

### Figure 2: Neural Network


![Supervise Learning Outlines](../assets/images/neural-networks/deep-neural-network.svg)

Following Figure 2, here below are some commonly used terms:

- **Layers**: The Neural Network is arranged in layers. The network presented in Figure 2 has 5 layers - marked L1-L5. Each layer consists of 4,4,4,2 and 1 neurons respectively.
- **Desnse Layers**: Fully connected layers. The Deep learning scheme is based on densely connected neural networks.
- **Input Layer**: A layer with input exposed to outside of the network. It is conventionally not counted in the layers count.
- **Hidden Layer**: Layers which both input and output are  connected to other network's layers - L1-L4 are Hidden layers.
- **Output Layer**: A layer with output exposed to outside of the network -  L5 in Figure 2.
- **Deep Neural Network**: A neural network with 'many layers'. There is no definite minimal number of layers for that, though 3 hidden layers used to be regarded "Deep".



Figure 3 depicts a Nueral Network with less layers and neurons, which might be more comfortable to illustrate its structure in more details. Some notes on the notations:
- **Superscript index in square bracketed**: This is the layer's index. 
- **Subscript index**:  This is the Neuron's index. 
- **weights**: The weights multiply the layer's data input at the Neuron's input. Example: \\(w_{21}^[2]\\) in Figure 3, weights the input coming from the Neuron 1 of Layer 1, to the Neuron 2 of layer 2. 
- **bias**: Bias multiplies a constant 1 and summed up with all weighted inputs. Example: \\(b_2^{[1]}\\) is the bias input of Neuron 2 of Layer 1.
- **Nuerons Output**: The Neurons output stage is represented by a, which stands for activation output. Example:  \\(a_2^{[1]}\\) is the output of first Neuron of Layer 1.



### Figure 3: Neural Network - A More Detailed Scheme



![Supervise Learning Outlines](../assets/images/neural-networks/neural-network.svg)


## Activation Functions

Each Neuron's operator consists of 2 parts, as depicted in Figure 1: The sum of wheigthed input with a bias, and a none linear operator. In Figure 1, the none linear operator is a sigmoid. Sigmoid is indeed the activation operator which performs Binary Decisions, used by Logistic Regression. This chapter presents more activation functions used for Deep Learning. 
Maybe here's the time to remark that in the absence of a non-linear activation functions, the Neural Network, which would be cascade of linear functions, could be replaced by a single Neurone with a linear function, so there would be no benefit over a single Neurone. 


The activation function is denoted by g(z). Figure 4 is almost identical to Figure 1, but depicts the Neuron with g(z).

### Figure 4: Neuton with Activation function g(z)
![Supervise Learning Outlines](../assets/images/neural-networks/general_neuron.png)




### Sigmoid

### Eq. 1: Sigmoid Function 

$$\sigma{x}=\frac{1}{1+e^{-x}}}$$

Sigmoid was introduced in the Logistic Regression post. With a decision threshold at 0.5, a range of [0,1], and a steep slope, Sigmoid is suitable a a binary decision function. and indeed it's very commonly used for binary classification.
Still, the sigmoid values flatens as at higher values of z. This the "Vanishing Gradient" problem, with which optimization algorithms such as Gradient Descent will not merge or merge very slowly. 

### Figure 5: Sigmoid



![Supervise Learning Outlines](../assets/images/neural-networks/sigmoid.png)

### tanh


### Eq. 2: tanh Function

$$
tanh(x)=\frac{e^x-e^{-x}}{e^{x}+e^{-x}}
$$

It's easy to see, by multiplying numerator, as shown in Eq 3. and denominator by \\(e^{-x}\\), that tanh is a scaled sigmoid. It is also depicted in Figure 6 that  tanh is a scaled sigmoid, centered around 0 instead of 0.5 with values [-1,1]..


### Eq. 3: tanh Function is a scaled sigmoid

tanh(x)=\frac{e^x-e^{-x}}{e^{x}+e^{-x}}*\frac{e^{-x}}{e^{-x}}= \frac{2}{1+e^{-2x}}-1=2\sigma(2x)-1$$

Tanh usually works better than Sigmoid for hidden layers. Actually, sigmoid is rarely used for hidden layers, but only for output layers, where the output is expected to be 0 or 1.

### Figure 6: tanh

![Supervise Learning Outlines](../assets/images/neural-networks/tanh.png)

### Relu

### Figure 7: Relu

### Eq. 4: Relu - Rectified Linear Unit

$$
Relu{x}=max(0,x)
$$

Relu solves the "Vanishing Gradient" problem. Derivative is 1 for the positive value. The derivative at x=0 is not defined, but that's not an issue and can be set to either 0 or 1. Relu implementation is simpler and cheaper computation wise then the other activation functions. is commonly used, actually it's in most cases the default activation function. 
Problem with Relu is the 0 gradient for negative values, so all units with negative value will slow down learning. Still, not considered a critical issues, as about half og the hidden unit are still expected to have values greater than 0.
Leaky Relu solves the 0 gradient issue anyway.


![Supervise Learning Outlines](../assets/images/neural-networks/relu.png)


### 

### Figure 8: Leaky Relu

### Eq. 5: Leaky Relu

$$
Relu{x}= maxx : 0.01*x)
$$

Leaky Relue adds a slope to the negative values, preventing the 0 gradient issue. The slope he is set to 0.01.

![Supervise Learning Outlines](../assets/images/neural-networks/leaky_relu.png)












If there are many layers without an activation function, it is always computing a linear prediction function, no matters how layers the network has.


The case with no activation function is the linear regression - predict a price etc.



and Deep learning is an approach to machine learning characterized by deep stacks of computations. This depth of computation is what has enabled deep learning models to disentangle the kinds of complex and hierarchical patterns found in the most challenging real-world datasets.

Through their power and scalability neural networks have become the defining model of deep learning. Neural networks are composed of neurons, where each neuron individually performs only a simple computation. The power of a neural network comes instead from the complexity of the connections these neurons can form.
