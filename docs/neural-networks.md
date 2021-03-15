## Deep Learning

Recall the Supervise Learning Model sketch, which was posted in the Logistic Regression post. Here it is again:

### Figure 1: Logistic Regression Data Network


![Supervise Learning Outlines](../assets/images/supervised/ML-data-sigmoid-network.svg)


The Logistic Regression data network module depicted in Figure 1 generates predicted value \\(\hat{y}\\). Figure 1 presents the module's sub elements elements:

****Input Data**** - The figure presents n dimensional input data \\( (x_1, x_2.....x_n) \\)
****Weights**** - The n input weigths vector  \\( (w_1, w_2.....w_n) \\) multiplies the input data vector.
****Bias**** - The bias coefficent b is summed with the weighted input data.
****Sigmoid**** - This is the non-linear element which is perated on the weighted input and bias.



The Neural Network scheme cosiders the above module a single Nueron, while a Neural Network, is a network of Neurons arranged as depicted in Figure 2.

### Figure 2: Deep Neural Network


![Supervise Learning Outlines](../assets/images/neural-networks/deep-neural-network.svg)


Let's start with noting some common tersm:

- Layers: The Neural Network is arranged in layers. The network presented in Figure 2 has 4 layers - marked L1-L4.
- Input Layer: The input layer is the first layer of the Nural Network. It is conventionally not counted in the layers count - (otherwise tFigure 5 would be regarded as a 5 layers networks).
- Hiden Layer: Layers which both input and output are both connected to other network's  layers - L1-L3 are Hidden layers.
- Output Layer: The Last network's layer-  L4 in Figure 2.
- Deep Neural Network: A neural network with many layers. There is no definite minimal number of layers for that, though 3 hidden layers used to be regarded "Deep".







A leyer is a group of Neurons, not connected to each other  
The network presented in Figure 2 has 4 layers, as marked on the diagram. The input layer is not counted as a layer - this is the common convention. The 3 inner network (L1-L3) are called Hidden Layers, since their inputs and outputs are internal.












a meshed networklayered meshed  of 


like logistic regression, repeated a lot of times

Previous posts presented Machine Learning, and its computation element - it is depicted in Figure 1.


If there are many layers without an activation function, it is always computing a linear prediction function, no matters how layers the network has.


The case with no activation function is the linear regression - predict a price etc.



and Deep learning is an approach to machine learning characterized by deep stacks of computations. This depth of computation is what has enabled deep learning models to disentangle the kinds of complex and hierarchical patterns found in the most challenging real-world datasets.

Through their power and scalability neural networks have become the defining model of deep learning. Neural networks are composed of neurons, where each neuron individually performs only a simple computation. The power of a neural network comes instead from the complexity of the connections these neurons can form.
