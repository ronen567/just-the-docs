# Forward Propogation

### Introduction
To move forward through the network, called a forward pass, we iteratively use a formula to calculate
This section describes the Forward Propogation in Neural Networks. Feed Forward is the transfer of the dataset input through the Neural Networks. We will present here in details the forwarding equations.

### Recap: Neural Networks

Neural Networks were introduced in the previous post. To recap, here are 2 already posted schemes: Figure 1 depicts a Neuron which is the building block of the Neural Networks, and Figure 2 presenta Neural Network, though quite quite shallow, which we'll use to illustrate the forward propogation.

 ### Figure 1: Neuron with Activation function g(z)
![Supervise Learning Outlines](../assets/images/neural-networks/general_neuron.svg)
 
 

 
 ### Figure 2: Neural Network - A More Detailed Scheme

![Supervise Learning Outlines](../assets/images/neural-networks/neural-network.svg)



 ### Feed Forward Equations For a Single Layer

As shown in figure 1, the Feed Forward equation of a single neuron is given by Eq. 1a -b

### Eq. 1a
$$z=b+w^Tx$$

### Eq. 1b
$$a=g(z)$$


We need a more compact matrix expression, not just for a single neuron, but to all layer's neurons together. This expression is presneted in Eq. 2a-b

### Eq. 2a: Weighted input of Layer l, l=1:L
$$
z^{[l]}=w^{[l]}a^{[l-1]}+b^{[l]}
$$

### Eq. 2b: Activation of Layer l, l=1:L
$$
a^{[1]}=g^{[l]}(z^{[l]})
$$


The variables of Eq. 2a and Eq. 2b are all matrix and vectors. Let's illustrate those equations with the 3 layers of Figure 2.

 


### Eq. 7a1: Layer 1 Weighted input

Layer 1 is different from all other layers in the input data set which is the dataset x and not the activation of the previous layer. But if we represent the input data vector \\(\begin{bmatrix}
x_1 \\\\\\
x_2
\end{bmatrix}\\) by \\(\begin{bmatrix}
a_1^{[0]} \\\\\\
a_2^{[0]}
\end{bmatrix}\\), the generalized expression can fir for this layer too:



$$\begin{bmatrix}
z_1^{[1]} \\\\\\\\ 
z_2^{[1]} \\\\\\\\ 
z_3^{[1]}
\end{bmatrix}=
\begin{bmatrix}
w_{11}^{[1]}  & w_{21}^{[1]} \\\\\\ 
w_{12}^{[1]}  & w_{22}^{[1]} \\\\\\ 
w_{13}^{[1]}  & w_{13}^{[1]} 
\end{bmatrix} \begin{bmatrix}
a_1^{[0]}  \\\\\\ 
a_2^{[0]}
\end{bmatrix}+\begin{bmatrix}
b_1^{[1]} \\\\\\ 
b_2^{[1]} \\\\\\ 
b_3^{[1]} 
\end{bmatrix}
$$

### Eq. 7b: Layer 1 activation

$$\begin{bmatrix}
a_1^{[1]} \\\\\\\\ 
a_2^{[1]} \\\\\\\\ 
a_3^{[1]}
\end{bmatrix}=$$\begin{bmatrix}
g_1^{[1]}(z_1^{[1]}) \\\\\\\\ 
g_2^{[1]}(z_2^{[1]}) \\\\\\\\ 
g_3^{[1]}(z_3^{[1]})
\end{bmatrix}=



### Eq. 8: Layer 2 Weighted input


$$
\begin{bmatrix}
z_1^{[2]} \\\\\\\\ 
z_2^{[2]}
\end{bmatrix}=
\begin{bmatrix}
w_{11}^{[2]}  & w_{21}^{[2]} & w_{31}^{[2]} \\\\\\ 
w_{12}^{[2]}  & w_{22}^{[2]} & w_{32}^{[2]}
\end{bmatrix} \begin{bmatrix}
a_1^{[1]} \\\\\\ 
a_2^{[1]}
\end{bmatrix}+\begin{bmatrix}
b_1^{[2]} \\\\\\ 
b_2^{[2]} \\\\\\ 
\end{bmatrix}
$$

### Eq. 10: Layer 2 activation i
$$
\begin{bmatrix}
a_1^{[2]} \\\\\\
a_2^{[2]} 
\end{bmatrix}=

\begin{bmatrix}
g_{1}^{[2]} (z_1^{[2]}) \\\\\\ 
g_{2}^{[2]}(z_2^{[2]})
\end{bmatrix}
$$

The last layer has a single Neuron, so obviously the equations are:

### Eq. 8: Layer 2 Weighted input


$$
\begin{bmatrix}
z_1^{[3]} \\\\\\\\ 
\end{bmatrix}=
\begin{bmatrix}
w_{11}^{[3]}  & w_{21}^{[3]}  \\\\\\ 
\end{bmatrix} \begin{bmatrix}
a_1^{[2]} 
\end{bmatrix}+\begin{bmatrix}
b_1^{[3]}
\end{bmatrix}
$$


## Matrix Dimensions
