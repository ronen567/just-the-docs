# Forward Propogation

### Introduction
To move forward through the network, called a forward pass, we iteratively use a formula to calculate
This section describes the Forward Propogation in Neural Networks. Feed Forward is the transfer of the dataset input through the Neural Networks. We will present here in details the forwarding equations.

### Recap: Neural Networks

Neural Networks were introduced in the previous post. To recap, here are 2 already posted schemes: Figure 1 depicts a Neuron which is the building block of the Neural Networks, and Figure 2 presenta Neural Network, though quite quite shallow, which we'll use to illustrate the forward propogation.

 ### Figure 1: Neuron with Activation function g(z)
![Supervise Learning Outlines](../assets/images/neural-networks/general_neuron.svg)
 
 
 
 ### Figure 3: Neural Network - A More Detailed Scheme

![Supervise Learning Outlines](../assets/images/neural-networks/neural-network.svg)





Let's start with the activation expressions for layer 1's Neurons:

### Eq. 1: Neuron 1 Layer 1 Weighted Input
$$
z_1^{[1]}=\begin{bmatrix}
w_{11}^{[1]} && 
w_{21}^{[1]} 
\end{bmatrix}\begin{bmatrix}
x_1 \\\\\\
x_2 \\\\\\
\end{bmatrix}+b_1^{[1]}
$$

### Eq. 2: Neuron 1 Layer 1 activation


$$
a_1^{[1]}=g_1^{[1]}(z_1^{[1]})
$$

### Eq. 3: Neuron 2 Layer 1 Weighted Input

$$
z_2^{[1]}=\begin{bmatrix}
w_{12}^{[1]} && 
w_{22}^{[1]} 
\end{bmatrix}\begin{bmatrix}
x_1 \\\\\\
x_2 \\\\\\
\end{bmatrix}+b_2^{[1]}
$$

### Eq. 4: Neuron 2 Layer 1 activation


$$
a_2^{[1]}=g_2^{[1]}(z_2^{[1]})
$$


### Eq. 5: Neuron 3 Layer 1 Weighted Input

$$
z_3^{[1]}=\begin{bmatrix}
w_{13}^{[1]} && 
w_{23}^{[1]} 
\end{bmatrix}\begin{bmatrix}
x_1 \\\\\\
x_2 \\\\\\
\end{bmatrix}+b_3^{[1]}
$$

### Eq. 6: Neuron 3 Layer 1 activation


$$
a_3^{[1]}=g_3^{[1]}(z_3^{[1]})
$$



Let's arrange the above equations in a more compact matrix expressions:


### Eq. 7: Layer 1 Weighted input in a more compact matric format


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
x_1 \\\\\\ 
x_2
\end{bmatrix}+\begin{bmatrix}
b_1^{[1]} \\\\\\ 
b_2^{[1]} \\\\\\ 
b_3^{[1]} 
\end{bmatrix}
$$

### Eq. 8: Layer 1 activation in a more compact matric format

$$
\begin{bmatrix}
a_1^{[1]} \\\\\\
a_2^{[1]} \\\\\\
a_3^{[1]}
\end{bmatrix}=
\begin{bmatrix}
g_{1}^{[1]} \\\\\\
g_{2}^{[1]} \\\\\\ 
g_{3}^{[1]} 
\end{bmatrix}
\begin{bmatrix}
z_1^{[1]} \\\\\\ 
z_2^{[1]} \\\\\\ 
z_3^{[1]} 
\end{bmatrix}
$$

Now we can re-write Eq. 7 and Eq. 8 for expression of Layer 2 activation. 


### Eq. 9: Layer 2 Weighted input in a more compact matric format


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

### Eq. 10: Layer 2 activation in a more compact matric format

$$
\begin{bmatrix}
a_1^{[2]} \\\\\\
a_2^{[2]} 
\end{bmatrix}=
\begin{bmatrix}
g_{1}^{[2]} \\\\\\
g_{2}^{[2]}
\end{bmatrix}
\begin{bmatrix}
z_1^{[2]} \\\\\\ 
z_2^{[2]}
\end{bmatrix}
$$

Note that Eq. 9 is similar to Eq. 7 in concept, except the input vector which is the activation output of the previous layer, i.e.  \\(\bar{a}^{[1]}\\).

To generelize the expression for all layers, we will denote the input vector of layer 1 by \\(\begin{bmatrix}
a_1^{[0]} \\\\\\
a_2^{[0]} \\\\\\
a_3^{[0]}
\end{bmatrix}\\). Now we can have a generalized expression for the activation of any layer l, l=1:L, in any Neural Network. The input will be denoted by \\(a^{[l-1]}\\), the output by  \\(a^{[l]}\\), the weights matrix by  \\(w^{[l]}\\)

### Eq. 11: Weighted input of Layer l, l=1:L
$$
z^{[l]}=w^{[l]}a^{[l-1]}+b^{[l]}
$$

### Eq. 12: Activation of Layer l, l=1:L
$$
A^{[1]}=g^{[l]}(Z^{[l]})
$$


## Matrix Dimensions
