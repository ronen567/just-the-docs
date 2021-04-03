# Back Propogation Example
In this post we will run a network Training (aka Fitting) example, based on the Back Propogation algorithm explained in the previous post.

The example will run a single Back Propogation cycle, to produce 2 outputs: \\(\frac{\mathrm{d} C}{\mathrm{d}{b^{[l]}}}\\) and \\(\frac{\mathrm{d} C}{\mathrm{d}{w^{[l]}}}\\) for 1<l<L.

## Example Details:

The network is depicted In Figure 1.

### Figure 1: The Network
![](../assets/images/deep-neural-network-fw-bw-example.png)

### Input data set:
The input data x, consists of m examples, where m=3. (Note that the examples are stacked in columns).

\\(A^{[0]}=\begin{bmatrix}
a_1^{[0]{(1)}}& a_1^{[0]{(2)}} & a_1^{[0]{(3)}} \\\\\\
a_2^{[0]{(1)}}& a_2^{[0]{(2)}} & a_2^{[0]{(3)}} \\\\\\
a_3^{[0]{(1)}}& a_3^{[0]{(2)}} & a_3^{[0]{(3)}}
\end{bmatrix}\\)

\\(A^{[0]}=\begin{bmatrix}
a_1^{[0](1)}& a_1^{[0](2)} & a_1^{[0](3)} \\\\\\
a_2^{[0](1)}& a_2^{[0](2)} & a_2^{[0](3)} \\\\\\
a_3^{[0](1)}& a_3^{[0](2)} & a_3^{[0](3)}\end{bmatrix}\\)

### Initial Paramerters

Here are the weights and bias of the 3 layers:

#### Weights layer 1:

\\(w^{[1]}=\begin{bmatrix}
w_{11}^{[1]} & w_{21}^{[1]} & w_{31}^{[1]}\\\\\\
w_{12}^{[1]} & w_{22}^{[1]} & w_{32}^{[1]}\\\\\\
w_{13}^{[1]} & w_{23}^{[1]} & w_{33}^{[1]}\\\\\\
w_{14}^{[1]} & w_{24}^{[1]} & w_{34}^{[1]}
\end{bmatrix}\\)

#### Bias layer 1:


\\(b^{[1]}=\begin{bmatrix}
b_{1}^{[1]}\\\\\\
w_{2}^{[1]}\\\\\\
w_{3}^{[1]}\\\\\\
w_{4}^{[1]}
\end{bmatrix}\\)

#### Weights layer 2:

\\(w^{[2]}=\begin{bmatrix}
w_{11}^{[2]} & w_{21}^{[2]} & w_{31}^{[2]} & w_{41}^{[2]}\\\\\\
w_{12}^{[2]} & w_{22}^{[2]} & w_{32}^{[2]} & w_{42}^{[2]}\\\\\\
w_{13}^{[2]} & w_{23}^{[2]} & w_{33}^{[2]} & w_{43}^{[2]}
\end{bmatrix}\\)

#### Bias layer 2:


\\(b^{[2]}=\begin{bmatrix}
b_{1}^{[2]}\\\\\\ 
w_{2}^{[2]}\\\\\\
w_{3}^{[2]} 
\end{bmatrix}\\)


#### Weights layer 2:

\\(w^{[3]}=\begin{bmatrix}
w_{11}^{[3]} & w_{21}^{[3]} & w_{31}^{[3]}
\end{bmatrix}\\)

#### Bias layer 3:


\\(b^{[2]}=\begin{bmatrix}
b_{1}^{[3]}
\end{bmatrix}\\)

## Feed Forward:

\\(Z^{[1]}=w^{[1]} \cdot A^{[0]} + b^{[1]}= \begin{bmatrix}
w_{11}^{[1]} & w_{21}^{[1]} & w_{31}^{[1]}\\\\\\
w_{12}^{[1]} & w_{22}^{[1]} & w_{32}^{[1]}\\\\\\
w_{13}^{[1]} & w_{23}^{[1]} & w_{33}^{[1]}\\\\\\
w_{14}^{[1]} & w_{24}^{[1]} & w_{34}^{[1]}
\end{bmatrix} \cdot 
\begin{bmatrix}
a_1^{[0]{(1)}}& a_1^{[0]{(2)}} & a_1^{[0]{(3)}} \\\\\\
a_2^{[0]{(1)}}& a_2^{[0]{(2)}} & a_2^{[0]{(3)}} \\\\\\
a_3^{[0]{(1)}}& a_3^{[0]{(2)}} & a_3^{[0]{(3)}}
\end{bmatrix} + 
\begin{bmatrix}
b_{1}^{[1]}\\\\\\
b_{2}^{[1]}\\\\\\
b_{3}^{[1]}\\\\\\
b_{4}^{[1]}
\end{bmatrix}=
\begin{bmatrix}
z_{11}^{[1]} & z_{12}^{[1]} & z_{13}^{[1]}\\\\\\
z_{21}^{[1]} & z_{22}^{[1]} & z_{23}^{[1]}\\\\\\
z_{31}^{[1]} & z_{32}^{[1]} & z_{33}^{[1]}\\\\\\
z_{41}^{[1]} & z_{42}^{[1]} & z_{43}^{[1]}
\end{bmatrix}
\\)






