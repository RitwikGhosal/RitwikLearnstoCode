# <span style="font-size: 20px;">Evaluate a Multi-Layer Perceptron</span>

<span style="font-size: 14px;">A multi-layer perceptron connects dense layers in sequence. Each layer receives the activation produced by the preceding layer, applies its own affine transformation, and passes the result through tanh. The final layer follows the same rule as every earlier layer in this problem.</span>

---

## <span style="font-size: 16px;">One layer at a time</span>

<span style="font-size: 14px;">Let $h_0$ be the supplied input vector. For layer $l$, the preactivation is:</span>

$$
a_l=W_lh_{l-1}+b_l
$$

<span style="font-size: 14px;">The layer output is:</span>

$$
h_l=\tanh(a_l)
$$

<span style="font-size: 14px;">The vector $h_l$ becomes the input to the next layer. This repeated composition is what makes the model multi-layered.</span>

---

## <span style="font-size: 16px;">Connected layer shapes</span>

<span style="font-size: 14px;">Each weight matrix has one row per output neuron and one column per input coordinate. If layer $l$ receives width $n_{l-1}$ and produces width $n_l$, then:</span>

$$
W_l\in\mathbb{R}^{n_l\times n_{l-1}}
$$

$$
b_l\in\mathbb{R}^{n_l}
$$

<span style="font-size: 14px;">The next layer must accept width $n_l$. Connected shapes ensure that every matrix-vector product is defined and that each bias vector aligns with its layer's neurons.</span>

---

## <span style="font-size: 16px;">Tanh applies after every layer</span>

<span style="font-size: 14px;">Some neural-network designs leave the final layer linear, but this problem follows the lecture model and applies tanh after the final affine transformation as well.</span>

$$
h_L=\tanh(W_Lh_{L-1}+b_L)
$$

<span style="font-size: 14px;">The final output is therefore bounded between $-1$ and $1$ in every coordinate. Omitting the last tanh would change both ordinary outputs and saturated cases.</span>

---

## <span style="font-size: 16px;">A one-layer example</span>

<span style="font-size: 14px;">Consider an input with coordinates $1$ and $2$. A single layer uses the identity matrix and zero bias, so its preactivation remains unchanged:</span>

$$
a_1=
\begin{pmatrix}
1&0\\
0&1
\end{pmatrix}
\begin{pmatrix}
1\\
2
\end{pmatrix}
=
\begin{pmatrix}
1\\
2
\end{pmatrix}
$$

<span style="font-size: 14px;">Applying tanh coordinate by coordinate gives approximately:</span>

$$
h_1=
\begin{pmatrix}
0.761594\\
0.964028
\end{pmatrix}
$$

<span style="font-size: 14px;">Because this network has one layer, the final output and the only recorded layer output contain the same tensor values.</span>

---

## <span style="font-size: 16px;">A two-layer cancellation example</span>

<span style="font-size: 14px;">Suppose a scalar input of $1$ feeds two first-layer neurons with weights $1$ and $-1$, both with zero bias. Their activations are equal in magnitude and opposite in sign:</span>

$$
h_1=
\begin{pmatrix}
\tanh(1)\\
\tanh(-1)
\end{pmatrix}
=
\begin{pmatrix}
0.761594\\
-0.761594
\end{pmatrix}
$$

<span style="font-size: 14px;">If the second layer adds those two coordinates with zero bias, its preactivation is zero and its tanh output is also zero. The example shows that each layer must consume the preceding activation rather than the original input.</span>

---

## <span style="font-size: 16px;">Why retain every layer output</span>

<span style="font-size: 14px;">The result includes the final output and a list of all intermediate layer outputs in forward order. The list provides a trace of how the representation changes as it moves through the network.</span>

<span style="font-size: 14px;">These saved activations are also the values a later manual backward pass would need for tanh derivatives and weight gradients. The final tensor is the last element of this activation trace, but it is returned separately for convenience.</span>

---

## <span style="font-size: 16px;">Saturation, dtype, and device</span>

<span style="font-size: 14px;">Large positive or negative preactivations push tanh close to its limiting values. For example, a scalar preactivation of $20$ rounds extremely close to $1$. Saturation is expected behavior rather than a clipping step added by the implementation.</span>

<span style="font-size: 14px;">The input, weight matrices, and bias vectors share one floating-point dtype and device. Every returned activation tensor preserves both.</span>

---

## <span style="font-size: 16px;">Pitfalls</span>

* <span style="font-size: 14px;">**Feeding every layer the original input.** Each layer must consume the preceding layer's activation.</span>
* <span style="font-size: 14px;">**Skipping the final tanh.** This lecture model applies the nonlinearity after every affine layer.</span>
* <span style="font-size: 14px;">**Returning preactivations as the trace.** The required layer outputs are the tensors after tanh.</span>
* <span style="font-size: 14px;">**Losing forward order.** The activation list begins with the first layer output and ends with the final layer output.</span>
* <span style="font-size: 14px;">**Changing dtype.** Every layer output must preserve the shared input dtype.</span>
* <span style="font-size: 14px;">**Modifying supplied parameters.** Forward evaluation reads the network without changing weights, biases, or input tensors.</span>

<span style="font-size: 14px;">An MLP forward pass is a chain of repeated transformations. Correct composition, connected shapes, final-layer tanh, and a faithful activation trace define the result.</span>

---