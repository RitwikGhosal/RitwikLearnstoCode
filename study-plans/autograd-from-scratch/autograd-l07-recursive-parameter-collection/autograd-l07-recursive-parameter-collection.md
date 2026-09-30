# <span style="font-size: 20px;">Collect Module Parameters Recursively</span>

<span style="font-size: 14px;">A neural-network module contains parameters inside layers and neurons rather than in one flat container. Parameter collection walks that hierarchy and exposes every scalar weight and bias in one deterministic order.</span>

---

## <span style="font-size: 16px;">The parameter hierarchy</span>

<span style="font-size: 14px;">The supplied network is organized at three nested levels:</span>

* <span style="font-size: 14px;">Layers appear in forward order.</span>
* <span style="font-size: 14px;">Neurons appear as weight-matrix rows.</span>
* <span style="font-size: 14px;">A neuron's input weights appear as columns within its row.</span>

<span style="font-size: 14px;">Each neuron also owns one aligned bias. The required traversal places that bias immediately after all weights belonging to the neuron.</span>

---

## <span style="font-size: 16px;">The exact ordering rule</span>

<span style="font-size: 14px;">For each layer, visit its neurons from the first row to the last. For each neuron, visit weights from the first input column to the last, then append that neuron's bias.</span>

<span style="font-size: 14px;">Conceptually, one neuron contributes the sequence:</span>

$$
w_{j,1},\ w_{j,2},\ldots,\ w_{j,n},\ b_j
$$

<span style="font-size: 14px;">The next neuron follows immediately, and the next layer begins only after every neuron in the current layer has been visited.</span>

---

## <span style="font-size: 16px;">A two-neuron layer</span>

<span style="font-size: 14px;">Consider one layer with two neurons and two input weights per neuron. The first weight row contains $1$ and $2$ with bias $5$. The second contains $3$ and $4$ with bias $6$.</span>

<span style="font-size: 14px;">The ordered scalar values are therefore:</span>

$$
1,\ 2,\ 5,\ 3,\ 4,\ 6
$$

<span style="font-size: 14px;">The bias $5$ appears before the second neuron's weights because biases are grouped with the neuron that owns them, not collected in a separate bias block.</span>

---

## <span style="font-size: 16px;">Several connected layers</span>

<span style="font-size: 14px;">In a multi-layer network, all parameters from the first layer appear before any parameter from the second. Within each layer, the neuron and input-column rules remain unchanged.</span>

<span style="font-size: 14px;">Connected shapes determine how many weights each later neuron owns. A layer receiving width $n$ gives each of its neurons $n$ weights followed by one bias. Parameter collection preserves the network hierarchy even though the returned container is flat.</span>

---

## <span style="font-size: 16px;">Why scalar views matter</span>

<span style="font-size: 14px;">The returned entries must be scalar tensor views into the supplied weight matrices and bias vectors. A view refers to the existing tensor storage rather than copying the numeric value into a new tensor.</span>

<span style="font-size: 14px;">This mirrors how a module exposes its parameters for optimization: the collected objects correspond to the actual stored parameters. Returning Python numbers or newly constructed tensors would preserve visible values but lose that relationship.</span>

---

## <span style="font-size: 16px;">Counting parameters</span>

<span style="font-size: 14px;">For a layer with $n_{\mathrm{in}}$ inputs and $n_{\mathrm{out}}$ neurons, the weight count is $n_{\mathrm{out}}n_{\mathrm{in}}$ and the bias count is $n_{\mathrm{out}}$:</span>

$$
N_{\mathrm{layer}}
=
n_{\mathrm{out}}n_{\mathrm{in}}+n_{\mathrm{out}}
$$

<span style="font-size: 14px;">The total network count is the sum across layers. It must equal the exact length of the returned parameter list, providing a direct consistency check on the traversal.</span>

<span style="font-size: 14px;">For a single scalar neuron, the count is simply one weight for each input plus one bias.</span>

---

## <span style="font-size: 16px;">Preserving dtype, device, and values</span>

<span style="font-size: 14px;">All supplied parameter tensors share one floating dtype and device. Scalar indexing preserves both properties in each returned view.</span>

<span style="font-size: 14px;">Collection is read-only. It does not reorder values inside a tensor, move parameters between devices, convert dtype, or modify any numeric entry. Only the returned list determines a traversal order.</span>

---

## <span style="font-size: 16px;">Pitfalls</span>

* <span style="font-size: 14px;">**Collecting all weights before all biases.** Each neuron's bias must follow that neuron's own weight row immediately.</span>
* <span style="font-size: 14px;">**Traversing columns before rows.** Neurons are weight-matrix rows and must remain in row order.</span>
* <span style="font-size: 14px;">**Returning copied tensors.** The contract requires scalar views into the supplied parameters.</span>
* <span style="font-size: 14px;">**Converting entries to Python numbers.** That loses tensor dtype, device, and view behavior.</span>
* <span style="font-size: 14px;">**Reporting an inferred count that differs from the list.** The integer count must equal the exact returned length.</span>
* <span style="font-size: 14px;">**Changing parameter values.** Collection observes the hierarchy without performing initialization or optimization.</span>

<span style="font-size: 14px;">Recursive parameter collection turns a nested module hierarchy into a deterministic flat view while preserving which tensors the scalars came from and how each neuron owns its parameters.</span>

---