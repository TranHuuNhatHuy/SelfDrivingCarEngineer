# FROM LINEAR REGRESSION TO FEEDFORWARD NEURAL NETWORKS



## I. Linear Regression

- **L2 Loss (Mean Squared Error - MSE)**: very commonly used, sensitive to outliers.
- **L1 Loss (Least Absolute Deviations - LAD)**: suitable for dataset containing many outliers.


## II. Logistic Regression

- $P(Y|X) = mx + b$
- Logistic function (sigmoid): $f(X) = (e^{mX+b}) / (1 + e^{mX + b})$
- Softmax function is extension of the logistic function to multiple classes and takes a vector as input instead of a real number. The softmax function outputs a discrete probability distribution: a vector of similar dimensions to the input but with all its components summing up to 1.


## III. Cross-Entropy Loss and One-Hot Encoding

- The **Cross Entropy (CE) loss** is the most common loss for classification problems. The total loss is equal to the sum over all the observations of the dot product of the ground truth one-hot encoded vector and the log of the softmax probability vector.

- For multiple classes classification problems, the ground truth labels need to be encoded as vectors to calculate. A common approach is the **one-hot encoding** method, where each label of the dataset is assigned an integer. This integer is used as the index of the only non zero element of the one-hot vector.


## IV. Gradient Descent

- Iterative optimization algorithm.
- Updates weight using **gradient** and **learning rate $\alpha$**.

    $W - W \alpha \frac{df}{dx}$


## V. Stochastic Gradient Descent

- Because of memory limitations, the entire dataset is almost never loaded at once and fed through the model, as is the case in **batch gradient descent**. Instead, **batches** of inputs are created.
- Gradient descent performed on batches of just one input at a time is called **stochastic gradient descent (SGD)**, while batches of more than one, but not all at once (e.g. 20 batches of 200 images each), are called **mini-batch** gradient descent.


## VI. Other optimizers

- One way to overcome the local minima issue is to add a **momentum term**.
- This approach is taking its inspiration from physics: the optimizer should gain velocity when taking multiple consecutive steps in the same direction.
- This velocity will help the optimizer to overcome local minima.


## VII. Learning Rates and Annealing

- **Learning rate annealing** is a technique that improves the performances of gradient descent methods, which consists of decreasing the learning rate during training.
- Different strategies exist, such as stepwise annealing, cosine annealing or exponential annealing. The term **learning scheduler** is also used.


## VIII. Neural Network

- **Feed Forward neural networks (FFNN)** are an extension of the logistic regression algorithm. We can think of logistic regression as a FFNN with a single layer. FFNN are stacking multiple **hidden layers** (any layers that's not the input or output layer) followed by non linear activation, such as the sigmoid or softmax activations.
- FFNN are only made of **fully connected** layers, where each neuron in one layer is connected to all the neurons in the previous layer.


## IX. Backpropagation

- The chain rule allows you to decompose the calculation of the derivative of a composite function. 
- Because we can think of an ANN as a giant composite function, the chain rule is at the core of the backpropagation algorithm.
- Backpropagation is the mechanism used to calculate the gradient of the loss with respect to each weight of the neural network.


## X. Image Classification with FFNN

- Because feed forward neural networks take vectors as inputs, we need to flatten the `HxWxC` image to a `(HxWxC)x1` vector.
- Training neural networks is a costly and lengthy process. Because so much can go wrong when training a NN, it is critical for the ML engineer to **babysit** the training process. **Visualizing the loss and the metrics** will help identify any problems with the training process and allow for the engineer to stop, fix and restart the training.


## XI. Glossary

- **Activation function**: a non linear, differentiable function used in NN.
- **Backpropagation**: mechanism that propagates the gradient through a neural network.
- **Cross Entropy (CE) loss**: a loss function calculated by taking the dot product of the ground truth vector and the log of the vector of output probabilities.
- **Gradient descent**: iterative algorithm used to find the minimum of the loss function.
- **Global minimum**: where the loss function takes the smallest value of its entire domain.
- **Learning rate**: scalar controlling the step size of the gradient descent algorithm.
- **Local minimum**: minimum in a local subset of the loss function domain.
- **Mean Absolute Error / L1 loss**: a loss function calculated by summing the absolute difference of the ground truths and the predictions.
- **Mean Square Error (MSE) / L2 loss**: a loss function calculated by summing the square difference of the ground truths and the predictions.
- **Optimizer**: other name for gradient descent algorithms.
- **Tensors**: Multi-dimensional TensorFlow data structure, similar to numpy arrays.
- **Variable**: TensorFlow tensors that can be changed through operations.