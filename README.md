# Perceptron and MultiLayerPerceptron
This is a Python code demonstrating the implementation of a single neuron Perceptron and a MultiLayerPerceptron using the NumPy library. The Perceptron models can be used to simulate logical gates such as the AND and OR gates, while the MultiLayerPerceptron demonstrates the training of an XOR gate.

## Dependencies
numpy
## Usage
### Perceptron Class
The Perceptron class represents a single neuron with the sigmoid activation function. It has the following attributes:

##### inputs: The number of inputs in the perceptron, not counting the bias.
##### bias: The bias term, which is set to 1.0 by default.
##### Initialization
    neuron = Perceptron(inputs=2, bias=1.0)
##### Setting Weights
    neuron.set_weights([10, 10, -15])  # Set weights for AND gate
##### Running the Perceptron
    result = neuron.run([0, 0])  # Run the perceptron with input [0, 0]
    print(result)  # Output: 0.5 (example output)
### MultiLayerPerceptron Class
The MultiLayerPerceptron class represents a multilayer perceptron (neural network) built using the Perceptron class. It has the following attributes:

##### layers: A list representing the number of neurons in each layer.
##### bias: The bias term used for all neurons in the network.
##### eta: The learning rate.
##### Initialization
    mlp = MultiLayerPerceptron(layers=[2, 2, 1], bias=1.0, eta=0.5)
##### Setting Weights
    mlp.set_weights([[[0.1, 0.2, -0.3], [0.4, -0.5, 0.6]], [[0.7, -0.8, 0.9]]])  # Set weights for the network
##### Running the MultiLayerPerceptron
    result = mlp.run([0, 0])  # Run the MLP with input [0, 0]
    print(result)  # Output: [0.48985245] (example output)
##### Training the MultiLayerPerceptron
    for i in range(3000):
        mse = 0.0
        mse += mlp.bp([0, 0], [0])  # Backpropagation with input [0, 0] and target [0]
        mse += mlp.bp([0, 1], [1])  # Backpropagation with input [0, 1] and target [1]
        mse += mlp.bp([1, 0], [1])  # Backpropagation with input [1, 0] and target [1]
        mse += mlp.bp([1, 1], [0])  # Backpropagation with input [1, 1] and target [0]
        mse = mse / 4
        if i % 100 == 0:
            print(mse)
##### Printing the Weights

    mlp.print_weights()
##### Example
The provided example demonstrates training an XOR gate using the MultiLayerPerceptron class. The weights of the network are initialized randomly, and the backpropagation algorithm is used to update the weights during training. Finally, the updated weights are printed, and the network is tested with different inputs.




