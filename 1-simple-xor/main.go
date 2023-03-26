package main

/*
XOR Gorgonia Example by Josh Pattman

Here is some version info about my setup (other versions may work):
* `go 1.20.2`
* `gorgonia v0.9.17`
* `tensor v0.9.24`
*/

import (
	"fmt"

	G "gorgonia.org/gorgonia"
	T "gorgonia.org/tensor"
)

func main() {
	// ------------------------ Create the dataset ------------------------
	// Create the x (input)
	// This is a 4x2 matrix where 4 means 4 samples and 2 means 2 inputs
	x := T.New(
		T.WithShape(4, 2),
		T.WithBacking([]float64{0, 0, 0, 1, 1, 0, 1, 1}),
	)
	// Create the y (output)
	// This is a 4x1 matrix where 4 means 4 samples and 1 means 1 output
	y := T.New(
		T.WithShape(4, 1),
		T.WithBacking([]float64{0, 1, 1, 0}),
	)

	// ------------------------ Create the neural net ------------------------
	// Define the shape of the neural network. This network will be a dense feed forward network.
	numInputs, numHidden, numOutputs := 2, 2, 1

	// Define how many sample are in the dataset
	datasetLength := 4

	// Create the graph. A graph is effectively a collection of connected nodes.
	// A node is a mathematical operation, for example `add` or `multiply`.
	// When using gorgonia, we represent a neural network as a graph.
	g := G.NewGraph()

	// Create the input node. This is the node that we put the input data into the graph.
	// The input data (x) is a 4x2 matrix, so we create a 4x2 matrix node.
	// We don't have to actually put any data into the node yet, we can do that later.
	input := G.NewMatrix(g, G.Float64, G.WithShape(datasetLength, numInputs))

	// Create the weight nodes. These store the weights for the neural network.
	// We will initialise these with random values using WithInit(GolrotN())
	// We also want to add bias nodes, so the input to each layer is one larger than the output to the next layer. Later on we will append 1s to the input to each layer.
	weightsHidden := G.NewMatrix(g, G.Float64, G.WithShape(numInputs+1, numHidden), G.WithInit(G.GlorotN(1.0)))
	wieghtsOuput := G.NewMatrix(g, G.Float64, G.WithShape(numHidden+1, numOutputs), G.WithInit(G.GlorotN(1.0)))

	// Now we will define what happens when we want to run the neural network.
	// This is called the forwards pass.
	// We do this by telling gorgonia every calculation we want to do.

	// Create a bias node. This matrix will be concatenated onto each layer to add a value of one to the end of each sample
	bias := G.NewMatrix(g, G.Float64, G.WithShape(datasetLength, 1), G.WithInit(G.Ones()))

	// Append the bias value of 1 to the input
	inputWithBias := G.Must(G.Concat(1, input, bias))
	// Multiply the layer input with the weights
	hiddenLayer := G.Must(G.Mul(inputWithBias, weightsHidden))
	// Apply the activation function to the layer
	hiddenLayer = G.Must(G.Sigmoid(hiddenLayer))
	// Append the bias value of 1 to the hidden layer
	hiddenLayerWithBias := G.Must(G.Concat(1, hiddenLayer, bias))
	// Multiply the hidden layer with the output weights
	outputLayer := G.Must(G.Mul(hiddenLayerWithBias, wieghtsOuput))
	// Apply the activation function to the output layer
	outputLayer = G.Must(G.Sigmoid(outputLayer))

	// Our `outputLayer` node is the output of the neural network.
	// However, we need to tell gorgonia to read the value of it once it is calculated.
	// If we don't do this, gorgonia may overwrite the value of the node during learning.
	var outputValue G.Value
	G.Read(outputLayer, &outputValue)

	// Now we need to tell gorgonia how to train the neural network.
	// First we must define where we will put the target output values.
	// We can leave this empty for now, we will fill it in later.
	targetOutput := G.NewMatrix(g, G.Float64, G.WithShape(datasetLength, numOutputs))

	// We can now define how `loss` should be calulated, which is a measure of how good the neural network is at predicting the output.
	// We will use the mean squared error loss function.
	lossNode := G.Must(G.Mean(G.Must(G.Square(G.Must(G.Sub(outputLayer, targetOutput))))))

	// We will also tell gorgonia to read the value of the loss node.
	// This is only used for debugging, so we can see how the loss is changing during training.
	var lossValue G.Value
	G.Read(lossNode, &lossValue)

	// Finally, we need to tell gorgonia which parameters we should calculate the gradient for.
	// This is how backpropagation works.
	// We only want to calculate gradient with respect to the weights.
	G.Grad(lossNode, weightsHidden, wieghtsOuput)

	// ------------------------ Train the neural net ------------------------
	// In gorgonia, a graph is a bit like a set of instructions.
	// They say how to perform the calculations but cannot actually perform them themselves.
	// To perform the calculations, we need to create a machine.

	// We are going to create a tape machine.
	// This compiles our graph into a program that can be run.
	machine := G.NewTapeMachine(g)
	defer machine.Close()

	// We also need to define a solver algorithm.
	// This is basically an algorithm that takes the gradients and updates the weights.
	// We will use the Adam algorithm.
	solver := G.NewAdamSolver(G.WithLearnRate(0.05))

	// Now we can start training the neural network.
	// We will show it the training data 100 times.
	// Each time we show it the data is known as an epoch.
	for epoch := 0; epoch < 100; epoch++ {
		// First we want to reset the state of the machine, so previous calculations don't affect the current one.
		machine.Reset()
		// Now we can copy the input data into the graph
		G.Let(input, x)
		// And the target output data
		G.Let(targetOutput, y)
		// Now we can run the machine.
		// This will perform the forwards pass and calculate the loss.
		// It will also calculate the gradients.
		if err := machine.RunAll(); err != nil {
			// Somthing has not gone quite right with the calculation.
			panic(err)
		}
		// Now we can use the solver to update the weights.
		// We need to give it a list of the nodes that we want to update.
		solver.Step(G.NodesToValueGrads(G.Nodes{weightsHidden, wieghtsOuput}))
		// Finally, we can print the loss to see how it is changing.
		fmt.Printf("Epoch: %d, Loss: %.3f\n", epoch, lossValue)
	}

	// ------------------------ Test the neural net ------------------------
	// Reset the machine to clear the previous calculations.
	machine.Reset()
	// Copy the test data into the graph.
	G.Let(input, x)
	// Because we are using the same graph to predict as to train, we need to supply the target output.
	// However, if you wanted to, you could create a new graph without any of the loss nodes just for prediction.
	G.Let(targetOutput, y)
	// Run the machine.
	if err := machine.RunAll(); err != nil {
		panic(err)
	}
	// Print the output of the neural network.
	fmt.Println("\nPredictions:")
	for i := 0; i < datasetLength; i++ {
		xi, _ := x.Slice(G.S(i, i+1))
		yi, _ := y.At(i, 0)
		ypi := outputValue.Data().([]float64)[i]
		fmt.Printf("X: %v, Y: %v, YP: %.2f\n", xi, yi, ypi)
	}
}
