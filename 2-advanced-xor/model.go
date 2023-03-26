package main

import (
	G "gorgonia.org/gorgonia"
	T "gorgonia.org/tensor"
)

// This is the type that we will use to represent our neural network.
// It contains only the nodes that we actually need to use.
type NeuralNetwork struct {
	// Stuff used for training and testing
	g                  *G.ExprGraph // The graph
	inputLayer         *G.Node      // The input node
	hiddenLayerWeights *G.Node      // The weights for the hidden layer
	outputLayerWeights *G.Node      // The weights for the output layer
	outputValue        G.Value      // The output value

	// Stuff used only for training
	targetOutputLayer *G.Node // The target output node
	lossValue         G.Value // The loss value

	// Stuff used for running
	machine G.VM
}

// Create a new neural network. We need to specify whether we are creating a network for training or testing.
// The reason for this is that we don't need to create the nodes that are only used for training (loss, target output) when we are testing.
// This is good for efficiency.
func NewNeuralNetwork(isForTraining bool) *NeuralNetwork {
	n := &NeuralNetwork{}

	// Calculate the batch size. When we are training we want to use the whole dataset, so we set this to 4.
	// When we are testing we want to use a single sample, so we set this to 1.
	batchSize := 4
	if !isForTraining {
		batchSize = 1
	}

	// Define the shape of the neural network. This network will be a dense feed forward network.
	// I have chosen 5 hidden nodes as this is less likely to get stuck in a local minima
	numInputs, numHidden, numOutputs := 2, 5, 1

	// Create the graph for the nodes to live on.
	n.g = G.NewGraph()

	// Create the input node to put data into the network.
	n.inputLayer = G.NewMatrix(n.g, G.Float64, G.WithShape(batchSize, numInputs))

	// Create the weight nodes. These store the weights for the neural network.
	// We will initialise these with random values using WithInit(GolrotN())
	// We also want to add bias nodes, so the input to each layer is one larger than the output to the next layer. Later on we will append 1s to the input to each layer.
	n.hiddenLayerWeights = G.NewMatrix(n.g, G.Float64, G.WithShape(numInputs+1, numHidden), G.WithInit(G.GlorotN(1.0)))
	n.outputLayerWeights = G.NewMatrix(n.g, G.Float64, G.WithShape(numHidden+1, numOutputs), G.WithInit(G.GlorotN(1.0)))

	// Create the bias node, which is just a 1 that gets appended to the end of each sample.
	bias := G.NewConstant(T.Ones(T.Float64, batchSize, 1))

	// Create the nodes for forward pass. We don't store most of these becasue we don't need them later.
	inputWithBias := G.Must(G.Concat(1, n.inputLayer, bias))
	hiddenLayer := G.Must(G.Mul(inputWithBias, n.hiddenLayerWeights))
	hiddenLayer = G.Must(G.Sigmoid(hiddenLayer))
	hiddenLayerWithBias := G.Must(G.Concat(1, hiddenLayer, bias))
	outputLayer := G.Must(G.Mul(hiddenLayerWithBias, n.outputLayerWeights))
	outputLayer = G.Must(G.Sigmoid(outputLayer))

	// Read the output value
	G.Read(outputLayer, &n.outputValue)

	// If we are just testing, then the network is complete. Otherwise, we need to add the target and toss nodes
	if isForTraining {
		// Create the target output node
		n.targetOutputLayer = G.NewMatrix(n.g, G.Float64, G.WithShape(batchSize, numOutputs))

		// Create the loss node
		loss := G.Must(G.Mean(G.Must(G.Square(G.Must(G.Sub(outputLayer, n.targetOutputLayer))))))
		G.Read(loss, &n.lossValue)

		// Tell gorgonia to backpropagate the loss
		G.Grad(loss, n.getTrainableParameters()...)
	}

	// Create the machine. It is much faster to do this once here and reset it every batch than to create it every time we want to do a pass.
	n.machine = G.NewTapeMachine(n.g, G.BindDualValues(n.getTrainableParameters()...))

	return n
}

// This function will train the neural network on a batch of data. The neural net must have been created with isForTraining = true.
// We will return the loss of the batch.
func (n *NeuralNetwork) FitBatch(inputs, outputs T.Tensor, solver G.Solver) float64 {
	if n.targetOutputLayer == nil {
		panic("Cannot train a neural network that was not created for training")
	}
	// Reset the machine
	n.machine.Reset()

	// Set the values of the input and target output nodes
	G.Let(n.inputLayer, inputs)
	G.Let(n.targetOutputLayer, outputs)

	// Run the machine
	if err := n.machine.RunAll(); err != nil {
		panic(err)
	}

	// Update the weights with the provided solver
	solver.Step(G.NodesToValueGrads(n.getTrainableParameters()))

	return n.lossValue.Data().(float64)
}

func (n *NeuralNetwork) PredictSingle(input T.Tensor) T.Tensor {
	if n.targetOutputLayer != nil {
		panic("Cannot predict with a neural network that was created for training")
	}
	// Reshape the input to be a batch of size 1. The provided input is a single sample, but we need to provide a batch of size 1.
	inputReshaped := input.Clone().(T.Tensor)
	inputReshaped.Reshape(append([]int{1}, input.Shape()...)...)

	// Reset the machine
	n.machine.Reset()

	// Set the value of the input node
	G.Let(n.inputLayer, inputReshaped)

	// Run the machine
	if err := n.machine.RunAll(); err != nil {
		panic(err)
	}

	// Read the predicted ouput into a tensor
	predictedOutput := T.New(T.WithShape(n.outputValue.Shape()...), T.WithBacking(n.outputValue.Data()))

	// Reshape the output to be a single sample. The output is a batch of size 1, but we want to return a single sample.
	predictedOutput.Reshape(n.outputValue.Shape()[1:]...)

	// Return the output value
	return predictedOutput
}

func (n *NeuralNetwork) CopyWeightsToModel(model *NeuralNetwork) {
	// Copy the weights from this network to the model network
	G.Let(model.hiddenLayerWeights, n.hiddenLayerWeights.Value())
	G.Let(model.outputLayerWeights, n.outputLayerWeights.Value())
}

// This function returns the nodes that we want to train.
func (n *NeuralNetwork) getTrainableParameters() G.Nodes {
	return G.Nodes{n.hiddenLayerWeights, n.outputLayerWeights}
}
