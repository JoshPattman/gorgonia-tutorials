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
	x := T.New(
		T.WithShape(4, 2),
		T.WithBacking([]float64{0, 0, 0, 1, 1, 0, 1, 1}),
	)
	// Create the y (output)
	y := T.New(
		T.WithShape(4, 1),
		T.WithBacking([]float64{0, 1, 1, 0}),
	)

	// ------------------------ Create the neural net ------------------------
	trainingModel := NewNeuralNetwork(true)
	testingModel := NewNeuralNetwork(false)

	// ------------------------ Train the neural net ------------------------
	// Create an ADAM solver
	solver := G.NewAdamSolver(G.WithLearnRate(0.05))

	// We will train the neural network for 1000 epochs.
	for epoch := 0; epoch < 1000; epoch++ {
		loss := trainingModel.FitBatch(x, y, solver)
		fmt.Printf("Epoch: %d, Loss: %.3f\n", epoch, loss)
	}

	// ------------------------ Test the neural net ------------------------
	// First we will copy the weights from the training model to the testing model.
	trainingModel.CopyWeightsToModel(testingModel)

	// Now for each sample in the dataset we will run the testing model and print the output.
	// I will create the input tensor just before it is tested.
	// This is supposed to represent that you could use this model for realtime prediction.
	testXRaw := [][]float64{{0, 0}, {0, 1}, {1, 0}, {1, 1}}
	testYRaw := [][]float64{{0}, {1}, {1}, {0}}
	for i := 0; i < 4; i++ {
		testXTensor := T.New(T.WithShape(2), T.WithBacking(testXRaw[i]))
		predictedYTensor := testingModel.PredictSingle(testXTensor)
		fmt.Printf("Input: %v, Predicted Output: %.3f, Actual Output: %.3f\n", testXRaw[i], predictedYTensor.Data().([]float64)[0], testYRaw[i][0])
	}
}
