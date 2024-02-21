package network

import (
	"errors"
)

type Network interface {
	Update(input []float64) error
	Output() []float64
	CreateSnapshot() *Snapshot
}

type Neuron struct {
	Weights   []float64
	Threshold float64
	Value     float64
}

type staticNetwork struct {
	input      []float64
	layers     [][]Neuron
	activation Activation
}

type Activation struct {
	Name string
	Run  func(input []float64, n Neuron) float64
}

func (n *staticNetwork) Update(input []float64) error {
	if len(input) != len(n.input) {
		return errors.New("input values does not match input neuron count")
	}

	n.input = input

	var values = input
	for _, layer := range n.layers {
		values = n.calculateLayer(values, layer)
		for i, _ := range layer {
			layer[i].Value = values[i]
		}
	}
	return nil
}

func (n *staticNetwork) calculateLayer(layerInput []float64, layerNeurons []Neuron) []float64 {
	result := make([]float64, len(layerNeurons))
	for lni, neuron := range layerNeurons {
		result[lni] = n.activation.Run(layerInput, neuron)
	}
	return result
}

func (n *staticNetwork) Output() []float64 {
	outputLayer := n.layers[len(n.layers)-1]
	result := make([]float64, len(outputLayer))
	for ni, neuron := range outputLayer {
		result[ni] = neuron.Value
	}
	return result
}
