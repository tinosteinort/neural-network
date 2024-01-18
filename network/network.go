package network

import (
	"errors"
	"fmt"
)

func StepFunction(input []int, n Neuron) int {
	value := 0
	for wi, w := range n.Weights {
		value = value + (input[wi] * w)
	}
	if value-n.Threshold < 0 {
		return 0
	} else {
		return 1
	}
}

type Network struct {
	inputNeurons int
	layers       [][]Neuron
	activate     ActivationFunction
}

type Neuron struct {
	Weights   []int
	Threshold int
}

type ActivationFunction func(input []int, n Neuron) int

func (n *Network) Run(input []int) ([]int, error) {
	if len(input) != n.inputNeurons {
		return nil, errors.New("input values does not match input neuron count")
	}
	var values = input
	for _, layer := range n.layers {
		values = n.calculateLayer(values, layer)
	}
	return values, nil
}

func (n *Network) calculateLayer(layerInput []int, layerNeurons []Neuron) []int {
	result := make([]int, len(layerNeurons))
	for lni, neuron := range layerNeurons {
		result[lni] = n.activate(layerInput, neuron)
	}
	return result
}

type NeuralNetworkBuilder struct {
	inputNeurons int
	layers       [][]Neuron
	activation   ActivationFunction
}

func NewNeuralNetworkBuilder(inputNeurons int) *NeuralNetworkBuilder {
	return &NeuralNetworkBuilder{
		inputNeurons: inputNeurons,
		activation:   StepFunction,
	}
}

func (b *NeuralNetworkBuilder) Build() (*Network, error) {
	if len(b.layers) == 0 {
		return nil, errors.New("no layer defined")
	}

	var layers [][]Neuron
	for layerIndex, layer := range b.layers {
		for _, neuron := range layer {
			if layerIndex == 0 {
				if len(neuron.Weights) != b.inputNeurons {
					return nil, errors.New("weight count does not match input neuron count")
				}
			} else {
				previousLayer := b.layers[layerIndex-1]
				if len(neuron.Weights) != len(previousLayer) {
					return nil, errors.New(fmt.Sprintf("weight count does not match previous layer[%d] neuron count", layerIndex-1))
				}
			}
		}
		layers = append(layers, layer)
	}

	return &Network{
		inputNeurons: b.inputNeurons,
		layers:       layers,
	}, nil
}

func (b *NeuralNetworkBuilder) WithLayer(neurons []Neuron) *NeuralNetworkBuilder {
	b.layers = append(b.layers, neurons)
	return b
}

func (b *NeuralNetworkBuilder) WithActivation(activation ActivationFunction) *NeuralNetworkBuilder {
	b.activation = activation
	return b
}
