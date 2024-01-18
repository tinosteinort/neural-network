package network

import (
	"errors"
	"fmt"
)

type Network struct {
	inputNeurons int
	layers       [][]Neuron
}

type Neuron struct {
	Weights   []int
	Threshold int
}

func (n *Network) Run(input []int) ([]int, error) {
	if len(input) != n.inputNeurons {
		return nil, errors.New("input values does not match input neuron count")
	}
	var values = input
	for _, layer := range n.layers {
		values = calculateLayer(values, layer)
	}
	return values, nil
}

func calculateLayer(layerInput []int, layerNeurons []Neuron) []int {
	result := make([]int, len(layerNeurons))
	for lni, n := range layerNeurons {
		result[lni] = activate(layerInput, n)
	}
	return result
}

func activate(layerInput []int, n Neuron) int {
	value := 0
	for wi, w := range n.Weights {
		value = value + (layerInput[wi] * w)
	}
	if value-n.Threshold < 0 {
		return 0
	} else {
		return 1
	}
}

type NeuralNetworkBuilder struct {
	inputNeurons int
	layers       [][]Neuron
}

func NewNeuralNetworkBuilder(inputNeurons int) *NeuralNetworkBuilder {
	return &NeuralNetworkBuilder{
		inputNeurons: inputNeurons,
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
