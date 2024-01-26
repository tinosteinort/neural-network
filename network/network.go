package network

import (
	"errors"
	"fmt"
)

type Network struct {
	inputNeurons int
	layers       [][]Neuron
	activation   Activation
}

type Neuron struct {
	Weights   []float64
	Threshold float64
}

type Activation struct {
	Name string
	Run  func(input []float64, n Neuron) float64
}

func (n *Network) Run(input []float64) ([]float64, error) {
	if len(input) != n.inputNeurons {
		return nil, errors.New("input values does not match input neuron count")
	}
	var values = input
	for _, layer := range n.layers {
		values = n.calculateLayer(values, layer)
	}
	return values, nil
}

func (n *Network) calculateLayer(layerInput []float64, layerNeurons []Neuron) []float64 {
	result := make([]float64, len(layerNeurons))
	for lni, neuron := range layerNeurons {
		result[lni] = n.activation.Run(layerInput, neuron)
	}
	return result
}

type Builder struct {
	inputNeurons int
	layers       [][]Neuron
	activation   Activation
}

func NewBuilder(inputNeurons int, activation Activation) *Builder {
	return &Builder{
		inputNeurons: inputNeurons,
		activation:   activation,
	}
}

func (b *Builder) Build() (*Network, error) {
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
					return nil, fmt.Errorf("weight count does not match previous layer[%d] neuron count", layerIndex-1)
				}
			}
		}
		layers = append(layers, layer)
	}

	return &Network{
		inputNeurons: b.inputNeurons,
		layers:       layers,
		activation:   b.activation,
	}, nil
}

func (b *Builder) WithLayer(neurons []Neuron) *Builder {
	b.layers = append(b.layers, neurons)
	return b
}

func (b *Builder) WithActivation(activation Activation) *Builder {
	b.activation = activation
	return b
}
