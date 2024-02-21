package network

import (
	"errors"
	"fmt"
)

type Builder struct {
	input      []float64
	layers     [][]Neuron
	activation Activation
}

func NewBuilder(activation Activation) *Builder {
	return &Builder{
		activation: activation,
	}
}

func (b *Builder) Build() (Network, error) {
	if len(b.layers) == 0 {
		return nil, errors.New("no layer defined")
	}

	if len(b.input) == 0 {
		return nil, errors.New("no input specified")
	}

	var layers [][]Neuron
	for layerIndex, layer := range b.layers {
		for _, neuron := range layer {
			if layerIndex == 0 {
				if len(neuron.Weights) != len(b.input) {
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

	return &staticNetwork{
		input:      b.input,
		layers:     layers,
		activation: b.activation,
	}, nil
}

func (b *Builder) WithInputNeurons(inputNeurons int) *Builder {
	b.input = make([]float64, inputNeurons)
	return b
}

func (b *Builder) WithInput(input []float64) *Builder {
	b.input = input
	return b
}

func (b *Builder) WithLayer(neurons []Neuron) *Builder {
	b.layers = append(b.layers, neurons)
	return b
}

func (b *Builder) WithActivation(activation Activation) *Builder {
	b.activation = activation
	return b
}
