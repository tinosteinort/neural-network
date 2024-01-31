package network

import (
	"errors"
	"fmt"
)

type Network struct {
	input      []float64
	layers     [][]Neuron
	activation Activation
}

type Neuron struct {
	Weights   []float64
	Threshold float64
	Value     float64
}

type Activation struct {
	Name string
	Run  func(input []float64, n Neuron) float64
}

func (n *Network) Update(input []float64) error {
	if len(input) != len(n.input) {
		return errors.New("input values does not match input neuron count")
	}
	var values = input
	for _, layer := range n.layers {
		values = n.calculateLayer(values, layer)
		for i, _ := range layer {
			layer[i].Value = values[i]
		}
	}
	return nil
}

func (n *Network) calculateLayer(layerInput []float64, layerNeurons []Neuron) []float64 {
	result := make([]float64, len(layerNeurons))
	for lni, neuron := range layerNeurons {
		result[lni] = n.activation.Run(layerInput, neuron)
	}
	return result
}

func (n *Network) Output(i int) (float64, error) {
	outputLayer := n.layers[len(n.layers)-1]
	if i < 0 || i >= len(outputLayer) {
		//if i >= len(outputLayer) {
		return float64(0), fmt.Errorf("no output at %d", i)
	}
	return outputLayer[i].Value, nil
}

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

func (b *Builder) Build() (*Network, error) {
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

	return &Network{
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
