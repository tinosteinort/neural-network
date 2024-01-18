package network

import (
	"errors"
	"fmt"
)

type Network struct {
	input  []inputNeuron
	layers [][]layerNeuron
}

type inputNeuron struct {
	value int
}

type layerNeuron struct {
	threshold int
	weights   []int
}

func (n *Network) Run(input []int) ([]int, error) {
	if len(input) != len(n.input) {
		return nil, errors.New("input values does not match input neuron count")
	}

	var values = input
	for _, layer := range n.layers {
		values = calculateLayer(values, layer)
	}
	return values, nil
}

func calculateLayer(layerInput []int, layerNeurons []layerNeuron) []int {
	result := make([]int, len(layerNeurons))
	for lni, n := range layerNeurons {
		result[lni] = activate(layerInput, n)
	}
	return result
}

func activate(layerInput []int, n layerNeuron) int {
	value := 0
	for wi, w := range n.weights {
		value = value + (layerInput[wi] * w)
	}
	if value-n.threshold < 0 {
		return 0
	} else {
		return 1
	}
}

type NeuralNetworkBuilder struct {
	inputNeurons int
	layers       []Layer
}

type Layer struct {
	Neurons []Neuron
}

type Neuron struct {
	Weights   []int
	Threshold int
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

	var layers [][]layerNeuron
	for layerIndex, layer := range b.layers {
		var layerNeurons []layerNeuron
		for _, neuron := range layer.Neurons {

			if layerIndex == 0 {
				if len(neuron.Weights) != b.inputNeurons {
					return nil, errors.New("weight count does not match input neuron count")
				}
			} else {
				previousLayer := b.layers[layerIndex-1]
				if len(neuron.Weights) != len(previousLayer.Neurons) {
					return nil, errors.New(fmt.Sprintf("weight count does not match previous layer[%d] neuron count", layerIndex-1))
				}
			}

			layerNeurons = append(layerNeurons, layerNeuron{
				threshold: neuron.Threshold,
				weights:   neuron.Weights,
			})
		}
		layers = append(layers, layerNeurons)
	}

	return &Network{
		input:  make([]inputNeuron, b.inputNeurons),
		layers: layers,
	}, nil
}

func (b *NeuralNetworkBuilder) WithLayer(neurons []Neuron) *NeuralNetworkBuilder {
	b.layers = append(b.layers, Layer{
		Neurons: neurons,
	})
	return b
}
