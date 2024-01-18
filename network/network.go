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
	value bool
}

type layerNeuron struct {
	value     bool
	threshold int
	weights   []int
}

func (n *Network) Update(input []bool) error {
	if len(input) != len(n.input) {
		return errors.New("input values does not match input neuron count")
	}

	for li, layer := range n.layers {
		var layerInput []bool
		if li == 0 {
			layerInput = input
		} else {
			layerInput = valuesOf(n.layers[li-1])
		}
		calculateLayer(layerInput, layer)
	}
	return nil
}

func calculateLayer(layerInput []bool, layerNeurons []layerNeuron) {
	for lni, n := range layerNeurons {
		neuronValue := calculateValue(layerInput, n)
		if neuronValue >= n.threshold {
			layerNeurons[lni].value = true
		} else {
			layerNeurons[lni].value = false
		}
	}
}

func valuesOf(l []layerNeuron) []bool {
	var result []bool
	for _, neuron := range l {
		result = append(result, neuron.value)
	}
	return result
}

func calculateValue(layerInput []bool, n layerNeuron) int {
	value := 0
	for wi, w := range n.weights {
		v := boolAsNumber(layerInput[wi])
		value = value + (v * w)
	}
	return value
}

func boolAsNumber(b bool) int {
	if b {
		return 1
	}
	return 0
}

func (n *Network) Output(i int) bool {
	return n.layers[len(n.layers)-1][i].value
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
