package network

import (
	"errors"
	"fmt"
	"gopkg.in/yaml.v3"
	"os"
)

type Network struct {
	inputNeurons int
	layers       [][]Neuron
	activate     Activation
}

type Neuron struct {
	Weights   []int
	Threshold int
}

type Activation struct {
	Name     string
	Function func(input []int, n Neuron) int
}

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
		result[lni] = n.activate.Function(layerInput, neuron)
	}
	return result
}

type NeuralNetworkBuilder struct {
	inputNeurons int
	layers       [][]Neuron
	activation   Activation
}

func NewNeuralNetworkBuilder(inputNeurons int, activation Activation) *NeuralNetworkBuilder {
	return &NeuralNetworkBuilder{
		inputNeurons: inputNeurons,
		activation:   activation,
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
		activate:     b.activation,
	}, nil
}

func (b *NeuralNetworkBuilder) WithLayer(neurons []Neuron) *NeuralNetworkBuilder {
	b.layers = append(b.layers, neurons)
	return b
}

func (b *NeuralNetworkBuilder) WithActivation(activation Activation) *NeuralNetworkBuilder {
	b.activation = activation
	return b
}

func (n *Network) Store(file string) error {

	var layers []yamlLayer
	for _, layer := range n.layers {
		var yn []yamlNeuron
		for _, neuron := range layer {
			yn = append(yn, yamlNeuron{
				Threshold: neuron.Threshold,
				Weights:   neuron.Weights,
			})
		}
		layers = append(layers, yamlLayer{
			Neurons: yn,
		})
	}

	yn := yamlNetwork{
		InputNeurons: n.inputNeurons,
		Activation:   n.activate.Name,
		Layers:       layers,
	}
	println(fmt.Sprintf("yamlNetwork: %v", &yn))

	data, err := yaml.Marshal(&yn)
	println(fmt.Sprintf("yamlNetwork: %v", data))
	if err != nil {
		return err
	}

	f, err := os.Create(file)
	if err != nil {
		return err
	}

	if _, err := f.Write(data); err != nil {
		return err
	}

	return nil
}

type yamlNetwork struct {
	InputNeurons int         `yaml:"input_neurons"`
	Activation   string      `yaml:"activation"`
	Layers       []yamlLayer `yaml:"layers"`
}

type yamlLayer struct {
	Neurons []yamlNeuron `yaml:"neuron"`
}

type yamlNeuron struct {
	Threshold int   `yaml:"threshold"`
	Weights   []int `yaml:"weights,flow"`
}
