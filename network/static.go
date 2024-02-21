package network

import "errors"

type staticNetwork struct {
	input      []float64
	layers     [][]Neuron
	activation Activation
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

func (n *staticNetwork) CreateSnapshot() *Snapshot {
	var layers []SnapshotLayer
	for _, layer := range n.layers {
		var yn []SnapshotNeuron
		for _, neuron := range layer {
			yn = append(yn, SnapshotNeuron{
				Threshold: neuron.Threshold,
				Weights:   neuron.Weights,
				Value:     neuron.Value,
			})
		}
		layers = append(layers, SnapshotLayer{
			Neurons: yn,
		})
	}

	return &Snapshot{
		Input:      n.input,
		Activation: n.activation.Name,
		Layers:     layers,
	}
}
