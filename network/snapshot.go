package network

type Snapshot struct {
	Activation string          `yaml:"activation"`
	Input      []SnapshotInput `yaml:"input"`
	Layers     []SnapshotLayer `yaml:"layers"`
}

type SnapshotLayer struct {
	Neurons []SnapshotNeuron `yaml:"neuron"`
}

type SnapshotNeuron struct {
	Threshold float64   `yaml:"threshold"`
	Weights   []float64 `yaml:"weights,flow"`
	Value     float64   `yaml:"value"`
}

type SnapshotInput struct {
	Value float64 `yaml:"value"`
}

func (n *Network) CreateSnapshot() *Snapshot {
	var input []SnapshotInput
	for _, i := range n.input {
		input = append(input, SnapshotInput{
			Value: i.Value,
		})
	}

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
		Input:      input,
		Activation: n.activation.Name,
		Layers:     layers,
	}
}
