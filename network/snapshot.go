package network

type Snapshot struct {
	InputNeurons int             `yaml:"input_neurons"`
	Activation   string          `yaml:"activation"`
	Layers       []SnapshotLayer `yaml:"layers"`
}

type SnapshotLayer struct {
	Neurons []SnapshotNeuron `yaml:"neuron"`
}

type SnapshotNeuron struct {
	Threshold int   `yaml:"threshold"`
	Weights   []int `yaml:"weights,flow"`
}

func (n *Network) CreateSnapshot() *Snapshot {
	var layers []SnapshotLayer
	for _, layer := range n.layers {
		var yn []SnapshotNeuron
		for _, neuron := range layer {
			yn = append(yn, SnapshotNeuron{
				Threshold: neuron.Threshold,
				Weights:   neuron.Weights,
			})
		}
		layers = append(layers, SnapshotLayer{
			Neurons: yn,
		})
	}

	return &Snapshot{
		InputNeurons: n.inputNeurons,
		Activation:   n.activation.Name,
		Layers:       layers,
	}
}
