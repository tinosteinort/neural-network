package snapshot

import (
	"github.com/tinosteionrt/neural-network/activation"
	"github.com/tinosteionrt/neural-network/network"
	"gopkg.in/yaml.v3"
	"os"
)

func Store(n *network.Network, file string) error {

	f, err := os.Create(file)
	if err != nil {
		return err
	}
	defer f.Close()

	data, err := yaml.Marshal(n.CreateSnapshot())
	if err != nil {
		return err
	}

	if _, err := f.Write(data); err != nil {
		return err
	}

	return nil
}

func Restore(file string) (*network.Network, error) {
	s, err := readYaml(file)
	if err != nil {
		return nil, err
	}

	b := network.NewNeuralNetworkBuilder(
		s.InputNeurons,
		activation.ByName(s.Activation),
	)

	for _, sl := range s.Layers {
		var neurons []network.Neuron
		for _, sn := range sl.Neurons {
			neurons = append(neurons, network.Neuron{
				Threshold: sn.Threshold,
				Weights:   sn.Weights,
			})
		}
		b.WithLayer(neurons)
	}

	return b.Build()
}

func readYaml(file string) (*network.Snapshot, error) {
	data, err := os.ReadFile(file)
	if err != nil {
		return nil, err
	}

	var n *network.Snapshot
	if err := yaml.Unmarshal(data, &n); err != nil {
		return nil, err
	}
	return n, nil
}
