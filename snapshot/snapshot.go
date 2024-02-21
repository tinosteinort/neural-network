package snapshot

import (
	"github.com/tinosteionrt/neural-network/activation"
	"github.com/tinosteionrt/neural-network/network"
	"gopkg.in/yaml.v3"
	"os"
)

func Store(n network.Network, file string) error {

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

func Restore(file string) (network.Network, error) {
	s, err := readYaml(file)
	if err != nil {
		return nil, err
	}

	af, err := activation.ByName(s.Activation)
	if err != nil {
		return nil, err
	}

	b := network.NewBuilder(
		af,
	)

	b.WithInput(s.Input)

	for _, sl := range s.Layers {
		b.WithLayer(restoreLayer(sl))
	}

	return b.Build()
}

func restoreLayer(l network.SnapshotLayer) []network.Neuron {
	var neurons []network.Neuron
	for _, n := range l.Neurons {
		neurons = append(neurons, network.Neuron{
			Threshold: n.Threshold,
			Weights:   n.Weights,
			Value:     n.Value,
		})
	}
	return neurons
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
