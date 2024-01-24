package snapshot

import (
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

func Restore() (*network.Network, error) {
	return nil, nil
}
