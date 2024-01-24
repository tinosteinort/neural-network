package activation

import (
	"fmt"
	"github.com/tinosteionrt/neural-network/network"
)

var functions = []network.Activation{
	{
		Name: "step",
		Run: func(input []int, n network.Neuron) int {
			value := 0
			for wi, w := range n.Weights {
				value = value + (input[wi] * w)
			}
			if value-n.Threshold < 0 {
				return 0
			} else {
				return 1
			}
		},
	},
}

func ByName(n string) (network.Activation, error) {
	for _, f := range functions {
		if f.Name == n {
			return f, nil
		}
	}
	return network.Activation{}, fmt.Errorf("activation function %s not found", n)
}
