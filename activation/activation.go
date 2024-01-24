package activation

import (
	"fmt"
	"github.com/tinosteionrt/neural-network/network"
)

var builtin = []network.Activation{
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

var custom []network.Activation

func ByName(n string) (network.Activation, error) {
	for _, f := range builtin {
		if f.Name == n {
			return f, nil
		}
	}
	return network.Activation{}, fmt.Errorf("activation function %s not found", n)
}

func WithCustom(functions []network.Activation) {
	custom = functions
}
