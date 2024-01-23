package activation

import "github.com/tinosteionrt/neural-network/network"

var StepFunction = network.Activation{
	Name: "step",
	Function: func(input []int, n network.Neuron) int {
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
}
