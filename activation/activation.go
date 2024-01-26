package activation

import (
	"fmt"
	"github.com/tinosteionrt/neural-network/network"
	"math"
)

var StepFunction = network.Activation{
	Name: "step",
	Run: func(input []float64, n network.Neuron) float64 {
		var t = calculateValue(input, n)
		if t < 0 {
			return 0.0
		} else {
			return 1.0
		}
	},
}

var SigmoidFunction = network.Activation{
	Name: "sigmoid",
	Run: func(input []float64, n network.Neuron) float64 {
		var t = calculateValue(input, n)
		return 1 / (1 * math.Exp(-t))
	},
}

func calculateValue(input []float64, n network.Neuron) float64 {
	var value float64 = 0
	for wi, w := range n.Weights {
		value = value + (input[wi] * w)
	}
	return value - n.Threshold
}

func WithCustom(functions []network.Activation) error {
	for _, cf := range functions {
		for _, bf := range builtin {
			if cf.Name == bf.Name {
				return fmt.Errorf("cannot override builtin function: %s", bf.Name)
			}
		}
	}
	custom = functions
	return nil
}

var custom []network.Activation

var builtin = []network.Activation{
	StepFunction,
	SigmoidFunction,
}

func ByName(n string) (network.Activation, error) {
	for _, f := range builtin {
		if f.Name == n {
			return f, nil
		}
	}
	for _, f := range custom {
		if f.Name == n {
			return f, nil
		}
	}
	return network.Activation{}, fmt.Errorf("activation function not found: %s", n)
}
