package dump_test

import (
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	"github.com/tinosteionrt/neural-network/activation"
	"github.com/tinosteionrt/neural-network/dump"
	"github.com/tinosteionrt/neural-network/network"
)

var _ = Describe("Dump", func() {

	It("store and restore network", func() {

		n, err := network.NewNeuralNetworkBuilder(
			2,
			activation.StepFunction,
		).WithLayer(
			[]network.Neuron{{
				Weights:   []int{0, 1},
				Threshold: 1,
			}, {
				Weights:   []int{1, 1},
				Threshold: 2,
			}, {
				Weights:   []int{1, 0},
				Threshold: 0,
			}},
		).WithLayer(
			[]network.Neuron{{
				Weights:   []int{0, 1, 0},
				Threshold: 1,
			}, {
				Weights:   []int{1, 0, 1},
				Threshold: 2,
			}},
		).Build()

		Expect(err).NotTo(HaveOccurred())
		Expect(dump.Store(n, "test.nn")).NotTo(HaveOccurred())
	})
})
