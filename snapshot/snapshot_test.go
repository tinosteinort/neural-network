package snapshot_test

import (
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	"github.com/tinosteionrt/neural-network/activation"
	"github.com/tinosteionrt/neural-network/network"
	"github.com/tinosteionrt/neural-network/snapshot"
)

var _ = Describe("Snapshot", func() {

	It("should store network", func() {

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
		Expect(snapshot.Store(n, "test.nn")).NotTo(HaveOccurred())
	})
})
