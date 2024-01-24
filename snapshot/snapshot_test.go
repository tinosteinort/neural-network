package snapshot_test

import (
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	"github.com/tinosteionrt/neural-network/activation"
	"github.com/tinosteionrt/neural-network/network"
	"github.com/tinosteionrt/neural-network/snapshot"
	"os"
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
		Expect(snapshot.Store(n, "snapshot1.nn")).NotTo(HaveOccurred())

		snapshot1, err := os.ReadFile("snapshot1.nn")
		Expect(err).NotTo(HaveOccurred())
		expected, err := os.ReadFile("snapshot1_expected.nn")
		Expect(err).NotTo(HaveOccurred())
		Expect(string(snapshot1)).To(Equal(string(expected)))
	})
})
