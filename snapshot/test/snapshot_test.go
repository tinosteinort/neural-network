package test_test

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

		n, err := network.NewBuilder(
			activation.StepFunction,
		).WithInput([]float64{
			1.0, 2.0,
		}).WithLayer(
			[]network.Neuron{{
				Weights:   []float64{0.0, 1.0},
				Threshold: 1.0,
				Value:     50.0,
			}, {
				Weights:   []float64{1.0, 1.0},
				Threshold: 2.0,
				Value:     60.0,
			}, {
				Weights:   []float64{1.0, 0.0},
				Threshold: 0.0,
				Value:     70.0,
			}},
		).WithLayer(
			[]network.Neuron{{
				Weights:   []float64{0.0, 1.0, 0.0},
				Threshold: 1.0,
				Value:     80.0,
			}, {
				Weights:   []float64{1.0, 0.0, 1.0},
				Threshold: 2.0,
				Value:     90.0,
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

	It("should restore network", func() {

		n, err := snapshot.Restore("example1.nn")
		Expect(err).NotTo(HaveOccurred())
		Expect(n).NotTo(BeNil())

		expected := network.Snapshot{
			Activation: "step",
			Input: []float64{
				1.0, 2.0, 3.0,
			},
			Layers: []network.SnapshotLayer{
				{
					Neurons: []network.SnapshotNeuron{
						{
							Threshold: 1.0,
							Weights:   []float64{0.0, 1.0, 1.0},
							Value:     0.5,
						}, {
							Threshold: 2.0,
							Weights:   []float64{1.0, 1.0, 0.0},
							Value:     0.6,
						}, {
							Threshold: 3.0,
							Weights:   []float64{1.0, 0.0, 0.0},
							Value:     0.7,
						}, {
							Threshold: 3.0,
							Weights:   []float64{0.0, 0.0, 1.0},
							Value:     0.8,
						},
					},
				}, {
					Neurons: []network.SnapshotNeuron{
						{
							Threshold: 1.0,
							Weights:   []float64{0.0, 1.0, 0.0, 1.0},
							Value:     0.9,
						}, {
							Threshold: 2.0,
							Weights:   []float64{1.0, 0.0, 1.0, 0.0},
							Value:     1.1,
						},
					},
				},
			},
		}
		Expect(n.CreateSnapshot()).To(Equal(&expected))
	})
})
