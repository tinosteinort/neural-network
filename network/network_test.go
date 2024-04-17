package network_test

import (
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	"github.com/tinosteinort/neural-network/activation"
	"github.com/tinosteinort/neural-network/network"
)

var _ = Describe("Network", func() {

	Describe("update", func() {

		It("update network with step function", func() {

			n, err := network.NewBuilder(
				activation.StepFunction,
			).WithInputNeurons(
				2,
			).WithLayer(
				[]network.Neuron{{
					Weights:   []float64{0.0, 1.0},
					Threshold: 1.0,
				}, {
					Weights:   []float64{1.0, 1.0},
					Threshold: 2.0,
				}, {
					Weights:   []float64{1.0, 0.0},
					Threshold: 0.0,
				}},
			).WithLayer(
				[]network.Neuron{{
					Weights:   []float64{0.0, 1.0, 0.0},
					Threshold: 1.0,
				}, {
					Weights:   []float64{1.0, 0.0, 1.0},
					Threshold: 2.0,
				}},
			).Build()
			Expect(err).NotTo(HaveOccurred())

			err = n.Update([]float64{0.0, 0.0})
			Expect(err).To(BeNil())
			result := n.Output()
			Expect(result[0]).To(Equal(0.0))
			Expect(result[1]).To(Equal(0.0))

			err = n.Update([]float64{1.0, 0.0})
			Expect(err).To(BeNil())
			result = n.Output()
			Expect(result[0]).To(Equal(0.0))
			Expect(result[1]).To(Equal(0.0))

			err = n.Update([]float64{0.0, 1.0})
			Expect(err).To(BeNil())
			result = n.Output()
			Expect(result[0]).To(Equal(0.0))
			Expect(result[1]).To(Equal(1.0))

			err = n.Update([]float64{1.0, 1.0})
			Expect(err).To(BeNil())
			result = n.Output()
			Expect(result[0]).To(Equal(1.0))
			Expect(result[1]).To(Equal(1.0))
		})

		It("updates network with another function", func() {

			// https://www.taralino.de/courses/neuralnetwork2/activation
			// https://www.taralino.de/courses/neuralnetwork/weights

			n, err := network.NewBuilder(
				activation.SigmoidFunction,
			).WithInputNeurons(
				2,
			).WithLayer(
				[]network.Neuron{{
					Weights:   []float64{-0.2, 1.2},
					Threshold: 0.3,
				}, {
					Weights:   []float64{0.8, -0.7},
					Threshold: -0.1,
				}, {
					Weights:   []float64{1.0, 1.4},
					Threshold: 0.6,
				}},
			).WithLayer(
				[]network.Neuron{{
					Weights:   []float64{-0.5, 1.1, -1.3},
					Threshold: -1.7,
				}, {
					Weights:   []float64{0.9, -1.0, 0.6},
					Threshold: 0.4,
				}},
			).Build()
			Expect(err).NotTo(HaveOccurred())

			err = n.Update([]float64{0.5, 0.8})
			Expect(err).To(BeNil())
			result := n.Output()
			Expect(result[0]).To(Equal(0.7230846231041326))
			Expect(result[1]).To(Equal(0.532152159729842))
		})

		It("update network saves also input values", func() {

			n, err := network.NewBuilder(
				activation.StepFunction,
			).WithInputNeurons(
				2,
			).WithLayer(
				[]network.Neuron{{
					Weights:   []float64{0.0, 1.0},
					Threshold: 1.0,
				}, {
					Weights:   []float64{1.0, 1.0},
					Threshold: 2.0,
				}},
			).Build()
			Expect(err).NotTo(HaveOccurred())

			err = n.Update([]float64{0.1, 0.4})
			Expect(err).To(BeNil())

			s := n.CreateSnapshot()
			Expect(s.Input).To(Equal([]float64{0.1, 0.4}))
		})
	})

	It("returns valid output even network was not updated", func() {

		n, err := network.NewBuilder(
			activation.StepFunction,
		).WithInputNeurons(
			2,
		).WithLayer([]network.Neuron{
			{Weights: []float64{0, 0}},
			{Weights: []float64{0, 0}},
		}).Build()
		Expect(err).NotTo(HaveOccurred())

		result := n.Output()
		Expect(result).NotTo(BeNil())
		Expect(len(result)).To(Equal(2))
		Expect(result[0]).To(Equal(float64(0)))
		Expect(result[1]).To(Equal(float64(0)))
	})

})
