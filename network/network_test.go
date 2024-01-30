package network_test

import (
	"errors"
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	"github.com/tinosteionrt/neural-network/activation"
	"github.com/tinosteionrt/neural-network/network"
)

var _ = Describe("Network", func() {

	It("should create new network", func() {

		_, err := network.NewBuilder(
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

		Expect(err).To(BeNil())
	})

	It("should not create new network because of invalid number of input neurons", func() {

		// https://www.taralino.de/courses/neuralnetwork1/network
		n, err := network.NewBuilder(
			activation.StepFunction,
		).WithInputNeurons(
			2,
		).WithLayer(
			[]network.Neuron{{
				Weights:   []float64{0.0, 1.0},
				Threshold: 1.0,
			}},
		).Build()

		Expect(err).To(BeNil())
		_, err = n.Run([]float64{1.0})
		Expect(err).To(Equal(errors.New("input values does not match input neuron count")))
	})

	It("run network with step function", func() {

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

		r, err := n.Run([]float64{0.0, 0.0})
		Expect(err).To(BeNil())
		Expect(r[0]).To(Equal(0.0))
		Expect(r[1]).To(Equal(0.0))

		r, err = n.Run([]float64{1.0, 0.0})
		Expect(err).To(BeNil())
		Expect(r[0]).To(Equal(0.0))
		Expect(r[1]).To(Equal(0.0))

		r, err = n.Run([]float64{0.0, 1.0})
		Expect(err).To(BeNil())
		Expect(r[0]).To(Equal(0.0))
		Expect(r[1]).To(Equal(1.0))

		r, err = n.Run([]float64{1.0, 1.0})
		Expect(err).To(BeNil())
		Expect(r[0]).To(Equal(1.0))
		Expect(r[1]).To(Equal(1.0))
	})

	It("run network with sigmoid function", func() {

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

		r, err := n.Run([]float64{0.5, 0.8})
		Expect(err).To(BeNil())
		Expect(r[0]).To(Equal(0.7230846231041326))
		Expect(r[1]).To(Equal(0.532152159729842))
	})

	Describe("Builder", func() {

		It("needs input neurons - by number", func() {

			_, err := network.NewBuilder(
				activation.StepFunction,
			).WithInputNeurons(
				2,
			).WithLayer(
				[]network.Neuron{{
					Weights: []float64{0.1, 0.2},
				}},
			).Build()

			Expect(err).NotTo(HaveOccurred())
		})

		It("needs input neurons - with values", func() {

			_, err := network.NewBuilder(
				activation.StepFunction,
			).WithInput([]float64{
				1.0, 2.0,
			},
			).WithLayer(
				[]network.Neuron{{
					Weights: []float64{0.1, 0.2},
				}},
			).Build()

			Expect(err).NotTo(HaveOccurred())
		})

		It("should not build network because of missing input", func() {

			_, err := network.NewBuilder(
				activation.StepFunction,
			).WithLayer(
				[]network.Neuron{{
					Weights: []float64{0.1, 0.2},
				}},
			).Build()

			Expect(err).To(Equal(errors.New("no input specified")))
		})

		It("needs at least one layer", func() {

			_, err := network.NewBuilder(
				activation.StepFunction,
			).WithInputNeurons(
				2,
			).Build()

			Expect(err).To(Equal(errors.New("no layer defined")))
		})

		It("weight-count need to match amount input neurons", func() {

			_, err := network.NewBuilder(
				activation.StepFunction,
			).WithInputNeurons(
				2,
			).WithLayer(
				[]network.Neuron{{
					Weights: []float64{0.0},
				}},
			).Build()

			Expect(err).To(Equal(errors.New("weight count does not match input neuron count")))
		})

		It("weight-count need to match amount of previous layers neurons", func() {

			_, err := network.NewBuilder(
				activation.StepFunction,
			).WithInputNeurons(
				2,
			).WithLayer(
				[]network.Neuron{{
					Weights: []float64{0.0, 0.0},
				}, {
					Weights: []float64{0.0, 0.0},
				}, {
					Weights: []float64{0.0, 0.0},
				}},
			).WithLayer(
				[]network.Neuron{{
					Weights: []float64{0.0, 0.0, 0.0, 0.0},
				}, {
					Weights: []float64{0.0, 0.0, 0.0, 0.0},
				}},
			).Build()

			Expect(err).To(Equal(errors.New("weight count does not match previous layer[0] neuron count")))
		})
	})

})
