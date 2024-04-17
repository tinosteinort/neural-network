package network_test

import (
	"errors"
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	"github.com/tinosteinort/neural-network/activation"
	"github.com/tinosteinort/neural-network/network"
)

var _ = Describe("Builder", func() {

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
		err = n.Update([]float64{1.0})
		Expect(err).To(Equal(errors.New("input values does not match input neuron count")))
	})

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
			3,
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
