package network_test

import (
	"errors"
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	"github.com/tinosteionrt/neural-network/activation"
	"github.com/tinosteionrt/neural-network/network"
)

var _ = Describe("Builder", func() {

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
