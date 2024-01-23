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

		_, err := network.NewNeuralNetworkBuilder(
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

		Expect(err).To(BeNil())
	})

	It("should not create new network because of invalid number of input neurons", func() {

		// https://www.taralino.de/courses/neuralnetwork1/network
		n, err := network.NewNeuralNetworkBuilder(
			2,
			activation.StepFunction,
		).WithLayer(
			[]network.Neuron{{
				Weights:   []int{0, 1},
				Threshold: 1,
			}},
		).Build()

		Expect(err).To(BeNil())
		_, err = n.Run([]int{1})
		Expect(err).To(Equal(errors.New("input values does not match input neuron count")))
	})

	It("update network", func() {

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

		Expect(err).To(BeNil())

		r, err := n.Run([]int{0, 0})
		Expect(err).To(BeNil())
		Expect(r[0]).To(Equal(0))
		Expect(r[1]).To(Equal(0))

		r, err = n.Run([]int{1, 0})
		Expect(err).To(BeNil())
		Expect(r[0]).To(Equal(0))
		Expect(r[1]).To(Equal(0))

		r, err = n.Run([]int{0, 1})
		Expect(err).To(BeNil())
		Expect(r[0]).To(Equal(0))
		Expect(r[1]).To(Equal(1))

		r, err = n.Run([]int{1, 1})
		Expect(err).To(BeNil())
		Expect(r[0]).To(Equal(1))
		Expect(r[1]).To(Equal(1))
	})

	Describe("NeuralNetworkBuilder", func() {

		It("needs at least one layer", func() {

			_, err := network.NewNeuralNetworkBuilder(
				2,
				activation.StepFunction,
			).Build()

			Expect(err).To(Equal(errors.New("no layer defined")))
		})

		It("weight-count need to match amount input neurons", func() {

			_, err := network.NewNeuralNetworkBuilder(
				2,
				activation.StepFunction,
			).WithLayer(
				[]network.Neuron{{
					Weights: []int{0},
				}},
			).Build()

			Expect(err).To(Equal(errors.New("weight count does not match input neuron count")))
		})

		It("weight-count need to match amount of previous layers neurons", func() {

			_, err := network.NewNeuralNetworkBuilder(
				2,
				activation.StepFunction,
			).WithLayer(
				[]network.Neuron{{
					Weights: []int{0, 0},
				}, {
					Weights: []int{0, 0},
				}, {
					Weights: []int{0, 0},
				}},
			).WithLayer(
				[]network.Neuron{{
					Weights: []int{0, 0, 0, 0},
				}, {
					Weights: []int{0, 0, 0, 0},
				}},
			).Build()

			Expect(err).To(Equal(errors.New("weight count does not match previous layer[0] neuron count")))
		})
	})
})
