package activation_test

import (
	"errors"
	"fmt"
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	"github.com/tinosteionrt/neural-network/activation"
	"github.com/tinosteionrt/neural-network/network"
)

var _ = Describe("Activation", func() {

	BeforeEach(func() {
		Expect(activation.WithCustom([]network.Activation{})).NotTo(HaveOccurred())
	})

	Describe("byName", func() {

		It("should not find function", func() {
			_, err := activation.ByName("does-not-exist")
			Expect(err).To(Equal(fmt.Errorf("activation function not found: does-not-exist")))
		})

		It("should find builtin function", func() {
			f, err := activation.ByName("step")
			Expect(f).NotTo(BeNil())
			Expect(err).NotTo(HaveOccurred())
			Expect(f.Name).To(Equal("step"))
		})
	})

	Describe("register", func() {
		It("should register and find custom function", func() {
			Expect(activation.WithCustom([]network.Activation{
				{
					Name: "custom-implementation",
					Run: func(input []float64, n network.Neuron) float64 {
						return 0
					},
				},
			})).NotTo(HaveOccurred())

			f, err := activation.ByName("custom-implementation")
			Expect(f).NotTo(BeNil())
			Expect(err).NotTo(HaveOccurred())
			Expect(f.Name).To(Equal("custom-implementation"))
		})

		It("should not be possible to override existing builtin function", func() {
			err := activation.WithCustom([]network.Activation{
				{
					Name: "step",
					Run: func(input []float64, n network.Neuron) float64 {
						return 0
					},
				},
			})
			Expect(err).To(Equal(errors.New("cannot override builtin function: step")))
		})
	})

	Describe("sigmoid", func() {
		It("should return correct value", func() {
			f, err := activation.ByName("sigmoid")
			Expect(err).NotTo(HaveOccurred())

			v := f.Run(
				[]float64{0.5, 0.8},
				network.Neuron{
					Threshold: 0.3,
					Weights:   []float64{-0.2, 1.2},
				},
			)
			Expect(v).To(Equal(0.6364525402815664))
		})
	})
})
