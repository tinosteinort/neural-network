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

	It("should register and find custom function", func() {
		Expect(activation.WithCustom([]network.Activation{
			{
				Name: "custom-implementation",
				Run: func(input []int, n network.Neuron) int {
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
				Run: func(input []int, n network.Neuron) int {
					return 0
				},
			},
		})
		Expect(err).To(Equal(errors.New("cannot override builtin function: step")))
	})
})
