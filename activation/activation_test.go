package activation_test

import (
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	"github.com/tinosteionrt/neural-network/activation"
	"github.com/tinosteionrt/neural-network/network"
)

var _ = Describe("Activation", func() {

	BeforeEach(func() {
		activation.WithCustom([]network.Activation{})
	})

	It("should not find function", func() {
		Expect(true).To(Equal(false))
	})

	It("should find custom function", func() {
		Expect(true).To(Equal(false))
	})

	It("should not be possible to register existing function", func() {
		Expect(true).To(Equal(false))
	})
})
