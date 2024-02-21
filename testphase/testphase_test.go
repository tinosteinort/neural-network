package testphase_test

import (
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	"github.com/tinosteionrt/neural-network/activation"
	"github.com/tinosteionrt/neural-network/dataset"
	"github.com/tinosteionrt/neural-network/network"
	"github.com/tinosteionrt/neural-network/testphase"
)

var _ = Describe("Testphase", func() {

	It("builds specific network", func() {
		// This test just exist to create a network which behaves
		// in a defined way for the given parameter, to test
		// the confusion matix

		// two outputs:
		// if input < 0.5  than result is 1, 0
		// if input >= 0.5 than result is 0, 1

		n, err := network.NewBuilder(activation.StepFunction).
			WithInputNeurons(1).
			WithLayer(
				[]network.Neuron{{
					Weights:   []float64{-1},
					Threshold: -0.4,
				}, {
					Weights:   []float64{1},
					Threshold: 0.5,
				}},
			).Build()
		Expect(err).NotTo(HaveOccurred())

		_ = n.Update([]float64{0.0})
		r := n.Output()
		Expect(r[0]).To(Equal(float64(1)))
		Expect(r[1]).To(Equal(float64(0)))

		_ = n.Update([]float64{0.1})
		r = n.Output()
		Expect(r[0]).To(Equal(float64(1)))
		Expect(r[1]).To(Equal(float64(0)))

		_ = n.Update([]float64{0.2})
		r = n.Output()
		Expect(r[0]).To(Equal(float64(1)))
		Expect(r[1]).To(Equal(float64(0)))

		_ = n.Update([]float64{0.3})
		r = n.Output()
		Expect(r[0]).To(Equal(float64(1)))
		Expect(r[1]).To(Equal(float64(0)))

		_ = n.Update([]float64{0.4})
		r = n.Output()
		Expect(r[0]).To(Equal(float64(1)))
		Expect(r[1]).To(Equal(float64(0)))

		_ = n.Update([]float64{0.5})
		r = n.Output()
		Expect(r[0]).To(Equal(float64(0)))
		Expect(r[1]).To(Equal(float64(1)))

		_ = n.Update([]float64{0.6})
		r = n.Output()
		Expect(r[0]).To(Equal(float64(0)))
		Expect(r[1]).To(Equal(float64(1)))

		_ = n.Update([]float64{0.7})
		r = n.Output()
		Expect(r[0]).To(Equal(float64(0)))
		Expect(r[1]).To(Equal(float64(1)))

		_ = n.Update([]float64{0.8})
		r = n.Output()
		Expect(r[0]).To(Equal(float64(0)))
		Expect(r[1]).To(Equal(float64(1)))

		_ = n.Update([]float64{0.9})
		r = n.Output()
		Expect(r[0]).To(Equal(float64(0)))
		Expect(r[1]).To(Equal(float64(1)))
	})

	Describe("evaluate result", func() {

		var n network.Network

		BeforeEach(func() {
			var err error
			n, err = network.NewBuilder(activation.StepFunction).
				WithInputNeurons(1).
				WithLayer(
					[]network.Neuron{{
						Weights:   []float64{-1},
						Threshold: -0.4,
					}, {
						Weights:   []float64{1},
						Threshold: 0.5,
					}},
				).Build()
			Expect(err).NotTo(HaveOccurred())
		})

		It("counts and calculates result", func() {

			knownResults := dataset.NewInMemory(
				[]dataset.Record{{
					Input: []float64{0.1}, Result: []int{1, 0},
				}, {
					Input: []float64{0.3}, Result: []int{1, 0},
				}, {
					Input: []float64{0.4}, Result: []int{1, 0},
				}, {
					Input: []float64{0.5}, Result: []int{0, 1},
				}, {
					Input: []float64{0.9}, Result: []int{1, 0}, // wrong on purpose
				}},
			)

			r, err := testphase.Execute(n, knownResults)
			Expect(err).NotTo(HaveOccurred())
			Expect(r.Overall).To(Equal(5))
			Expect(r.Correct).To(Equal(4))
			Expect(r.Wrong).To(Equal(1))
			Expect(r.SuccessRate).To(Equal(float64(80)))
		})
	})

	It("checks string representation to result", func() {

		r := testphase.Result{
			Overall:     10,
			Correct:     6,
			Wrong:       4,
			SuccessRate: float64(60),
		}

		Expect(r.String()).To(Equal("correct: 60.00 % (6/10)"))
	})
})
