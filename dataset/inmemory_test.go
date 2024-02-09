package dataset_test

import (
	"errors"
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"

	"github.com/tinosteionrt/neural-network/dataset"
)

var _ = Describe("Inmemory", func() {

	It("iterate over inmemory dataset", func() {

		ds := dataset.NewInMemoryDataSet(
			[]dataset.Record{{
				Input:  []float64{0.2},
				Result: []float64{1, 0},
			}, {
				Input:  []float64{0.6},
				Result: []float64{0, 1},
			}},
		)

		r1, err := ds.Next()
		Expect(err).NotTo(HaveOccurred())
		Expect(r1).NotTo(BeNil())
		Expect(r1).To(Equal(&dataset.Record{
			Input:  []float64{0.2},
			Result: []float64{1, 0},
		}))

		r2, err := ds.Next()
		Expect(err).NotTo(HaveOccurred())
		Expect(r2).NotTo(BeNil())
		Expect(r2).To(Equal(&dataset.Record{
			Input:  []float64{0.6},
			Result: []float64{0, 1},
		}))

		r3, err := ds.Next()
		Expect(err).To(Equal(errors.New("no records left in dataset")))
		Expect(r3).To(BeNil())
	})
})
