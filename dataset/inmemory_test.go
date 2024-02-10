package dataset_test

import (
	"errors"
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"

	"github.com/tinosteionrt/neural-network/dataset"
)

var _ = Describe("Inmemory", func() {

	It("gets record from inmemory dataset", func() {

		ds := dataset.NewInMemoryDataSet(
			[]dataset.Record{{
				Input:  []float64{0.4, 0.1},
				Result: []float64{0, 1},
			}, {
				Input:  []float64{0.2, 0.3},
				Result: []float64{1, 0},
			}},
		)

		ds.HasNext()
		_, err := ds.Next()

		ds.HasNext()
		r, err := ds.Next()
		Expect(err).NotTo(HaveOccurred())
		Expect(r).NotTo(BeNil())
		Expect(r).To(Equal(&dataset.Record{
			Input:  []float64{0.2, 0.3},
			Result: []float64{1, 0},
		}))
	})

	It("expects calling HasNext before Next", func() {

		ds := dataset.NewInMemoryDataSet(
			[]dataset.Record{{
				Input:  []float64{0.2},
				Result: []float64{1, 0},
			}},
		)
		defer ds.Close()

		r, err := ds.Next()
		Expect(err).To(Equal(errors.New("HasNext() was not called before Next()")))
		Expect(r).To(BeNil())
	})

	It("checks if next record exist", func() {

		ds := dataset.NewInMemoryDataSet(
			[]dataset.Record{{
				Input:  []float64{0.2},
				Result: []float64{1, 0},
			}, {
				Input:  []float64{0.6},
				Result: []float64{0, 1},
			}},
		)
		defer ds.Close()

		Expect(ds.HasNext()).To(BeTrue())
		Expect(ds.HasNext()).To(BeTrue())
		Expect(ds.HasNext()).To(BeFalse())
	})

	It("returns error if no next record exist but next is called", func() {

		ds := dataset.NewInMemoryDataSet(
			[]dataset.Record{{
				Input:  []float64{0.2},
				Result: []float64{1, 0},
			}},
		)
		defer ds.Close()

		Expect(ds.HasNext()).To(BeTrue())
		Expect(ds.HasNext()).To(BeFalse())

		r, err := ds.Next()
		Expect(err).To(Equal(errors.New("no records left in dataset")))
		Expect(r).To(BeNil())
	})
})
