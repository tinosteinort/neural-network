package dataset_test

import (
	"errors"
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	"github.com/tinosteionrt/neural-network/dataset"
)

var _ = Describe("File", func() {

	It("reads records", func() {

		ds, err := dataset.NewFileDataSet("example.ds")
		Expect(err).NotTo(HaveOccurred())
		Expect(ds).NotTo(BeNil())
		defer ds.Close()

		ds.HasNext()
		r, err := ds.Next()
		Expect(err).NotTo(HaveOccurred())
		Expect(r).NotTo(BeNil())
		Expect(r).To(Equal(&dataset.Record{
			Input:  []float64{0.1, 0.2},
			Result: []float64{1, 0},
		}))
	})

	It("expects calling HasNext before Next", func() {

		ds, err := dataset.NewFileDataSet("example.ds")
		Expect(err).NotTo(HaveOccurred())
		defer ds.Close()

		r, err := ds.Next()
		Expect(err).To(Equal(errors.New("HasNext() was not called before Next()")))
		Expect(r).To(BeNil())
	})

	It("checks if next record exist", func() {

		ds, err := dataset.NewFileDataSet("example.ds")
		Expect(err).NotTo(HaveOccurred())
		defer ds.Close()

		Expect(ds.HasNext()).To(BeTrue())
		Expect(ds.HasNext()).To(BeTrue())
		Expect(ds.HasNext()).To(BeFalse())
	})

	It("returns error if no next record exist but next is called", func() {

		ds, err := dataset.NewFileDataSet("example.ds")
		Expect(err).NotTo(HaveOccurred())
		defer ds.Close()

		Expect(ds.HasNext()).To(BeTrue())
		Expect(ds.HasNext()).To(BeTrue())
		Expect(ds.HasNext()).To(BeFalse())

		r, err := ds.Next()
		Expect(err).To(Equal(errors.New("no records left in dataset")))
		Expect(r).To(BeNil())
	})
})
