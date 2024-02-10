package dataset

import "errors"

type inMemoryDataSet struct {
	Records []Record
	index   int
	hasNext *bool
}

func NewInMemoryDataSet(records []Record) DataSet {
	return &inMemoryDataSet{
		Records: records,
		index:   -1,
	}
}

func (ds *inMemoryDataSet) HasNext() bool {
	ds.index += 1
	hasNext := ds.index < len(ds.Records)
	ds.hasNext = &hasNext
	return hasNext
}

func (ds *inMemoryDataSet) Next() (*Record, error) {
	if ds.hasNext == nil {
		return nil, errors.New("HasNext() was not called before Next()")
	}

	if !*ds.hasNext {
		return nil, errors.New("no records left in dataset")
	}

	d := &ds.Records[ds.index]
	ds.hasNext = nil

	return d, nil
}

func (ds *inMemoryDataSet) Close() error {
	return nil
}
