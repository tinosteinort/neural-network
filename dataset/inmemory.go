package dataset

import "errors"

type inMemoryDataSet struct {
	Records []Record
	index   int
}

func (ds *inMemoryDataSet) HasNext() bool {
	return false
}

func (ds *inMemoryDataSet) Next() (*Record, error) {
	if ds.index >= len(ds.Records) {
		return nil, errors.New("no records left in dataset")
	}
	d := &ds.Records[ds.index]
	ds.index += 1
	return d, nil
}

func (ds *inMemoryDataSet) Close() error {
	return nil
}
