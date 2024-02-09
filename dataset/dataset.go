package dataset

type DataSet interface {
	Next() (*Record, error)
	Close() error
}

type Record struct {
	Input  []float64
	Result []float64
}

func NewInMemoryDataSet(records []Record) DataSet {
	return &inMemoryDataSet{
		Records: records,
	}
}

func NewFileDataSet(filename string) (DataSet, error) {
	return &fileDataSet{
		Filename: filename,
	}, nil
}
