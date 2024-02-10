package dataset

type DataSet interface {
	HasNext() bool
	Next() (*Record, error)
	Close() error
}

type Record struct {
	Input  []float64
	Result []float64
}
