package dataset

import (
	"bufio"
	"errors"
	"os"
)

type fileDataSet struct {
	Filename string
	file     *os.File
	scanner  *bufio.Scanner
}

func (ds *fileDataSet) Next() (*Record, error) {
	if ds.file == nil {
		file, err := os.Open(ds.Filename)
		if err != nil {
			return nil, err
		}
		ds.file = file

		ds.scanner = bufio.NewScanner(ds.file)
		ds.scanner.Split(bufio.ScanLines)
	}

	if ds.scanner == nil {
		return nil, errors.New("scanner ist null")
	}

	hasNext := ds.scanner.Scan()
	if !hasNext {
		return nil, nil
	}
	ds.scanner.Text()

	return nil, nil
}

func (ds *fileDataSet) Close() error {
	return ds.file.Close()
}
