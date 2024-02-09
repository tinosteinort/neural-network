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
	if ds.file != nil {
		file, err := os.Open(ds.Filename)
		if err != nil {
			return nil, err
		}
		ds.file = file

		s := bufio.NewScanner(ds.file)
		s.Split(bufio.ScanLines)
		ds.scanner = s
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
