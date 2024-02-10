package dataset

import (
	"bufio"
	"errors"
	"os"
	"strconv"
	"strings"
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
		return nil, errors.New("no records left in dataset")
	}
	l := ds.scanner.Text()

	return asRecord(&l)
}

func asRecord(s *string) (*Record, error) {
	input, result, err := parseLine(s)
	if err != nil {
		return nil, err
	}

	return &Record{
		Input:  input,
		Result: result,
	}, nil
}

func parseLine(s *string) (input []float64, result []float64, err error) {
	segments := strings.Split(*s, ";")

	input, err = asFloats(&segments[0])
	if err != nil {
		return nil, nil, err
	}

	result, err = asFloats(&segments[1])
	if err != nil {
		return nil, nil, err
	}

	return input, result, nil
}

func asFloats(s *string) ([]float64, error) {
	var floats []float64
	for _, v := range strings.Split(*s, ",") {
		f, err := strconv.ParseFloat(v, 64)
		if err != nil {
			return nil, err
		}
		floats = append(floats, f)
	}
	return floats, nil
}

func (ds *fileDataSet) Close() error {
	return ds.file.Close()
}
