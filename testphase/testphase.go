package testphase

import (
	"fmt"
	"github.com/tinosteinort/neural-network/dataset"
	"github.com/tinosteinort/neural-network/network"
	"reflect"
)

type Result struct {
	Overall     int
	Correct     int
	Wrong       int
	SuccessRate float64
}

func Execute(n *network.Network, ds dataset.DataSet) (r Result, err error) {
	for ok := ds.HasNext(); ok; ok = ds.HasNext() {

		record, err := ds.Next()
		if err != nil {
			return Result{}, err
		}

		if err := n.Update(record.Input); err != nil {
			return Result{}, err
		}

		result := record.Result
		output := normalise(n.Output())

		r.Overall++
		if reflect.DeepEqual(result, output) {
			r.Correct++
		} else {
			r.Wrong++
		}
	}
	r.SuccessRate = (float64(r.Correct) / float64(r.Overall)) * 100

	return r, nil
}

func normalise(floats []float64) []int {
	maxValue := -1.0
	indexOfMaxValue := -1

	for i, v := range floats {
		if v > maxValue {
			maxValue = v
			indexOfMaxValue = i
		}
	}
	result := make([]int, len(floats))
	result[indexOfMaxValue] = 1
	return result
}

func (r Result) String() string {
	return fmt.Sprintf("correct: %.2f %% (%d/%d)", r.SuccessRate, r.Correct, r.Overall)
}
