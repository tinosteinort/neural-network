package testphase

import (
	"github.com/tinosteionrt/neural-network/dataset"
	"github.com/tinosteionrt/neural-network/network"
)

type ConfusionMatrix struct {
	Results []Evaluation
}

type Evaluation struct {
	Expected []int
	Actual   []int
}

func Execute(n *network.Network, ds dataset.DataSet) (ConfusionMatrix, error) {
	var results []Evaluation

	for ok := ds.HasNext(); ok; ok = ds.HasNext() {

		record, err := ds.Next()
		if err != nil {
			return ConfusionMatrix{}, err
		}

		if err := n.Update(record.Input); err != nil {
			return ConfusionMatrix{}, err
		}

		results = append(results, Evaluation{
			Expected: record.Result,
			Actual:   normalise(n.Output()),
		})
	}

	return ConfusionMatrix{
		Results: results,
	}, nil
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
