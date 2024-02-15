package testphase

import (
	"github.com/tinosteionrt/neural-network/dataset"
	"github.com/tinosteionrt/neural-network/network"
)

type ConfusionMatrix struct {
	Results []Result
}

type Result struct {
	Expected []int
	Actual   []int
}

func Execute(n *network.Network, ds dataset.DataSet) (ConfusionMatrix, error) {
	var result []Result

	for ok := true; ok; ok = ds.HasNext() {

		record, err := ds.Next()
		if err != nil {
			return ConfusionMatrix{}, err
		}

		if err := n.Update(record.Input); err != nil {
			return ConfusionMatrix{}, err
		}

		result = append(result, Result{
			//Expected: record.Result, // TODO needs to be []int
		})
	}

	return ConfusionMatrix{}, nil
}
