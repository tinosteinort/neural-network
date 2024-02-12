package testphase

import (
	"errors"
	"github.com/tinosteionrt/neural-network/dataset"
	"github.com/tinosteionrt/neural-network/network"
)

type ConfusionMatrix struct {
}

func Execute(n *network.Network, ds dataset.DataSet) (ConfusionMatrix, error) {
	return ConfusionMatrix{}, errors.New("not implemented")
}
