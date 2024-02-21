package network

type Network interface {
	Update(input []float64) error
	Output() []float64
	CreateSnapshot() *Snapshot
}

type Neuron struct {
	Weights   []float64
	Threshold float64
	Value     float64
}

type Activation struct {
	Name string
	Run  func(input []float64, n Neuron) float64
}
