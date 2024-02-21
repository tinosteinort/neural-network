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

type Snapshot struct {
	Activation string          `yaml:"activation"`
	Input      []float64       `yaml:"input,flow"`
	Layers     []SnapshotLayer `yaml:"layers"`
}

type SnapshotLayer struct {
	Neurons []SnapshotNeuron `yaml:"neuron"`
}

type SnapshotNeuron struct {
	Threshold float64   `yaml:"threshold"`
	Weights   []float64 `yaml:"weights,flow"`
	Value     float64   `yaml:"value"`
}
