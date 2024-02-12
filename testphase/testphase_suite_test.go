package testphase_test

import (
	"testing"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
)

func TestTestphase(t *testing.T) {
	RegisterFailHandler(Fail)
	RunSpecs(t, "Testphase Suite")
}
