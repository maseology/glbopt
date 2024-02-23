package glbopt

import (
	"math"
	"math/rand"
)

const pertubationValue = .2 // Tolson: "I never change this value"

// DDS - The Dynamically Dimensioned Search algorithm
// Tolson B.A. and C.A. Shoemaker, 2007. Dynamically dimensioned search algorithm for computationally efficient watershed model calibration. Water Resources Research 43(1): 16pp.
func DDS(nDim, nSmpl int, rng *rand.Rand, fun func(u []float64) float64, minimize bool) ([]float64, float64) {
	// step2: random selection of initial samples
	ubest := make([]float64, nDim)
	for d := 0; d < nDim; d++ {
		ubest[d] = rng.Float64()
	}
	fbest := fun(ubest)

	ldnm := math.Log(float64(nSmpl))
	for i := 0; i < nSmpl; i++ {
		// step 3: random select
		pi := 1 - math.Log(float64(i))/ldnm
		var nn []int
		for d := 0; d < nDim; d++ {
			if rng.Float64() > pi {
				nn = append(nn, d)
			}
		}
		if len(nn) == 0 {
			nn = []int{rng.Intn(nDim)}
		}
		// step 4: perturb
		unew := make([]float64, nDim)
		copy(unew, ubest)
		for _, j := range nn {
			unew[j] = ubest[j] + rng.NormFloat64()*pertubationValue
			if unew[j] < 0. { // reflect
				unew[j] *= -1.
				if unew[j] > 1. {
					unew[j] = 0.
				}
			}
			if unew[j] > 1. { // reflect
				unew[j] = 2. - unew[j]
				if unew[j] < 0. {
					unew[j] = 1.
				}
			}
		}
		// step 5: evaluate
		fnew := fun(unew)
		if minimize {
			if fnew <= fbest {
				fbest = fnew
				copy(ubest, unew)
			}
		} else {
			if fnew >= fbest {
				fbest = fnew
				copy(ubest, unew)
			}
		}
	}

	return ubest, fbest
}
