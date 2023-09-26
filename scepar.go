package glbopt

import (
	"fmt"
	"math/rand"
	"sort"
	"sync"

	"github.com/maseology/mmaths"
)

func SCEpar(nComplx, nsimult, nDim int, rng *rand.Rand, fun func(u []float64) float64, minimize bool) ([][]float64, []float64) {
	if nsimult > 1 {
		type result struct {
			u []float64
			f float64
		}

		s := nComplx / nsimult
		res := make(chan result, s)
		fmt.Printf(" SCEpar: generating %d SCE subsets to fulfill %d %d-dimensional complexes..\n", s, nComplx, nDim)
		go func() {
			var wg sync.WaitGroup
			wg.Add(s)
			for k := 0; k < s; k++ {
				go func() {
					defer wg.Done()
					u, f := SCE(s, nDim, rng, fun, minimize)
					res <- result{u, f}
				}()
			}
			wg.Wait()
			close(res)
		}()

		us, fs := make([][]float64, 0, s), make([]float64, 0, s)
		for r := range res {
			us = append(us, r.u)
			fs = append(fs, r.f)
		}
		// return us, fs

		// sort by objective function
		var ofs mmaths.IndexedSlice
		ofs.New(fs)
		sort.Sort(ofs)
		us2, fs2 := make([][]float64, s), make([]float64, s)
		for i, ii := range ofs.Indx {
			us2[i] = us[ii]
			fs2[i] = fs[ii]
		}
		if !minimize { // reverse
			for i, j := 0, len(us2)-1; i < j; i, j = i+1, j-1 {
				us2[i], us2[j] = us2[j], us2[i]
				fs2[i], fs2[j] = fs2[j], fs2[i]
			}
		}
		return us2, fs2
	} else {
		pars, of := SCE(nComplx, nDim, rng, fun, minimize)
		return [][]float64{pars}, []float64{of}
	}
}
