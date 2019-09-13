package glbopt

import (
	"fmt"
	"sort"
	"sync"

	"github.com/maseology/mmaths"
	"github.com/maseology/mmio"
	"github.com/maseology/montecarlo/smpln"
)

// GenerateSamples returns the result from s evalutaions of fun() sampling from n-hypercube
func GenerateSamples(fun func(u []float64) float64, n, s int) ([][]float64, []float64) { // ([][]float64, []float64, []int) {
	var wg sync.WaitGroup
	smpls := make(chan []float64, s)
	results := make(chan []float64, s)
	wg.Add(s)
	for k := 0; k < s; k++ {
		go func() {
			defer wg.Done()
			s := <-smpls
			results <- append(s, fun(s))
		}()
	}

	sp := smpln.NewLHC(s, n) // smpln.NewHalton(s, n)
	for k := 0; k < s; k++ {
		ut := make([]float64, n)
		for j := 0; j < n; j++ {
			ut[j] = sp.U[j][k]
		}
		smpls <- ut
	}
	wg.Wait()
	close(smpls)

	f := make([]float64, s) // function value
	// d := make([]int, s)       // function index, used for ranking
	u := make([][]float64, s) // sample points
	for k := 0; k < s; k++ {
		u[k] = make([]float64, n)
		r := <-results
		for j := 0; j < n; j++ {
			u[k][j] = r[j]
		}
		f[k] = r[n]
		// d[k] = k
	}
	return u, f //, d
}

// RankSamples ranks samples accoring to evaluation value
func RankSamples(f []float64, minimize bool) []int {
	f2 := make([]float64, len(f))
	copy(f2, f)
	d := mmaths.Sequential(len(f) - 1) // resetting d
	sort.Sort(mmaths.IndexedSlice{Indx: d, Val: f2})
	if !minimize {
		mmaths.Rev(d) // ordering from best (highest evaluated score) to worst
	}
	return d
}

// RankedUnBiased returns s n-dimensional samples of fun()
func RankedUnBiased(fp string, fun func(u []float64) float64, n, s int) error {
	fmt.Printf(" generating %d LHC samples from %d dimensions..\n", s, n)
	u, f := GenerateSamples(fun, n, s)
	d := RankSamples(f, true)
	t, err := mmio.NewTXTwriter(fp)
	defer t.Close()
	if err != nil {
		return fmt.Errorf(" Definition.SaveAs: %v", err)
	}
	str := fmt.Sprintf("rank(%d),eval", s)
	for j := 0; j < n; j++ {
		str = str + fmt.Sprintf(",p%03d", j)
	}
	t.WriteLine(str)
	for i, dd := range d {
		str := fmt.Sprintf("%d,%f", i, f[dd])
		for j := 0; j < n; j++ {
			str = str + fmt.Sprintf(",%f", u[dd][j])
		}
		t.WriteLine(str)
	}
	return nil
}
