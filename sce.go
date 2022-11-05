package glbopt

import (
	"fmt"
	"log"
	"math"
	"math/rand"
	"sort"
	"sync"

	"github.com/maseology/mmaths"
	"github.com/maseology/mmaths/slice"
	"github.com/maseology/montecarlo/smpln"
)

const (
	maxgen    = 500
	cnvrgcrit = 0.01 // hypercube that contains all samples as a proportion to the initial hypercube
	dstngcnt  = 10   // number of repeated generations where no change in objfunc is found
	alpha     = 1    // number of evolutionary steps; Duan etal (1993) sets this equal to 1. [alpha >= 1]
)

// cmplx complex struct
type cmplx struct {
	u [][]float64
	f []float64
	k int
}

// SCE (-UA) (Shuffled Complex Evolution, University of Arizona)
// Duan, Q.Y., V.K. Gupta, and S. Sorooshian, 1993. Shuffled Complex Evolution Approach for Effective and Efficient Global Minimization. Journal of Optimization Theory and Applications 76(3) pp.501-521.
func SCE(nComplx, nDim int, rng *rand.Rand, fun func(u []float64) float64, minimize bool) ([]float64, float64) {
	// p: number of complexes (p>=1);  n: number of dimensions;  m: number of points per complex (m>=n+1)
	if nDim <= 0 {
		log.Panicf("SCE error: nDim = %v\nSCE is best used for problem having greater than 1 dimension", nDim)
	}

	// step 0 initialize
	p := nComplx          // number of complexes
	n := nDim             // number of dimensions
	m := 2*nDim + 1       // number of points per complex. Can be specified, but default (Duan etal 1993) used here. Just ensure m >= n+1
	s := p * m            // sample size
	a := make([][]int, p) // complex cross-reference
	for i := 0; i < p; i++ {
		a[i] = make([]int, m)
	}

	// step 1 generate sample. Note: u() and f() never to change order, only certain samples are replaced through evolution.
	fmt.Printf(" SCE: generating %d initial samples to fulfill %d %d-dimensional complexes..\n", s, p, n)
	// u, f := GenerateSamples(fun, n, s)
	// d := mmaths.Sequential(s - 1)
	u, f, d := generateSamples(fun, n, s, rng)

	//  CCE step 0 initialize
	beta := m  // number of offspring; Duan etal (1993) sets this equal to m [beta >= 1]
	q := n + 1 // number of points that define a subcomplex; setting q=n+1 is the standard simplex size specified by Nelder and Mead (1965), as set by Duan etal (1993) [2<=q<=m]

	//  CCE step 1 assign triangular weights; sum(w)=1
	w := make([]float64, m)
	for i := 0; i < m; i++ {
		w[i] = float64(2*(m-i)) / float64(m*(m+1)) // altered from Duan etal. (1993), to handle a zero-based array
	}

	// step 2 rank points (f() is not to change order)
	rank := func() {
		f2 := make([]float64, len(f))
		copy(f2, f)
		d = slice.Sequential(len(d) - 1) // resetting d
		sort.Sort(mmaths.IndexedSlice{Indx: d, Val: f2})
		if !minimize {
			slice.Rev(d) // ordering from best (highest evaluated score) to worst
		}
	}
	rank()

	ngen, flst, fcnt := 0, -1., 0
	fmt.Println(" SCE: evolving..")
	fmt.Printf("  gen\tcnv\t\tOF\t\t[U]\n")
	fmt.Printf("  %d\tNA\t\t%f\n\t%.3f\n", ngen, f[d[0]], u[d[0]])
nextgen:
	// step 3 partition into complexes
	for k := 0; k < p; k++ {
		for j := 0; j < m; j++ {
			a[k][j] = d[k+p*j] // complexes a() becomes a pointer to u() and f()
		}
	}

	// step 4 evolve - competitive complex evolution (CCE):
	var wg sync.WaitGroup
	cmplxs := make(chan cmplx)
	cnvs := make(chan float64, p)
	wg.Add(p)
	for k := 0; k < p; k++ {
		go func() {
			defer wg.Done()
			c := <-cmplxs
			c.cce(cnvs, w, n, m, q, beta, minimize, fun, rng)

			// reset function values parameter samples
			for i := 0; i < m; i++ {
				f[a[c.k][i]] = c.f[i]
				u[a[c.k][i]] = c.u[i]
			}
		}()
	}

	for k := 0; k < p; k++ {
		uk := make([][]float64, m)
		fk := make([]float64, m)
		for j := 0; j < m; j++ {
			uk[j] = u[a[k][j]] // pointer to u-array
			fk[j] = f[a[k][j]]
		}
		c := cmplx{uk, fk, k}
		cmplxs <- c
	}

	cnv := 0.
	for k := 0; k < p; k++ {
		ck := <-cnvs
		if ck > cnv {
			cnv = ck
		}
	}
	wg.Wait()
	close(cmplxs)
	close(cnvs)

	// step 5 shuffle complexes first by re-ranking d() (Note: f() has never changed order)
	rank()

	// step 6 check for convergence
	ngen++
	fmt.Printf("  %d\t%.6f\t%f\n\t%.3f\n", ngen, cnv, f[d[0]], u[d[0]])
	if ngen >= maxgen { // failure
		log.Printf("maximimum iterations (generations) of %v reached, failed to converge on optimum\n", maxgen)
		goto finish
	} else if cnv < cnvrgcrit { // parameter convergence
		goto finish
	} else if flst == f[d[0]] {
		if fcnt >= dstngcnt {
			goto finish
		}
		fcnt++
		goto nextgen
	} else {
		flst = f[d[0]]
		fcnt = 0
		goto nextgen
	}

finish:
	return u[d[0]], f[d[0]]
}

func generateSamples(fun func(p []float64) float64, n, s int, rng *rand.Rand) ([][]float64, []float64, []int) {
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

	sp := smpln.NewLHC(rng, s, n, false) // smpln.NewHalton(s, n)
	for k := 0; k < s; k++ {
		ut := make([]float64, n)
		for j := 0; j < n; j++ {
			ut[j] = sp.U[j][k]
		}
		smpls <- ut
	}
	wg.Wait()
	close(smpls)

	f := make([]float64, s)   // function value
	d := make([]int, s)       // function rank; d[0] is the best evaluated
	u := make([][]float64, s) // sample points
	for k := 0; k < s; k++ {
		u[k] = make([]float64, n)
		r := <-results
		for j := 0; j < n; j++ {
			u[k][j] = r[j]
		}
		f[k] = r[n]
		d[k] = k
	}
	close(results)

	// serial version for testing
	// sp := smpln.NewLHC(rng, s, n, false)
	// f := make([]float64, s)   // function value
	// d := make([]int, s)       // function rank; d[0] is the best evaluated
	// u := make([][]float64, s) // sample points
	// for k := 0; k < s; k++ {
	// 	fmt.Print(k)
	// 	ut := make([]float64, n)
	// 	for j := 0; j < n; j++ {
	// 		ut[j] = sp.U[j][k]
	// 	}
	// 	r := append(ut, fun(ut))
	// 	u[k] = make([]float64, n)
	// 	for j := 0; j < n; j++ {
	// 		u[k][j] = r[j]
	// 	}
	// 	f[k] = r[n]
	// 	d[k] = k
	// 	fmt.Println(".")
	// }

	return u, f, d
}

func (c *cmplx) cce(cnv chan<- float64, w []float64, n, m, q, beta int, minimize bool, fun func(p []float64) float64, rng *rand.Rand) {
	//  CCE step 0 initialize (assigned above)
	//  CCE step 1 assign triangular weights; sum(w)=1 (built above)
	//  CCE steps 2-5, applied to every k complex
	for j := 0; j < beta; j++ {
		//  CCE step 2 select parents & step 3 generate offspring
		c.sceuacce(w, n, m, q, minimize, fun, rng)

		// step 4 replace parents by offspring
		ct, ft, fi := make([][]float64, m), make([]float64, m), slice.Sequential(m-1)
		copy(ft, c.f)
		sort.Sort(mmaths.IndexedSlice{Indx: fi, Val: c.f})
		if !minimize { // ordering from best (highest evaluated score) to worst
			slice.Rev(fi)
			slice.RevF(c.f)
		}
		copy(ct, c.u)
		for i := 0; i < m; i++ {
			c.u[i] = ct[fi[i]]
		}
	}
	cnv <- converge(smallestHypercube(c.u, m, n)) // for checking parameter convergence on a per-complex basis
}

func converge(hn, hx []float64) float64 {

	// sx := 0.
	// for i := 0; i < len(hn); i++ {
	// 	if hx[i]-hn[i] > sx { // looking for maximum range in sample space s
	// 		sx = hx[i] - hn[i]
	// 	}
	// }
	// return sx // max dimension

	// sx := 0.
	// for i := 0; i < len(hn); i++ {
	// 	sx += hx[i] - hn[i]
	// }
	// return sx / float64(len(hn)) // arithmetic mean

	sx := 1.
	for i := 0; i < len(hn); i++ {
		sx *= hx[i] - hn[i]
	}
	return math.Pow(sx, 1./float64(len(hn))) // geometric mean
}

func smallestHypercube(u [][]float64, s, n int) ([]float64, []float64) {
	// compute H, the smallest hypercube containing A^k in Duan etal (1993)
	hn, hx := make([]float64, n), make([]float64, n)
	for j := 0; j < n; j++ {
		hn[j] = 1.
		hx[j] = 0.
	}
	for i := 0; i < s; i++ {
		for j := 0; j < n; j++ {
			if u[i][j] < hn[j] {
				hn[j] = u[i][j]
			}
			if u[i][j] > hx[j] {
				hx[j] = u[i][j]
			}
		}
	}
	return hn, hx
}

// competitive complex evolution (CCE)
// from: Duan, Q.Y., V.K. Gupta, and S. Sorooshian, 1993. Shuffled Complex Evolution Approach for Effective and Efficient Global Minimization. Journal of Optimization Theory and Applications 76(3) pp.501-521.
func (c *cmplx) sceuacce(w []float64, n, m, q int, minimize bool, fun func(p []float64) float64, rng *rand.Rand) {
	// step 2 select q parents using assigned weights; q is defined as a subcomplex
	l := make([]int, q)      // L: the locations of 'a' which are used to construct B=[ui, vi]
	vi := make([]float64, q) // function values associated with ui
	for i := 0; i < q; i++ {
		bl := make([]bool, m)
	redo2:
		r1, r2 := rng.Float64(), 0.
		for j := 0; j < m; j++ {
			r2 += w[j]
			if r2 >= r1 && !bl[j] {
				bl[j] = true
				l[i] = j
				vi[i] = c.f[j]
				goto continue2
			}
		}
		goto redo2
	continue2:
	}

	// step 3 generate offspring
	hn, hx := smallestHypercube(c.u, m, n)
	for x := 0; x < alpha; x++ {
		//  3a) sort L and B by function value
		sort.Sort(mmaths.IndexedSlice{Indx: l, Val: vi})
		if !minimize { // ordering from best (highest evaluated score) to worst
			slice.Rev(l)
			slice.RevF(vi)
		}

		// determine subcomplex centroid g
		g, r := make([]float64, n), make([]float64, n) // g() centroid, r() reflection step
		for i := 0; i < n; i++ {
			sum := 0.0
			for j := 0; j < q-1; j++ {
				sum += c.u[l[j]][i]
			}
			g[i] = sum / float64(q-1) // ie, average value of all but the worst evaluated function
		}

		// 3b) compute new point (reflection)
		for i := 0; i < n; i++ {
			r[i] = 2.*g[i] - c.u[l[q-1]][i]
		}

		// 3c)
		for i := 0; i < n; i++ { // check if r() is in feasible space
			if r[i] > 1. || r[i] < 0. {
				goto _3cMutate
			}
		}
		goto _3d
	_3cMutate: // reflection went outside feasible space, mutate from smallest hypercube
		for i := 0; i < n; i++ { // mutation
			r[i] = hn[i] + rng.Float64()*(hx[i]-hn[i]) // r() is used here in place of z() in Duan etal (1993)
		}

	_3d: // 3d)
		fr := fun(r)
		if (minimize && fr < vi[q-1]) || (!minimize && fr > vi[q-1]) { // improvement
			vi[q-1] = fr
			copy(c.u[l[q-1]], r)
			goto _3f
		} else {
			for i := 0; i < n; i++ { // contraction step
				r[i] = (g[i] + c.u[l[q-1]][i]) / 2. // r() is used here in place of c() in Duan etal (1993)
			}
		}

		// 3e)
		fr = fun(r)                                                    // fc
		if (minimize && fr < vi[q-1]) || (!minimize && fr > vi[q-1]) { // improvement
			vi[q-1] = fr
			copy(c.u[l[q-1]], r)
		} else {
			for i := 0; i < n; i++ { // mutation step
				r[i] = hn[i] + rng.Float64()*(hx[i]-hn[i]) // r() is used here in place of z() in Duan etal (1993)
			}
			vi[q-1] = fun(r) // fz
			copy(c.u[l[q-1]], r)
		}

	_3f: // 3f) Repeat Steps (a) through (e) alpha times
	}

	// reset f
	for i := 0; i < q; i++ {
		c.f[l[i]] = vi[i]
	}
}
