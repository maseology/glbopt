package glbopt

import (
	"log"
	"math/rand"
	"sort"

	"github.com/maseology/montecarlo/smpln"
)

const (
	alpha     = 1 // number of evolutionary steps; Duan etal (1993) sets this equal to 1
	maxiter   = 10000
	cnvrgcrit = 0.00001
)

// indexedSlice : is an implimentation of Go's sort.Sort used to return the new index of a sorted slice
type indexedSlice struct {
	indx []int
	val  []float64
}

func (is indexedSlice) Len() int {
	return len(is.indx)
}
func (is indexedSlice) Less(i, j int) bool {
	return is.val[i] < is.val[j]
}
func (is indexedSlice) Swap(i, j int) {
	is.indx[i], is.indx[j] = is.indx[j], is.indx[i]
	is.val[i], is.val[j] = is.val[j], is.val[i]
}

// quick function used to reverse order of a slice
func rev(s []int) {
	for i, j := 0, len(s)-1; i < j; i, j = i+1, j-1 {
		s[i], s[j] = s[j], s[i]
	}
}

// SCE (-UA) (Shuffled Complex Evolution, University of Arizona)
// Duan, Q.Y., V.K. Gupta, and S. Sorooshian, 1993. Shuffled Complex Evolution Approach for Effective and Efficient Global Minimization. Journal of Optimization Theory and Applications 76(3) pp.501-521.
func SCE(nComplx, nDim int, r *rand.Rand, fun func(u []float64) float64, minimize bool) ([]float64, float64) {
	// p: number of complexes (p>=1);  n: number of dimensions;  m: number of points per complex (m>=n+1)
	if nDim <= 0 {
		log.Panicf("SCE error: nDim = %v\nSCE is best used for problem having greater than 1 dimension", nDim)
	}
	if nComplx <= 1 {
		panic("SCE error: need to have at least one more complex than number of dimensions")
	}

	// step 0 initialize
	p := nComplx
	n := nDim
	m := 2*nDim + 1        // can specify m, but default used. Just ensure m >= n+1
	s := (p + 1) * (m + 1) // sample size
	u := make([][]float64, s)
	us := make([]float64, n)
	f := make([]float64, s)
	d := make([]int, s)
	a := make([][]int, p)
	for i := 0; i < s; i++ {
		u[i] = make([]float64, n)
	}
	for i := 0; i < p; i++ {
		a[i] = make([]int, m)
	}

	// step 1 generate sample. Note: u() and f() never to change order, only certain samples are replaced through evolution.
	sp := smpln.NewHalton(s, n)
	for k := 0; k < s; k++ {
		for j := 0; j < n; j++ {
			v := sp.U[j][k]
			u[k][j] = v
			us[j] = v
		}
		f[k] = fun(us)
		d[k] = k
	}

	// step 2 rank points (f() is not to change order)
	f2 := make([]float64, len(f))
	copy(f2, f)
	sort.Sort(indexedSlice{indx: d, val: f2})
	if !minimize {
		rev(d) // ordering from best (highest evaluated score) to worst
	}

	// step 3 partition into complexes
	for k := 0; k < p; k++ {
		for j := 0; j < m; j++ {
			a[k][j] = d[k+p*j] // complexes a() becomes a pointer to u() and f()
		}
	}

	// step 4 evolve
	usmn := make([]float64, n)
	usmx := make([]float64, n)
	iter := 0
newgen:
	sceuacce(u, a, f, d, n+1, m, r, fun, minimize)

	// step 5 shuffle
	// this step was completed at the end of function SCE_UA_CCE (step 4)

	// step 6 check for convergence
	cnv := 0.
	for i := 0; i < n; i++ { //determine size of sample hypercube
		usmn[i] = 1.
		usmx[i] = 0.
	}
	for i := 0; i < s; i++ {
		for j := 0; j < n; j++ {
			if u[i][j] < usmn[j] {
				usmn[j] = u[i][j]
			}
			if u[i][j] > usmx[j] {
				usmx[j] = u[i][j]
			}
		}
	}
	for i := 0; i < n; i++ { // for checking parameter convergence
		if usmx[i]-usmn[i] > cnv {
			cnv = usmx[i] - usmn[i] // looking for maximum range in sample space s
		}
	}

	if iter > maxiter { // failure
		log.Println("maximimum iterations reached, failed to converge")
		goto finish
	} else if cnv < cnvrgcrit { // parameter convergence
		goto finish
	} else {
		iter++
		goto newgen
	}

finish:
	for i := 0; i < n; i++ {
		us[i] = u[d[0]][i]
	}
	return us, f[d[0]]

}

// competitive complex evolution (CCE)
// from: Duan, Q.Y., V.K. Gupta, and S. Sorooshian, 1993. Shuffled Complex Evolution Approach for Effective and Efficient Global Minimization. Journal of Optimization Theory and Applications 76(3) pp.501-521.
func sceuacce(u [][]float64, a [][]int, f []float64, d []int, nParents, beta int, rng *rand.Rand, fun func(p []float64) float64, minimize bool) {
	// step 0 initialize
	q := nParents - 1 // number of points (i.e., number of subcomplexes); setting q=n+1 is the standard simplex size specified by Nelder and Mead, 1965, as set by Duan etal (1993)
	m := len(a[0])    // number of points per complex: m >= n + 1
	n := len(u[0])    // number of dimensions
	p := len(a)       // number of complexes
	if q < 1 {
		q = 1
	} // minimum 2 parents
	if q > m {
		q = m
	} // m>=n+1
	if beta < 1 {
		beta = 1
	} // number of offspring; Duan etal (1993) sets this equal to m

	// step 1 develop triangular weights; sum(w)=1
	w := make([]float64, m)
	for i := 0; i < m; i++ {
		w[i] = float64(2*(m-i)) / float64(m) / float64(m+1) // altered from Duan etal., 1993 to incorporate a zero-based array
	}

	for y := 0; y < beta; y++ {
		L := make([]int, q)
		vi := make([]float64, q)
		ui := make([][]float64, q)
		kl := make([]int, q)
		jl := make([]int, q)
		for l := 0; l < q; l++ {
			ui[l] = make([]float64, n)
		}

		// step 2 select q parents using assigned weights; q is defined as a subcomplex
		for i := 0; i < q; i++ {
			db1, db2 := rng.Float64(), 0.
			k, j := 0, -1
		do1:
			j++
			if j >= m {
				k++
				j = 0
			}
			db2 += w[j] / float64(p)
			for h := 0; h < i; h++ {
				if kl[h] == k && jl[h] == j {
					goto do1
				}
			} // added this block to avoid the (rare) case of a potential parent being selected twice
			if db2 <= db1 {
				goto do1
			}
			kl[i] = k
			jl[i] = j
			L[i] = a[k][j] // L() is an index to point u() and evaluated objective function f() from wich q parents were selected
			vi[i] = f[L[i]]
			for j := 0; j < n; j++ {
				ui[i][j] = u[L[i]][j] // ui and vi make up B() in Duan etal (1993)
			}
		}

		umn, umx := make([]float64, n), make([]float64, n) // in place of H (i.e., smallest hypercube) in Duan etal (1993)
		for i := 0; i < n; i++ {
			umn[i] = 1.
			umx[i] = 0.
		}
		for i := 0; i < len(u); i++ {
			for j := 0; j < n; j++ {
				if u[i][j] < umn[j] {
					umn[j] = u[i][j]
				}
				if u[i][j] > umx[j] {
					umx[j] = u[i][j]
				}
			}
		}

		// step 3 generate offspring
		for x := 0; x < alpha; x++ {
			// 3a)
			ui2, L2 := make([][]float64, q), make([]int, q)
			vix := make([]int, q)
			for i := 0; i < q; i++ {
				ui2[i] = make([]float64, n)
				copy(ui2[i], ui[i])
				L2[i] = L[i]
				vix[i] = i
			}
			sort.Sort(indexedSlice{indx: vix, val: vi})
			if !minimize {
				rev(vix) // ordering from best (highest evaluated score) to worst
			}
			for i := 0; i < q; i++ {
				L[i] = L2[vix[i]]
				for j := 0; j < n; j++ {
					ui[i][j] = ui2[vix[i]][j] // note: vi is already sorted
				}
			} // Sort L() and B() accornding to inA() (best to worst)

			g, r := make([]float64, n), make([]float64, n) // g() centroid, r() reflection step
			for i := 0; i < n; i++ {
				sum := 0.0
				for j := 0; j < q-1; j++ {
					sum += ui[j][i]
				}
				g[i] = sum / float64(q-1) // ie, average value of all but the worst evaluated function
			}

			// 3b) reflection
			for i := 0; i < n; i++ {
				r[i] = 2.*g[i] - ui[q-1][i]
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
				r[i] = umn[i] + rng.Float64()*(umx[i]-umn[i]) // r() is used here in place of z() in Duan etal (1993)
			}

		_3d: // 3d)
			eval := fun(r)
			if (minimize && eval < vi[q-1]) || (!minimize && eval > vi[q-1]) { // improvement
				vi[q-1] = eval
				for i := 0; i < n; i++ {
					ui[q-1][i] = r[i]
				}
				goto _3f
			} else {
				for i := 0; i < n; i++ { // contraction step
					r[i] = (g[i] + ui[q-1][i]) / 2. // r() is used here in place of c() in Duan etal (1993)
				}
			}

			// 3e)
			eval = fun(r)
			if (minimize && eval < vi[q-1]) || (!minimize && eval > vi[q-1]) { // improvement
				vi[q-1] = eval
				for i := 0; i < n; i++ {
					ui[q-1][i] = r[i]
				}
				goto _3f
			} else {
				for i := 0; i < n; i++ { // mutation step
					r[i] = umn[i] + rng.Float64()*(umx[i]-umn[i])
					ui[q-1][i] = r[i]
				}
				vi[q-1] = fun(r)
			}

		_3f: // 3f)
		}

		// step 4 replace parents by offspring
		for i := 0; i < q; i++ {
			for j := 0; j < n; j++ {
				u[L[i]][j] = ui[i][j]
			}
			f[L[i]] = vi[i]
		}
		f2 := make([]float64, len(f))
		copy(f2, f)
		sort.Sort(indexedSlice{indx: d, val: f2})
		if !minimize {
			rev(d) // ordering from best (highest evaluated score) to worst
		}

		for k := 0; k < p; k++ {
			for j := 0; j < m; j++ {
				a[k][j] = d[k+p*j]
			}
		}
		// step 5 iterate
	}
}
