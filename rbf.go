package glbopt

import (
	"math"
	"math/rand"

	"github.com/maseology/montecarlo/smpln"
	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/plotutil"
	"gonum.org/v1/plot/vg"
)

// const (
// 	multiquadric = 1 << iota
// 	inverseMultiquadric
// 	inverseQuadric
// 	gaussian
// 	linear
// 	cubic
// 	quintic
// 	thinPlate
// 	surfaceSpline // aka polyharmonic spline
// )

// locals
var d int
var z [][]float64             // sample points
var y, lambda, poly []float64 // (expensive) function evaluation; lambda; polynominal tail

// default settings
// var rbfsurf, rbfsKappa, rbfsEpsilon = cubic, 0, 0.
var w, wi = [4]float64{0.3, 0.5, 0.8, 0.95}, 0

// SurrogateRBF : A radial basis function surrogate model algorithm for computationally expensive nonlinear mixed-integer black-box global optimization problems
func SurrogateRBF(nIter, nDim int, rng *rand.Rand, fun func(u []float64) float64) ([]float64, float64) {
	// ref: Müller Shoemaker Piché 2013 SO-MI: A surrogate model algorithm for computationally expensive nonlinear mixed-integer black-box global optimization problems for implementation
	// only minimization function supported
	d = nDim
	nc := initialize(nIter, fun, rng)
	for k := 0; k < nIter; k++ {
		addEvaluations(nc, fun, rng)
		// plotRBF(fmt.Sprintf("rbf%v.png", k+1), fun)
	}

	// locate optimal sample
	ksv, ysv := -1, math.MaxFloat64
	for k := 0; k < len(y); k++ {
		if y[k] < ysv {
			ksv = k
			ysv = y[k]
		}
	}
	if ksv == -1 {
		panic("Surrogate radial basis function could not find a solution")
	}

	return z[ksv], y[ksv]
}

func initialize(nIter int, fun func(u []float64) float64, rng *rand.Rand) int {
	s := 2 * (d + 1) // hard-coded multiple of initial runs (see Müller Shoemaker 2014 Influence of ensemble surrogate models and sampling strategy on the solution quality of algorithms for computationally expensive black-box global optimization problems)
	nc := 500 * d    // number of candidates
	// if nc > 5000 {
	// 	nc = 5000
	// }
	z = make([][]float64, s, s+2*nIter)
	y = make([]float64, s, s+2*nIter)
	sp := smpln.NewLHC(s, d)
	sp.Make(rng, false)
	for k := 0; k < s; k++ {
		z1 := make([]float64, d)
		for j := 0; j < d; j++ {
			z1[j] = sp.U[j][k]
		}

		z[k] = z1
		y[k] = fun(z1)
	}
	solveLambdaPoly()
	// plotInit("rbf0.png", fun)

	// if multiquadric&rbfsurf == multiquadric || inverseMultiquadric&rbfsurf == inverseMultiquadric || surfaceSpline&rbfsurf == surfaceSpline {
	// 	if rbfsKappa <= 0 {
	// 		log.Panic("RBF surrogate error, kappa must be greater than zero for the chosen RBF surface")
	// 	}
	// }
	return nc
}

func addEvaluations(nc int, fun func(u []float64) float64, rng *rand.Rand) {
	addGlobalEvaluation(nc, fun, rng) // select potential minimum candidate
	addLocalEvaluation(nc, fun, rng)  // select potential local minimum candidate
}

func addGlobalEvaluation(nc int, fun func(u []float64) float64, rng *rand.Rand) {
	c := make([][]float64, nc) // candate points
	sp := smpln.NewLHC(nc, d)
	sp.Make(rng, false)
	for i := 0; i < nc; i++ {
		c[i] = make([]float64, d)
		for j := 0; j < d; j++ {
			c[i][j] = sp.U[j][i]
		}
	}
	evaluateFunction(c, nc, fun)
}

func addLocalEvaluation(nc int, fun func(u []float64) float64, rng *rand.Rand) {
	ymin, zmin := math.MaxFloat64, -1
	for i := 0; i < len(z); i++ {
		if y[i] < ymin {
			ymin = y[i]
			zmin = i
		}
	}
	c := make([][]float64, nc) // candate points
	for i := 0; i < nc; i++ {
		c[i] = make([]float64, d)
		perm := math.Pow10(-(rng.Intn(3) + 1))
	redo:
		loop := true
		for j := 0; j < d; j++ {
			if d <= 5 || 5./float64(d) > rng.Float64() {
				if rng.Float64() < 0.5 {
					c[i][j] = math.Max(z[zmin][j]-perm*rng.Float64(), 0.0)
				} else {
					c[i][j] = math.Min(z[zmin][j]+perm*rng.Float64(), 1.0)
				}
				loop = false
			} else {
				c[i][j] = z[zmin][j]
			}
			if loop {
				goto redo
			}
		}
	}
	evaluateFunction(c, nc, fun)
}

func evaluateFunction(c [][]float64, nc int, fun func(u []float64) float64) {
	s := make([]float64, nc) // response surface
	r := make([]float64, nc) // Euclidean norm
	sn, sx, rn, rx := math.MaxFloat64, -math.MaxFloat64, math.MaxFloat64, -math.MaxFloat64
	for i := 0; i < nc; i++ {
		s[i], r[i] = evaluateSurrogate(c[i]) // cheap evaluation
		if s[i] < sn {
			sn = s[i]
		}
		if s[i] > sx {
			sx = s[i]
		}
		if r[i] < rn {
			rn = r[i]
		}
		if r[i] > rx {
			rx = r[i]
		}
	}
	qsv, isv := math.MaxFloat64, -1
	for i := 0; i < nc; i++ {
		vs, vd := 1., 1.
		if sx > sn {
			vs = (s[i] - sn) / (sx - sn)
		}
		if rx > rn {
			vd = (rx - r[i]) / (rx - rn)
		}
		qi := w[wi]*vs + (1.-w[wi])*vd
		if qi < qsv {
			isv = i
			qsv = qi
		}
	}
	if isv == -1 {
		for i := 0; i < nc; i++ {
			println(r[i])
		}
		for i := 0; i < len(z); i++ {
			println(lambda[i])
		}
		panic("Surrogate radial basis function could not find a potential minimum candidate")
	}

	z = append(z, c[isv])
	y = append(y, fun(c[isv])) // expensive evaluation
	solveLambdaPoly()
	cycleWeights()
}

func solveLambdaPoly() {
	// see Müller Shoemaker Piché 2013 SO-MI- A surrogate model algorithm for computationally expensive nonlinear mixed-integer black-box global optimization problems for implimentation)
	n := len(z)
	phi := make([][]float64, n)
	p := make([][]float64, n)
	f := make([]float64, n)
	for i := 0; i < n; i++ {
		phi[i] = make([]float64, n)
		p[i] = make([]float64, d+1)
		f[i] = y[i]
		for g := 0; g < d; g++ {
			p[i][g] = z[i][g]
		}
		p[i][d] = 1. // m_Phi (I believe; see Regis Shoemaker, 2005) = -1 if Gaussian; 0 if linear or multiquadric; 1 if cubic or the thin plate spline
		if i == n-1 {
			goto exit1
		}
		for j := i + 1; j < n; j++ {
			r := 0.
			for g := 0; g < d; g++ {
				r += math.Pow(z[i][g]-z[j][g], 2.)
			}
			phi[i][j] = math.Pow(math.Sqrt(r), 3.) // radialBasisFunction(math.Sqrt(r))
		}
	}
exit1:
	for j := 0; j < n-1; j++ { // build lower triangle of Phi matrix
		for i := j + 1; i < n; i++ {
			phi[i][j] = phi[j][i]
		}
	}

	a := make([]float64, (n+d+1)*(n+d+1))
	b := make([]float64, n+d+1)
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			a[i*(n+d+1)+j] = phi[i][j]
		}
		b[i] = f[i]
		for j := 0; j <= d; j++ {
			a[i*(n+d+1)+n+j] = p[i][j]
		}
	}
	for i := 0; i <= d; i++ {
		for j := 0; j < n; j++ {
			a[(i+n)*(n+d+1)+j] = p[j][i]
		}
	}

	am := mat.NewDense(n+d+1, n+d+1, a)
	bm := mat.NewVecDense(n+d+1, b)
	x := svdSolve(am,bm)

	// am := mat.NewDense(n+d+1, n+d+1, a)
	// bm := mat.NewVecDense(n+d+1, b)
	// var x mat.VecDense
	// if err := x.SolveVec(am, bm); err != nil {
	// 	fmt.Println(err)
	// }

	// x := mat.NewDense(n+d+1, 1, make([]float64, n+d+1))
	// if err := x.Solve(am, bm); err != nil {
	// 	fmt.Println(err)
	// } // A*x=b
	// // if err := am.Inverse(am); err != nil {
	// // 	fmt.Println(err)
	// // }
	// // x.Mul(am, bm)

	lambda = make([]float64, n)
	poly = make([]float64, d+1)
	for i := 0; i < n; i++ {
		lambda[i] = x.At(i, 0)
	}
	for i := 0; i <= d; i++ {
		poly[i] = x.At(n+i, 0)
	}

	// var a maths.Matrix = make([][]float64, n+d+1)
	// var b maths.Matrix = make([][]float64, n+d+1)
	// for i := 0; i < n; i++ {
	// 	a[i] = make([]float64, n+d+1)
	// 	b[i] = make([]float64, 1)
	// 	for j := 0; j < n; j++ {
	// 		a[i][j] = phi[i][j]
	// 	}
	// 	b[i][0] = f[i]
	// 	for j := 0; j <= d; j++ {
	// 		a[i][n+j] = p[i][j]
	// 	}
	// }
	// for i := 0; i <= d; i++ {
	// 	a[n+i] = make([]float64, n+d+1)
	// 	b[n+i] = make([]float64, 1)
	// 	for j := 0; j < n; j++ {
	// 		a[n+i][j] = p[j][i]
	// 	}
	// }

	// x := a.GaussJordanElimination().Multiply(b) // x=a^-1*b
	// lambda = make([]float64, n)
	// poly = make([]float64, d+1)
	// for i := 0; i < n; i++ {
	// 	lambda[i] = x[i][0]
	// }
	// for i := 0; i <= d; i++ {
	// 	poly[i] = x[n+i][0]
	// }

	// // println(n + d + 1)
	// // println()
	// // for i := 0; i < n; i++ {
	// // 	println(z[i][0], y[i])
	// // }
	// // println()
	// // for i := 0; i < n+d+1; i++ {
	// // 	for j := 0; j < n+d+1; j++ {
	// // 		fmt.Printf("%2.4f", a[i][j])
	// // 		fmt.Print(" ")
	// // 	}
	// // 	println()
	// // }
	// // println()
	// // for i := 0; i < n+d+1; i++ {
	// // 	println(x[i][0])
	// // }
	// // println()
	// // panic("asdf")

}

func svdSolve(a *mat.Dense, b *mat.VecDense) *mat.VecDense {
	// following https://www.youtube.com/watch?v=oTCLm-WnX9Y
	// svdSolve(mat.NewDense(3, 2, []float64{1., 0., 0., 2., 0., 1.}), mat.NewVecDense(3, []float64{0., 1., 0.}))
	// Solve x in Ax=b
	ar, ac := a.Dims()

	var svd mat.SVD
	if !svd.Factorize(a, mat.SVDFull) {
		panic("SVD solver error")
	}
	u := svd.UTo(nil)
	v := svd.VTo(nil)
	sv := svd.Values(nil) // sigma vectors
	for i := 0; i < len(sv); i++ {
		if sv[i] != 0. {
			sv[i] = 1. / sv[i]
		}
	}
	s := mat.NewDiagonalRect(ar, ac, sv)
	si := mat.DenseCopyOf(s.T()) // pseudo-inverse

	z := mat.NewVecDense(ar, nil)
	z.MulVec(u.T(), b)

	y := mat.NewVecDense(ac, nil)
	y.MulVec(si, z)

	x := mat.NewVecDense(ac, nil)
	x.MulVec(v, y)

	return x
}

func evaluateSurrogate(c []float64) (float64, float64) {
	// s: RBF interpolant/response surface
	s, rmin := 0.0, math.MaxFloat64
	for i := 0; i < len(z); i++ {
		r := 0.
		for j := 0; j < d; j++ {
			r += math.Pow(c[j]-z[i][j], 2.)
		}
		r = math.Sqrt(r)
		if r < rmin {
			rmin = r
		}
		// if r == 0 { ///////////////////////////////////////////////////////////////
		// 	for j := 0; j < d; j++ {
		// 		println(c[j])
		// 	}
		// 	println()
		// 	for j := 0; j < d; j++ {
		// 		println(z[i][j])
		// 	}
		// 	println()
		// }
		s += lambda[i] * math.Pow(r, 3.) // radialBasisFunction(r)
	}
	p := poly[d] // polynomial tail
	for i := 0; i < d; i++ {
		p += poly[i] * c[i]
	}
	return s + p, rmin
}

// func radialBasisFunction(r float64) float64 {
// 	if r <= 0. {
// 		log.Panicf("RBF surrogate interolation error, r = %v\n", r)
// 	}
// 	switch rbfsurf {
// 	case gaussian:
// 		return math.Exp(-math.Pow(rbfsEpsilon*r, 2.))
// 	case inverseQuadric:
// 		return 1.0 / (math.Pow(r/rbfsEpsilon, 2.) + 1.0)
// 	case multiquadric:
// 		// Return Sqrt((r / _rbfs_epsilon) ^ 2.0 + 1.0) ' as given from various online references
// 		return math.Pow(math.Pow(r, 2.)+math.Pow(rbfsEpsilon, 2.), float64(rbfsKappa)) // as given by Shoemaker
// 	case inverseMultiquadric:
// 		// Return 1.0 / Sqrt((r / _rbfs_epsilon) ^ 2.0 + 1.0) // as given from various online references
// 		return math.Pow(math.Pow(r, 2.)+math.Pow(rbfsEpsilon, 2.), -float64(rbfsKappa)) // as given by Shoemaker
// 	case linear:
// 		return r
// 	case cubic:
// 		return math.Pow(r, 3.)
// 	case quintic:
// 		return math.Pow(r, 5.)
// 	case thinPlate:
// 		return math.Pow(r, 2.) * math.Log(r)
// 	case surfaceSpline:
// 		if rbfsKappa%2 != 0 { // kappa is odd
// 			return math.Pow(r, float64(rbfsKappa))
// 		}
// 		return math.Pow(r, float64(rbfsKappa)) * math.Log(r)
// 	}
// 	return r
// }

func cycleWeights() {
	wi++
	if wi >= len(w) {
		wi = 0
	}
}

func plotInit(fp string, fun func(u []float64) float64) {
	p, err := plot.New()
	if err != nil {
		panic(err)
	}

	p.Title.Text = fp
	p.X.Label.Text = "z"
	p.Y.Label.Text = "y"

	z0 := make([]float64, len(z))
	for i := 0; i < len(z); i++ {
		z0[i] = z[i][0] //maths.LinearTransform(-2., 2., z[i][0])
	}

	err = plotutil.AddLines(p, objfunc(fun))
	if err != nil {
		panic(err)
	}
	err = plotutil.AddScatters(p, objfuncPts(z0, y))
	if err != nil {
		panic(err)
	}

	// Save the plot to a PNG file.
	if err := p.Save(4*vg.Inch, 4*vg.Inch, fp); err != nil {
		panic(err)
	}
}

func plotRBF(fp string, fun func(u []float64) float64) {
	p, err := plot.New()
	if err != nil {
		panic(err)
	}

	p.Title.Text = fp
	p.X.Label.Text = "z"
	p.Y.Label.Text = "y"

	z0 := make([]float64, len(z))
	for i := 0; i < len(z); i++ {
		z0[i] = z[i][0] //maths.LinearTransform(-2., 2., z[i][0])
	}
	// c0 := make([]float64, len(z))
	// for i := 0; i < len(c); i++ {
	// 	c0[i] = maths.LinearTransform(-2., 2., c[i][0])
	// }

	err = plotutil.AddLines(p, objfunc(fun), surrogatefunc(fun))
	if err != nil {
		panic(err)
	}
	err = plotutil.AddScatters(p, objfuncPts(z0, y))
	if err != nil {
		panic(err)
	}

	// Save the plot to a PNG file.
	if err := p.Save(4*vg.Inch, 4*vg.Inch, fp); err != nil {
		panic(err)
	}
}

func objfunc(fun func(u []float64) float64) plotter.XYs {
	pts := make(plotter.XYs, 100)
	for i := 0; i < 100; i++ {
		u := []float64{float64(i) / 100.}
		pts[i].X = u[0] // maths.LinearTransform(-2., 2., u[0])
		pts[i].Y = fun(u)
	}
	return pts
}

func surrogatefunc(fun func(u []float64) float64) plotter.XYs {
	pts := make(plotter.XYs, 100)
	for i := 0; i < 100; i++ {
		u := []float64{float64(i) / 100.}
		pts[i].X = u[0] //maths.LinearTransform(-2., 2., u[0])
		pts[i].Y, _ = evaluateSurrogate(u)
	}
	return pts
}

func objfuncPts(x, y []float64) plotter.XYs {
	if len(x) != len(y) {
		panic("mmplt.scatter error: unequal array sizes")
	}
	pts := make(plotter.XYs, len(x))
	for i := range pts {
		pts[i].X = x[i]
		pts[i].Y = y[i]
	}
	return pts
}
