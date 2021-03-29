package glbopt

import (
	"fmt"
	"log"
	"math"
	"math/rand"
	"runtime"

	"github.com/maseology/montecarlo"
	"github.com/maseology/montecarlo/smpln"
	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/plotutil"
	"gonum.org/v1/plot/vg"
)

const (
	multiquadric = 1 << iota
	inverseMultiquadric
	inverseQuadric
	gaussian
	linear
	cubic
	quintic
	thinPlate
	surfaceSpline // aka polyharmonic spline
)
const rbfsurf, rbfsKappa, rbfsEpsilon = cubic, 0, 0.

type rbf struct {
	z               [][]float64 // sample points
	y, lambda, poly []float64   // (expensive) function evaluation; lambda; polynominal tail
	d, nc, wi       int
}

var w = [4]float64{0.3, 0.5, 0.8, 0.95}

// SurrogateRBF : A radial basis function surrogate model algorithm for computationally expensive nonlinear mixed-integer black-box global optimization problems
func SurrogateRBF(nIter, nDim int, rng *rand.Rand, fun func(u []float64) float64) ([]float64, float64) {
	// ref: Müller Shoemaker Piché 2013 SO-MI: A surrogate model algorithm for computationally expensive nonlinear mixed-integer black-box global optimization problems for implementation
	// only minimization function supported

	var r rbf
	r.d = nDim
	r.initialize(nIter, fun, rng)
	for k := 0; k < nIter; k++ {
		r.addEvaluations(fun, rng)
		// plotRBF(fmt.Sprintf("rbf%v.png", k+1), fun)
	}

	// locate optimal sample
	ksv, ysv := -1, math.MaxFloat64
	for k := 0; k < len(r.y); k++ {
		if r.y[k] < ysv {
			ksv = k
			ysv = r.y[k]
		}
	}
	if ksv == -1 {
		panic("Surrogate radial basis function could not find a solution")
	}

	return r.z[ksv], r.y[ksv]
}

func (r *rbf) initialize(nIter int, fun func(u []float64) float64, rng *rand.Rand) {
	s := 2 * (r.d + 1) //* runtime.GOMAXPROCS(0) // hard-coded multiple of initial runs (see Müller Shoemaker 2014 Influence of ensemble surrogate models and sampling strategy on the solution quality of algorithms for computationally expensive black-box global optimization problems)
	if s < runtime.GOMAXPROCS(0) {
		s = runtime.GOMAXPROCS(0)
	}
	r.nc = 500 * r.d // hard-coded number of candidates
	if r.nc > 5000 {
		r.nc = 5000
	}

	r.z = make([][]float64, s, s+2*nIter)
	r.y = make([]float64, s, s+2*nIter)

	fmt.Printf(" RBF: generating initial surface from %d samples..", s)
	u, f := montecarlo.GenerateSamples(fun, r.d, s)
	fmt.Println("complete")
	for k := 0; k < s; k++ {
		z1 := make([]float64, r.d)
		for j := 0; j < r.d; j++ {
			z1[j] = u[k][j]
		}
		r.z[k] = z1
		r.y[k] = f[k]
	}

	// uin := make(chan []float64)
	// fout := make(chan float64)
	// sp := smpln.NewLHC(s, r.d)
	// sp.Make(rng, false)
	// for k := 0; k < s; k++ {
	// 	go func() {
	// 		fout <- fun(<-uin)
	// 	}()
	// 	z1 := make([]float64, r.d)
	// 	for j := 0; j < r.d; j++ {
	// 		z1[j] = sp.U[j][k]
	// 	}
	// 	r.z[k] = z1
	// 	uin <- z1
	// }

	// // collect results
	// for k := 0; k < s; k++ {
	// 	r.y[k] = <-fout
	// }
	// close(uin)
	// close(fout)

	r.solveLambdaPoly()
	// r.plotInit("rbf0.png", fun)

	if multiquadric&rbfsurf == multiquadric || inverseMultiquadric&rbfsurf == inverseMultiquadric || surfaceSpline&rbfsurf == surfaceSpline {
		if rbfsKappa <= 0 {
			log.Panic("RBF surrogate error, kappa must be greater than zero for the chosen RBF surface")
		}
	}
}

func (r *rbf) addEvaluations(fun func(u []float64) float64, rng *rand.Rand) {
	r.addGlobalEvaluation(fun, rng) // select potential minimum candidate
	r.addLocalEvaluation(fun, rng)  // select potential local minimum candidate
}

func (r *rbf) addGlobalEvaluation(fun func(u []float64) float64, rng *rand.Rand) {
	c := make([][]float64, r.nc) // candate points
	sp := smpln.NewLHC(rng, r.nc, r.d, false)
	for i := 0; i < r.nc; i++ {
		c[i] = make([]float64, r.d)
		for j := 0; j < r.d; j++ {
			c[i][j] = sp.U[j][i]
		}
	}
	r.evaluateFunction(c, fun)
}

func (r *rbf) addLocalEvaluation(fun func(u []float64) float64, rng *rand.Rand) {
	ymin, zmin := math.MaxFloat64, -1
	for i := 0; i < len(r.z); i++ {
		if r.y[i] < ymin {
			ymin = r.y[i]
			zmin = i
		}
	}
	c := make([][]float64, r.nc) // candate points
	for i := 0; i < r.nc; i++ {
		c[i] = make([]float64, r.d)
	redo:
		perm := math.Pow10(-(rng.Intn(3) + 1))
		loop := true
		for j := 0; j < r.d; j++ {
			if r.d <= 5 || 5./float64(r.d) > rng.Float64() {
				if rng.Float64() < 0.5 {
					c[i][j] = math.Max(r.z[zmin][j]-perm*rng.Float64(), 0.)
					if c[i][j] <= 0. {
						c[i][j] = r.z[zmin][j] / 2.
					}
				} else {
					c[i][j] = math.Min(r.z[zmin][j]+perm*rng.Float64(), 1.)
					if c[i][j] >= 1. {
						c[i][j] = (r.z[zmin][j] + 1.) / 2.
					}
				}
				loop = false
			} else {
				c[i][j] = r.z[zmin][j]
			}
			if loop {
				goto redo
			}
		}
	}
	r.evaluateFunction(c, fun)
}

func (r *rbf) evaluateFunction(c [][]float64, fun func(u []float64) float64) {
	s := make([]float64, r.nc)     // response surface
	rnorm := make([]float64, r.nc) // Euclidean norm
	sn, sx, rn, rx := math.MaxFloat64, -math.MaxFloat64, math.MaxFloat64, -math.MaxFloat64
	for i := 0; i < r.nc; i++ {
		s[i], rnorm[i] = r.evaluateSurrogate(c[i]) // cheap evaluation go routine did not help here; kept serial
		if s[i] < sn {
			sn = s[i]
		}
		if s[i] > sx {
			sx = s[i]
		}
		if rnorm[i] < rn {
			rn = rnorm[i]
		}
		if rnorm[i] > rx {
			rx = rnorm[i]
		}
	}
	qsv, isv := math.MaxFloat64, -1
	for i := 0; i < r.nc; i++ {
		vs, vd := 1., 1.
		if sx > sn {
			vs = (s[i] - sn) / (sx - sn)
		}
		if rx > rn {
			vd = (rx - rnorm[i]) / (rx - rn)
		}
		qi := w[r.wi]*vs + (1.-w[r.wi])*vd
		if qi < qsv {
			isv = i
			qsv = qi
		}
	}
	if isv == -1 {
		for i := 0; i < r.nc; i++ {
			fmt.Println(rnorm[i])
		}
		for i := 0; i < len(r.z); i++ {
			fmt.Println(r.lambda[i])
		}
		log.Fatalln("Surrogate radial basis function could not find a potential minimum candidate")
	}

	r.z = append(r.z, c[isv])
	r.y = append(r.y, fun(c[isv])) // expensive evaluation
	r.solveLambdaPoly()

	r.wi++ // cycle weights
	if r.wi >= len(w) {
		r.wi = 0
	}
}

func (r *rbf) evaluateSurrogate(c []float64) (float64, float64) {
	// s: RBF interpolant/response surface
	s, xmin := 0., math.MaxFloat64
	for i := 0; i < len(r.z); i++ {
		x := 0.
		for j := 0; j < r.d; j++ {
			x += math.Pow(c[j]-r.z[i][j], 2.)
		}
		x = math.Sqrt(x)
		if x < xmin {
			xmin = x
		}
		s += r.lambda[i] * radialBasisFunction(x) // r.lambda[i] * math.Pow(x, 3.) //
	}
	p := r.poly[r.d] // polynomial tail
	for i := 0; i < r.d; i++ {
		p += r.poly[i] * c[i]
	}
	return s + p, xmin
}

func (r *rbf) solveLambdaPoly() {
	// see Müller Shoemaker Piché 2013 SO-MI- A surrogate model algorithm for computationally expensive nonlinear mixed-integer black-box global optimization problems for implimentation)
	n := len(r.z)
	phi := make([][]float64, n)
	p := make([][]float64, n)
	f := make([]float64, n)
	for i := 0; i < n; i++ {
		phi[i] = make([]float64, n)
		p[i] = make([]float64, r.d+1)
		f[i] = r.y[i]
		for g := 0; g < r.d; g++ {
			p[i][g] = r.z[i][g]
		}
		p[i][r.d] = 1. // m_Phi (I believe; see Regis Shoemaker, 2005) = -1 if Gaussian; 0 if linear or multiquadric; 1 if cubic or the thin plate spline
		if i == n-1 {
			goto exit1
		}
		for j := i + 1; j < n; j++ {
			x := 0.
			for g := 0; g < r.d; g++ {
				x += math.Pow(r.z[i][g]-r.z[j][g], 2.)
			}
			phi[i][j] = radialBasisFunction(math.Sqrt(x)) // math.Pow(math.Sqrt(x), 3.) //
		}
	}
exit1:
	for j := 0; j < n-1; j++ { // build lower triangle of Phi matrix
		for i := j + 1; i < n; i++ {
			phi[i][j] = phi[j][i]
		}
	}

	a := make([]float64, (n+r.d+1)*(n+r.d+1))
	b := make([]float64, n+r.d+1)
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			a[i*(n+r.d+1)+j] = phi[i][j]
		}
		b[i] = f[i]
		for j := 0; j <= r.d; j++ {
			a[i*(n+r.d+1)+n+j] = p[i][j]
		}
	}
	for i := 0; i <= r.d; i++ {
		for j := 0; j < n; j++ {
			a[(i+n)*(n+r.d+1)+j] = p[j][i]
		}
	}

	am := mat.NewDense(n+r.d+1, n+r.d+1, a)
	bm := mat.NewVecDense(n+r.d+1, b)
	x := svdSolve(am, bm)

	r.lambda = make([]float64, n)
	r.poly = make([]float64, r.d+1)
	for i := 0; i < n; i++ {
		r.lambda[i] = x.At(i, 0)
	}
	for i := 0; i <= r.d; i++ {
		r.poly[i] = x.At(n+i, 0)
	}
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
	var u, v *mat.Dense
	svd.UTo(u)
	svd.VTo(v)
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

func radialBasisFunction(r float64) float64 {
	if r <= 0. {
		log.Panicf("RBF surrogate interpolation error, r = %v\n", r)
	}
	switch rbfsurf {
	case gaussian:
		return math.Exp(-math.Pow(rbfsEpsilon*r, 2.))
	case inverseQuadric:
		return 1.0 / (math.Pow(r/rbfsEpsilon, 2.) + 1.0)
	case multiquadric:
		// Return Sqrt((r / _rbfs_epsilon) ^ 2.0 + 1.0) ' as given from various online references
		return math.Pow(math.Pow(r, 2.)+math.Pow(rbfsEpsilon, 2.), float64(rbfsKappa)) // as given by Shoemaker
	case inverseMultiquadric:
		// Return 1.0 / Sqrt((r / _rbfs_epsilon) ^ 2.0 + 1.0) // as given from various online references
		return math.Pow(math.Pow(r, 2.)+math.Pow(rbfsEpsilon, 2.), -float64(rbfsKappa)) // as given by Shoemaker
	case linear:
		return r
	case cubic:
		return math.Pow(r, 3.)
	case quintic:
		return math.Pow(r, 5.)
	case thinPlate:
		return math.Pow(r, 2.) * math.Log(r)
	case surfaceSpline:
		if rbfsKappa%2 != 0 { // kappa is odd
			return math.Pow(r, float64(rbfsKappa))
		}
		return math.Pow(r, float64(rbfsKappa)) * math.Log(r)
	}
	return r
}

func (r *rbf) plotInit(fp string, fun func(u []float64) float64) {
	p := plot.New()

	p.Title.Text = fp
	p.X.Label.Text = "z"
	p.Y.Label.Text = "y"

	z0 := make([]float64, len(r.z))
	for i := 0; i < len(r.z); i++ {
		z0[i] = r.z[i][0] //maths.LinearTransform(-2., 2., z[i][0])
	}

	err := plotutil.AddLines(p, objfunc(fun))
	if err != nil {
		panic(err)
	}
	err = plotutil.AddScatters(p, objfuncPts(z0, r.y))
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

// func (r *rbf) plotRBF(fp string, fun func(u []float64) float64) {
// 	p, err := plot.New()
// 	if err != nil {
// 		panic(err)
// 	}

// 	p.Title.Text = fp
// 	p.X.Label.Text = "z"
// 	p.Y.Label.Text = "y"

// 	z0 := make([]float64, len(r.z))
// 	for i := 0; i < len(r.z); i++ {
// 		z0[i] = r.z[i][0] //maths.LinearTransform(-2., 2., z[i][0])
// 	}
// 	// c0 := make([]float64, len(z))
// 	// for i := 0; i < len(c); i++ {
// 	// 	c0[i] = maths.LinearTransform(-2., 2., c[i][0])
// 	// }

// 	err = plotutil.AddLines(p, objfunc(fun), r.surrogatefunc(fun))
// 	if err != nil {
// 		panic(err)
// 	}
// 	err = plotutil.AddScatters(p, objfuncPts(z0, r.y))
// 	if err != nil {
// 		panic(err)
// 	}

// 	// Save the plot to a PNG file.
// 	if err := p.Save(4*vg.Inch, 4*vg.Inch, fp); err != nil {
// 		panic(err)
// 	}
// }

// func (r *rbf) surrogatefunc(fun func(u []float64) float64) plotter.XYs {
// 	pts := make(plotter.XYs, 100)
// 	for i := 0; i < 100; i++ {
// 		u := []float64{float64(i) / 100.}
// 		pts[i].X = u[0] //maths.LinearTransform(-2., 2., u[0])
// 		pts[i].Y, _ = r.evaluateSurrogate(u)
// 	}
// 	return pts
// }
