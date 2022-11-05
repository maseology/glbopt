# go glbopt

A go package for global optimization

## glbopt includes:

**Fibonacci** optimization used to minimize any 1D continuous function

**Shuffled Complex Evolution**-University of Arizona (SCE-UA - Duan et. al., 1993)

**SurrogateRBF:** A radial basis function surrogate model algorithm for computationally expensive nonlinear mixed-integer black-box global optimization problems (after Müller et. al., 2013; and Regis and Shoemaker, 2005)

## Example

```go
func main() {
	start := time.Now()
	uFib, yFib := glbopt.Fibonacci(griewank)
	xFib := mmaths.LinearTransform(-10., 15., uFib) // griewank
	fmt.Println("\ny=", yFib, "\tx=", xFib, "\tu:", uFib)
	fmt.Println(time.Now().Sub(start))
}

func griewank(u []float64) float64 {
	// x(i)~[-500,700], optimum at origin
	// there are errors in Duan etal (1993) and has been corrected here
	// see: Regis Shoemaker 2007 Supplement for A Stochastic Radial Basis Function Method for the Global Optimization of Expensive Functions
	xi := mmaths.LinearTransform(-10., 15., u[0])
	return math.Pow(xi, 2)/4000. - math.Cos(xi) + 1.
}
```

## dependencies:

* go montecarlo (https://github.com/maseology/montecarlo)
* go pseudo-random number generator (mm) (https://github.com/maseology/pnrg)
* mmplot (https://github.com/maseology/mmPlot)
* mmaths (https://github.com/maseology/mmaths)
* gonum (https://github.com/gonum/gonum)

## References

Duan, Q.Y., V.K. Gupta, and S. Sorooshian, 1993. Shuffled Complex Evolution Approach for Effective and Efficient Global Minimization. Journal of Optimization Theory and Applications 76(3) pp.501-521.

Müller, J., C.A. Shoemaker, and R. Piché 2013. SO-MI: A surrogate model algorithm for computationally expensive nonlinear mixed-integer black-box global optimization problems. Computers & Operations Research 40(5): 1383-1400.

Regis, R.G. and C.A. Shoemaker, 2005. Constrained Global Optimization of Expensive BlackBox Functions Using Radial Basis Functions. Journal of Global Optimization 31: 153–171.