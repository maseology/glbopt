package glbopt

const (
	// uncertaintyLength       = 1.e-10 // range disctretization
	distinguishabilityConst = 1.e-10
)

// Fibonacci optimization used to minimize any 1D continuous function
//  ref: Bazaraa, M.S., H.D. Sherali, and C.M. Shetty, 2006. Nonlinear Programming: Theory and Algorithms, 3rd ed. John Wiley & Sonc, Inc. New Jersey. 853pp.
//  UncertaintyLength = 0.01 'uncertainty length (l)
//  DistinguishabilityConst = 0.01 'distinguishability constant (e)
//  Sample range can be set to anything. Keeping consistent with other glbopt funcations, range hard coded to U[0.,1.]
//  This is only a 1-parameter optimizer, but need to keep slice variable input to maintain interface compatibility
func Fibonacci(fun func(u1 []float64) float64) (float64, float64) {
	// initilization step
	const (
		n  = 91 // largest Fibonacci number F(91) before overflow
		a1 = 0. // range min
		b1 = 1. // range max
	)

	// prime go functions
	done := make(chan interface{})
	defer close(done)
	fun2 := func(done <-chan interface{}, v float64) <-chan float64 {
		r := make(chan float64)
		go func() {
			defer close(r)
			for {
				select {
				case r <- fun([]float64{v}):
				case <-done:
					return
				}
			}
		}()
		return r
	}

	// main step
	ff1, ff2 := make([]float64, n+1), make([]float64, n+1)
	ak, bk, k := make([]float64, n+1), make([]float64, n+1), 1
	ak[k] = a1
	bk[k] = b1
	ff1[k] = fibonacciSample(ak[k], bk[k], float64(fibonacci(n-2)), float64(fibonacci(n))) // lambda
	ff2[k] = fibonacciSample(ak[k], bk[k], float64(fibonacci(n-1)), float64(fibonacci(n))) // mu
	ch1, ch2 := fun2(done, ff1[k]), fun2(done, ff2[k])
	ffr1, ffr2 := <-ch1, <-ch2

	for {
		if ffr1 > ffr2 { // step 2
			ak[k+1] = ff1[k]
			bk[k+1] = bk[k]
			ff1[k+1] = ff2[k]
			ff2[k+1] = fibonacciSample(ak[k+1], bk[k+1], float64(fibonacci(n-k-1)), float64(fibonacci(n-k)))
			if k == n-2 {
				break
			}
			ffr1 = ffr2
			ffr2 = fun([]float64{ff2[k+1]})
		} else { // step 3
			bk[k+1] = ff2[k]
			ak[k+1] = ak[k]
			ff2[k+1] = ff1[k]
			ff1[k+1] = fibonacciSample(ak[k+1], bk[k+1], float64(fibonacci(n-k-2)), float64(fibonacci(n-k)))
			if k == n-2 {
				break
			}
			ffr2 = ffr1
			ffr1 = fun([]float64{ff1[k+1]})
		}
		k++ // step 4
		print(".")
	}

	// step 5
	ff1[n] = ff1[n-1]
	ff2[n] = ff1[n-1] + distinguishabilityConst
	if ff2[n] > b1 {
		ff2[n] = b1
	}
	ffr1 = fun([]float64{ff1[n]})
	ffr2 = fun([]float64{ff2[n]})
	if ffr1 > ffr2 {
		ak[n] = ff1[n]
		bk[n] = bk[n-1]
	} else {
		ak[n] = ak[n-1]
		bk[n] = ff1[n]
	}

	// the optimum solution lies in the interval [ak(n), bk(n)], thus take average value
	uopt := []float64{0.5 * (ak[n] + bk[n])}
	return uopt[0], fun(uopt)
}

func fibonacciSample(a, b, fibonacciNumer, fibonacciDenom float64) float64 {
	return a + fibonacciNumer/fibonacciDenom*(b-a) // for Fibonacci iteration
}

func fibonacci(n int) int {
	max := n
	if max < 2 {
		max = 2 // to avoid errors
	}
	f := make([]int, max+1)
	f[0] = 1
	f[1] = 1
	for i := 2; i <= max; i++ {
		f[i] = f[i-1] + f[i-2]
	}
	return f[n]
}
