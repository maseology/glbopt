# go glbopt

A go package for global optimization

## glbopt includes:

**Shuffled Complex Evolution**-University of Arizona (SCE-UA - Duan et. al., 1993)

**SurrogateRBF:** A radial basis function surrogate model algorithm for computationally expensive nonlinear mixed-integer black-box global optimization problems (after Müller et. al., 2013; and Regis and Shoemaker, 2005)

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