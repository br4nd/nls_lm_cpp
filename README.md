# Nonlinear Least Squares for Localization and Autosurvey using Levenberg-Marquardt from unsupported/Eigen

This folder provides a c++ source file (nls_lm.cpp) and associated Makefile with
examples of how to use the Eigen::Levenberg-Marquardt class to solve a number
of Localization and Autosurvey problems.

## Requirements

* c++11: for the convenience of initializer lists.  The example Makefile builds with g++ --std=c++0x.
* Eigen (http://eigen.tuxfamily.org/index.php?title=Main_Page) library in the include path.  I put mine at /usr/local/include/Eigen.
* Eigen/unsupported library (https://eigen.tuxfamily.org/dox/unsupported/index.html). I put copied mine to /usr/local/include/Eigen/unsupported, then created a symbolic link.

## What's in nls_lm.cpp

nls_lm.cpp contains 6 example "Functor" structures together with their 6 test functions.
The main() simply calles the testbenches in order.
```
struct Fmin2Ax1M  // 2 Anchors, 1 mobile, 2D
void Fmin2Ax1M_test()

struct Fmin3Ax1M // 3 equations, 3 unknowns (Mx,My,Mz)
void Fmin3Ax1M_test()

struct Fmin4Ax1M // 4 equations, 3 unknowns (Mx,My,Mz)
void Fmin2Ax1M_test()

struct Fmin3Ax1M // 3 equations, 3 unknowns (Mx,My,Mz)
void Fmin3Ax1M_test()

struct Fmin4Ax1M // 4 equations, 3 unknowns (x,y,z)
void Fmin4Ax1M_test()

struct Fmin3A_autosurvey // 3 anchor autosurvey 2D - 3 range equations, 3 unknowns [x1, x2, y2]
void Fmin4A_autosurvey_test()

int main(int argc, char *argv[])
{
  Fmin2Ax1M_test();
  Fmin3Ax1M_test();
  Fmin4Ax1M_test();
  Fmin3A_autosurvey_test();
  Fmin4A_autosurvey_test();
  return 0;
}
```

## How it works

The Eigen::LM algorithm accepts as input "Functor" structures each contain an operator() method and a df() method.  The operator() contains the set of
equations to be minimized ("constraints") and the df() method contains the Jacobian matrix of this set of equations.

### Example of Functor structure

The authors call these structures "Functors".  I renamed them based on the problem they are solving.
The simplest example is a 2D solver that finds M = (x,y) given two anchor locations (AA) and two range measurements (in vector Rm):

```cpp
struct Fmin2Ax1M  // 2 Anchors, 1 mobile, 2D
{
  // This Functor structure is used to solve for 2 equations rn=sqrt((x-x0)^2+(y-y0)^2, n=0,1 and
  // finds 2 unknowns M = [x,y].

  // User enters Anchor Array locations and Range Measurements using these members before calling lm.minimize()
  Eigen::MatrixXd AA;
  Eigen::VectorXd Rm;

  // This operator() method is iteratively called by LM.
  // M is (x_seed, y_seed) which eventually contains the localization estimat M=(x_hat, y_hat),
  // The second argument is a pointer to the equations that are simultaneously minimized.
  int operator()(const Eigen::VectorXd &M, Eigen::VectorXd &fvec) const
  {
    fvec(0) = pow(M(0)-AA(0,0),2) + pow(M(1)-AA(0,1),2) - Rm(0)*Rm(0);
    fvec(1) = pow(M(0)-AA(1,0),2) + pow(M(1)-AA(1,1),2) - Rm(1)*Rm(1);
    return 0;
  }

  // Requires M=(x,y) and returns computed Jacobian matrix
  int df(const Eigen::VectorXd &M, Eigen::MatrixXd &fjac) const
  {
    fjac(0,0) = 2.0*(M(0)-AA(0,0));  fjac(0,1) = 2.0*(M(1)-AA(0,1));
    fjac(1,0) = 2.0*(M(0)-AA(1,0));  fjac(1,1) = 2.0*(M(1)-AA(1,1));
    return 0;
  }

  // Eigen::LM alg requires a values() function specifying the number of constraints (equations).
  int values() const { return 2; } // number of constraints
};
```
Other than operator() and df() methods, the Eigen::LM API also mandates a values() method.
This returns the number of "constraints" or "equations" in the operator() method.

I added the AA and Rm variables to the structure for our purposes.

BTW one may use Eigen::NumericalDiff() to estimate the gradient at each iteration rather than an explicit Jacobian.  
But this requires a lot more computation than if you explicitly code the Jacobian in the df() method.
Since the Jacobians for a set of simultaneous range equations is pretty straightforward I took this track.

### Using Fmin2Ax1M

To use Fmin2Ax1M to solve for optimal x,y:
1. create an anchor array (AA), a measured range vector (Rm), and a seed location (M)
2. create the Functor structure Fmin
Fmin2Ax1M Fmin;
3. create the LevenbergMarquardt object (lm) with Fmin as init arg.
Eigen::LevenbergMarquardt<Fmin2Ax1M, double> lm(Fmin);
3. pass AA and Rm to the structure
Fmin.AA = AA;
Fmin.Rm = Rm;
4. call lm.minimize(M)

M will then contain the optimal values, and the lm object will contain a bunch of things that can be inspected such as #iterations, and what caused it to exit.

You can also change the default tolerances (how small the error needs to be before exiting, but I didn't find much difference in performance.

Here's the actual testbench code from nls_lm.cpp:

```
void Fmin2Ax1M_test()
{
  // Function for testing Fmin2x2
  std::cout << "\n Fmin2Ax1M_test: 2D (2 eqns, 2 unknowns M=x,y )" << std::endl;

  // Set truth AA (Anchor Array) and Mt (true mobile location)
  Eigen::MatrixXd AA(2,2);
  AA <<  0.0, -1.0,
         0.0, 1.0;
  std::cout << "AA:\n" << AA << std::endl;

  Eigen::VectorXd Mt(2);
  Mt << 10.0, 0.5;
  std::cout << "Mt: (" << Mt(0) << ", " << Mt(1) << ")" << std::endl;

  // Rt is Range Truth
  Eigen::VectorXd Rt(2);
  Rt(0) = pow(pow(Mt(0)-AA(0,0),2)+pow(Mt(1)-AA(0,1),2),0.5);
  Rt(1) = pow(pow(Mt(0)-AA(1,0),2)+pow(Mt(1)-AA(1,1),2),0.5);
  std::cout << "Rt: (" << Rt(0) << ", " << Rt(1) << ")" << std::endl;

  // Rm is simulated measurements (random noise added to Rt)
  Eigen::VectorXd Rm(2), Rnoise(2);
  srand (time(NULL));
  float rmin = -0.02, rmax = 0.02;
  Rnoise(0) = (rmax-rmin)*rand()/double(RAND_MAX) + rmin;
  Rnoise(1) = (rmax-rmin)*rand()/double(RAND_MAX) + rmin;
  std::cout << "Rnoise: (" << Rnoise(0) << ", " << Rnoise(1) << ")" << std::endl;
  Rm = Rt + Rnoise;
  std::cout << "Rm: (" << Rm(0) << ", " << Rm(1) << ")" << std::endl;

  Eigen::VectorXd M(2), Ms(2);
  M << 1, 0;  // Not necessary to save the seed unless  compute error later
  std::cout << "M(seed): (" << M(0) << ", " << M(1) << ")" << std::endl;

  /////  NLS starts here
  Fmin2Ax1M Fmin;
  Eigen::LevenbergMarquardt<Fmin2Ax1M, double> lm(Fmin);
  Fmin.AA = AA;
  Fmin.Rm = Rm;
  lm.parameters.maxfev = 1000;  // max number of Functor (Fmin) evaluations (default 400)
  // I thought these settings would make a tighter solution, but not sure this does anything at all?
  lm.parameters.ftol = 1.0e-18;
  lm.parameters.xtol = 1.0e-18;
  lm.parameters.gtol = 1.0e-18;
  int ret = lm.minimize(M);
  std::cout << "M(optimal) = (" << M(0) << ", " << M(1) << ")"<< std::endl;
  std::cout << "factor,  maxfev,   ftol,      xtol,     gtol,  epsfcn" << std::endl;
  std::cout << lm.parameters.factor << ",     " << lm.parameters.maxfev << ",     " << lm.parameters.ftol << ",     " << lm.parameters.xtol << ",    " << lm.parameters.gtol  << ",     " << lm.parameters.epsfcn << std::endl;
  std::cout << "iterations: " << lm.iter << ", retval: " << ret << "  " << ret_status[ret] << std::endl;
  Eigen::Vector2d Merr = M - Mt;
  std::cout << "Error = Mest - Mt = (" << Merr(0) << ", " << Merr(1) << "); norm = " << Merr.norm() << std::endl;
}
```


## Other thoughts

The online tuxfamily.org LM documentation is pretty poor.  Google "How to use eigen levenberg marquardt" instead.
Also inspect /usr/local/include/eigen3/unsupported/test/levenberg_marquardt.cpp
