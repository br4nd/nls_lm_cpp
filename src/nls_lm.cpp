////////////////////////////////////////////////////////////////////////////////
// nls_lm.cpp
// Examples of Localization and Autosurvey using Eigen Levenberg-Marquardt
//
// The Eigen::LM algorithm accepts as input "Functor" structures each contain
// an operator() method and a df() method.  The operator() contains the set of
// equations to be minimized ("constraints") and the df() method contains
// the Jacobian matrix of this set of equations.
//
// (BTW one may use Eigen::NumericalDiff() to estimate the gradient at each
// iteration rather than an explicit Jacobian.  But this requires a lot more
// computation and the interweb says we should use a manually generated df()
// function when possible.  Since the Jacobian for a set of simultaneous range
// equations is pretty straightforward I took this track.)
//
// I added the anchor array (AA) matrix and range measurement (Rm) vector to
// each "Functor" structure and named these structures FminMxN, where M is the
// number of equations (aka constraints) and N is the number of unknowns.
// For instance a 4 anchor, 1 mobile 3D localization "Functor" in this code is
// named Fmin4x3 since it implements 4 range equations to solve for the 3
// variables (x,y,z) of the mobile.
// Likewise, the 4 anchor autosurvey Functor is named "Fmin6x5" because it uses
// 6 range equations to solve for the 5 unknowns [x1, x2, y2, x3, y3].
//
// Each Functor definition is followed by its associated testbench function,
// FminMxN_test() which defines an Anchor Array (AA) and Mt (true Mobile
// location), adds random error for simulated Rm measurements, then provides
// an example for using the LM object to find M = (x,y,z).
//
// An associated Makefile assumes Eigen and unsupported/Eigen are installed and
// in the include path.  See http://eigen.tuxfamily.org/dox/GettingStarted.html
// and https://eigen.tuxfamily.org/dox/unsupported/index.html, and
// https://eigen.tuxfamily.org/dox/unsupported/classEigen_1_1LevenbergMarquardt.html
//
// Although the online tuxfamily.org LM documentation is pretty poor.
// Google "How to use eigen levenberg marquardt" instead.
//
// Brandon Dewberry Dec 7, 2017
////////////////////////////////////////////////////////////////////////////////
#include <iostream>
#include <Eigen/Dense>
#include <unsupported/Eigen/NonLinearOptimization>
#include <map>
// #include <unsupported/Eigen/NumericalDiff>
// #include <random>

// Uses Eigen/unsupported LevenbergMarquardt in NonLinearOptimization.h
// to localize.
// I got these return statuses from
// /usr/local/include/unsupported/Eigen/src/NonlinearOptimization/LevenbergMarquardt.h
std::map<int, std::string>
ret_status = {{-2, "NotStarted"},
          {-1, "Running"},
          { 0, "ImproperInputParameters"},
          { 1, "RelativeReductionTooSmall"},
          { 2, "RelativeErrorTooSmall"},
          { 3, "RelativeErrorAndReductionTooSmall"},
          { 4, "CosinusTooSmall"},
          { 5, "TooManyFunctionEvaluation"},
          { 6, "FtolTooSmall"},
          { 7, "XtolTooSmall"},
          { 8, "GtolTooSmall"},
          { 9, "UserAsked" }};

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

struct Fmin3Ax1M // 3 equations, 3 unknowns (Mx,My,Mz)
{
  // User enters Anchor Array locations and Range Measurements using these members before calling lm.minimize()
  Eigen::MatrixXd AA;
  Eigen::VectorXd Rm;

  // This operator() method is called many times by LM.
  int operator()(const Eigen::VectorXd &M, Eigen::VectorXd &fvec) const
  {
    fvec(0) = pow(M(0)-AA(0,0),2) + pow(M(1)-AA(0,1),2) + pow(M(2)-AA(0,2),2) - Rm(0)*Rm(0);
    fvec(1) = pow(M(0)-AA(1,0),2) + pow(M(1)-AA(1,1),2) + pow(M(2)-AA(0,2),2) - Rm(1)*Rm(1);
    fvec(2) = pow(M(0)-AA(2,0),2) + pow(M(1)-AA(2,1),2) + pow(M(2)-AA(0,2),2) - Rm(2)*Rm(2);
    return 0;
  }

  // This operator is called to compute the Jacobian, which is the multidimensional descent vector
  int df(const Eigen::VectorXd &M, Eigen::MatrixXd &fjac) const
  {
    fjac(0,0) = 2.0*(M(0)-AA(0,0));  fjac(0,1) = 2.0*(M(1)-AA(0,1));  fjac(0,2) = 2.0*(M(2)-AA(0,2));
    fjac(1,0) = 2.0*(M(0)-AA(1,0));  fjac(1,1) = 2.0*(M(1)-AA(1,1));  fjac(1,2) = 2.0*(M(2)-AA(1,2));
    fjac(2,0) = 2.0*(M(0)-AA(2,0));  fjac(2,1) = 2.0*(M(1)-AA(2,1));  fjac(2,2) = 2.0*(M(2)-AA(2,2));
    return 0;
  }

  // Eigen::LM API requires an values() function specifying the number of constraints.
  int values() const { return 3; }
};

void Fmin3Ax1M_test()
{
  std::cout << "\n Fmin3Ax1M_test: 3D (3 eqns, 3 unknowns M=x,y,z )" << std::endl;

  // Declare truth AA (Anchor Array) and Mt (true mobile location)
  Eigen::MatrixXd AA(3,3);
  AA <<   10.0, -10.0, 10.0,
          10.0,  10.0, 10.0,
         -10.0,  10.0, 10.0;
  std::cout << "AA:\n" << AA << std::endl;

  Eigen::VectorXd Mt(3);
  Mt << 1.0, 1.0, 0.5;
  std::cout << "Mt: (" << Mt(0) << ", " << Mt(1) << ", " << Mt(2) << ")" << std::endl;

  // Find Range Truth
  Eigen::VectorXd Rt(3);
  Rt(0) = pow(pow(Mt(0)-AA(0,0),2)+pow(Mt(1)-AA(0,1),2)+pow(Mt(2)-AA(0,2),2),0.5);
  Rt(1) = pow(pow(Mt(0)-AA(1,0),2)+pow(Mt(1)-AA(1,1),2)+pow(Mt(2)-AA(1,2),2),0.5);
  Rt(2) = pow(pow(Mt(0)-AA(2,0),2)+pow(Mt(1)-AA(2,1),2)+pow(Mt(2)-AA(2,2),2),0.5);
  std::cout << "Rt: (" << Rt(0) << ", " << Rt(1) << ", " << Rt(2) << ")" << std::endl;

  // Simulate some noisy ranges
  Eigen::VectorXd Rm(3), Rnoise(3);
  srand (time(NULL));
  float rmin = -0.02, rmax = 0.02;
  Rnoise(0) = (rmax-rmin)*rand()/double(RAND_MAX) + rmin;
  Rnoise(1) = (rmax-rmin)*rand()/double(RAND_MAX) + rmin;
  Rnoise(2) = (rmax-rmin)*rand()/double(RAND_MAX) + rmin;
  std::cout << "Rnoise: (" << Rnoise(0) << ", " << Rnoise(1) << ", " << Rnoise(2) << ")" << std::endl;
  Rm = Rt + Rnoise;
  std::cout << "Rm: (" << Rm(0) << ", " << Rm(1) << ", " << Rm(2) << ")" << std::endl;

  // Declare the seed point (somewhere below the anchor plane)
  Eigen::VectorXd M(3);
  M << 0.0, 0.0, 0.0;
  std::cout << "M(seed): (" << M(0) << ", " << M(1) << ", " << M(2) << ")" << std::endl;

  /////  NLS starts here
  Fmin3Ax1M Fmin;
  Fmin.AA = AA;
  Fmin.Rm = Rm;
  Eigen::LevenbergMarquardt<Fmin3Ax1M, double> lm(Fmin);
  lm.parameters.maxfev = 1000;  // max number of Functor (Fmin) evaluations (default 400)
  // I thought these settings would make a tighter solution, but not sure this does anything at all?
  lm.parameters.ftol = 1.0e-18;
  lm.parameters.xtol = 1.0e-18;
  lm.parameters.gtol = 1.0e-18;
  int ret = lm.minimize(M);
  std::cout << "M(optimal): (" << M(0) << ", " << M(1) << ", " << M(2) << ")" << std::endl;
  std::cout << "factor,  maxfev,   ftol,      xtol,     gtol,  epsfcn" << std::endl;
  std::cout << lm.parameters.factor << ",     " << lm.parameters.maxfev << ",     " << lm.parameters.ftol << ",     " << lm.parameters.xtol << ",    " << lm.parameters.gtol  << ",     " << lm.parameters.epsfcn << std::endl;
  std::cout << "iterations: " << lm.iter << ", retval: " << ret << "  " << ret_status[ret] << std::endl;
  Eigen::Vector3d Merr = M - Mt;
  std::cout << "Error = Mest - Mt = (" << Merr(0) << ", " << Merr(1) << ", " << Merr(2) << "); norm = " << Merr.norm() << std::endl;
}

struct Fmin4Ax1M // 4 equations, 3 unknowns (x,y,z)
{
  // User enters Anchor Array locations and Range Measurements using these members before calling lm.minimize()
  Eigen::MatrixXd AA;
  Eigen::VectorXd Rm;

  // This operator() method is called many times by LM.
  int operator()(const Eigen::VectorXd &M, Eigen::VectorXd &fvec) const
  {
    fvec(0) = pow(M(0)-AA(0,0),2) + pow(M(1)-AA(0,1),2) + pow(M(2)-AA(0,2),2) - Rm(0)*Rm(0);
    fvec(1) = pow(M(0)-AA(1,0),2) + pow(M(1)-AA(1,1),2) + pow(M(2)-AA(0,2),2) - Rm(1)*Rm(1);
    fvec(2) = pow(M(0)-AA(2,0),2) + pow(M(1)-AA(2,1),2) + pow(M(2)-AA(0,2),2) - Rm(2)*Rm(2);
    fvec(3) = pow(M(0)-AA(3,0),2) + pow(M(1)-AA(3,1),2) + pow(M(2)-AA(0,2),2) - Rm(3)*Rm(3);
    return 0;
  }

  // Requires M=(x,y) and returns computed Jacobian matrix
  int df(const Eigen::VectorXd &M, Eigen::MatrixXd &fjac) const
  {
    fjac(0,0) = 2.0*(M(0)-AA(0,0));  fjac(0,1) = 2.0*(M(1)-AA(0,1));  fjac(0,2) = 2.0*(M(2)-AA(0,2));
    fjac(1,0) = 2.0*(M(0)-AA(1,0));  fjac(1,1) = 2.0*(M(1)-AA(1,1));  fjac(1,2) = 2.0*(M(2)-AA(1,2));
    fjac(2,0) = 2.0*(M(0)-AA(2,0));  fjac(2,1) = 2.0*(M(1)-AA(2,1));  fjac(2,2) = 2.0*(M(2)-AA(2,2));
    fjac(3,0) = 2.0*(M(0)-AA(3,0));  fjac(3,1) = 2.0*(M(1)-AA(3,1));  fjac(3,2) = 2.0*(M(2)-AA(3,2));
    return 0;
  }

  int inputs() const { return 3; }
  int values() const { return 4; } // number of constraints
};

void Fmin4Ax1M_test()
{
  std::cout << "\n Fmin4Ax1M_test: 3D (4 eqns, 3 unknowns M=x,y,z )" << std::endl;

  // Declare truth AA (Anchor Array) and Mt (true mobile location)
  Eigen::MatrixXd AA(4,3);
  AA <<   10.0, -10.0, 10.0,
          10.0,  10.0, 10.0,
         -10.0,  10.0, 10.0,
         -10.0, -10.0, 10.0;
  std::cout << "AA:\n" << AA << std::endl;

  Eigen::VectorXd Mt(3);
  Mt << 1.0, 1.0, 0.5;
  std::cout << "Mt: (" << Mt(0) << ", " << Mt(1) << ", " << Mt(2) << ")" << std::endl;

  // Find Range Truth
  Eigen::VectorXd Rt(4);
  Rt(0) = pow(pow(Mt(0)-AA(0,0),2)+pow(Mt(1)-AA(0,1),2)+pow(Mt(2)-AA(0,2),2),0.5);
  Rt(1) = pow(pow(Mt(0)-AA(1,0),2)+pow(Mt(1)-AA(1,1),2)+pow(Mt(2)-AA(1,2),2),0.5);
  Rt(2) = pow(pow(Mt(0)-AA(2,0),2)+pow(Mt(1)-AA(2,1),2)+pow(Mt(2)-AA(2,2),2),0.5);
  Rt(3) = pow(pow(Mt(0)-AA(3,0),2)+pow(Mt(1)-AA(3,1),2)+pow(Mt(2)-AA(3,2),2),0.5);
  std::cout << "Rt: (" << Rt(0) << ", " << Rt(1) << ", " << Rt(2) << ", " << Rt(3) << ")" << std::endl;

  // Simulate some noisy ranges
  Eigen::VectorXd Rm(4);
  srand (time(NULL));
  Rm(0) = Rt(0) + 0.2*rand()/double(RAND_MAX) - 0.1;
  Rm(1) = Rt(1) + 0.2*rand()/double(RAND_MAX) - 0.1;
  Rm(2) = Rt(2) + 0.2*rand()/double(RAND_MAX) - 0.1;
  Rm(3) = Rt(3) + 0.2*rand()/double(RAND_MAX) - 0.1;
  std::cout << "Rm: (" << Rm(0) << ", " << Rm(1) << ", " << Rm(2) << ", " << Rm(3) << ")" << std::endl;

  // Declare the seed point (somewhere below the anchor plane)
  Eigen::VectorXd M(3);
  M << 0.0, 0.0, 0.0;
  std::cout << "M(seed): (" << M(0) << ", " << M(1) << ", " << M(2) << ")" << std::endl;

  /////  NLS starts here
  Fmin4Ax1M Fmin;
  Fmin.AA = AA;
  Fmin.Rm = Rm;
  Eigen::LevenbergMarquardt<Fmin4Ax1M, double> lm(Fmin);
  lm.parameters.maxfev = 1000;  // max number of Functor (Fmin) evaluations (default 400)
  // I thought these settings would make a tighter solution, but not sure this does anything at all?
  lm.parameters.ftol = 1.0e-18;
  lm.parameters.xtol = 1.0e-18;
  lm.parameters.gtol = 1.0e-18;
  int ret = lm.minimize(M);
  std::cout << "M(optimal): (" << M(0) << ", " << M(1) << ", " << M(2) << ")" << std::endl;
  std::cout << "factor,  maxfev,   ftol,      xtol,     gtol,  epsfcn" << std::endl;
  std::cout << lm.parameters.factor << ",     " << lm.parameters.maxfev << ",     " << lm.parameters.ftol << ",     " << lm.parameters.xtol << ",    " << lm.parameters.gtol  << ",     " << lm.parameters.epsfcn << std::endl;
  std::cout << "iterations: " << lm.iter << ", retval: " << ret << "  " << ret_status[ret] << std::endl;
  Eigen::Vector3d Merr = M - Mt;
  std::cout << "Error = Mest - Mt = (" << Merr(0) << ", " << Merr(1) << ", " << Merr(2) << "); norm = " << Merr.norm() << std::endl;
}

struct Fmin3A_autosurvey // 3 anchor autosurvey 2D - 3 range equations, 3 unknowns [x1, x2, y2]
{
  // User enters seed Anchor Array locations and Range Measurements Rm
  Eigen::MatrixXd AA;
  Eigen::VectorXd Rm;

  // This operator() method is called many times by LM.
  // Note the values being constrained are in vector U.
  // fvec is being iterativel forced to zero.
  int operator()(const Eigen::VectorXd &U, Eigen::VectorXd &fvec) const
  {
    // Define 3 constraint equations with U =[x1, x2, y2] as the vector of UNKNOWNS
    // Anchor 0 to Anchor 1 constraint (Range 0)
    fvec(0) = pow(AA(0,0)-U(0),2) + pow(AA(0,1)-AA(1,1),2) - Rm(0)*Rm(0);
    // Anchor 0 to Anchor 2 constraint (Range 1)
    fvec(1) = pow(AA(0,0)-U(1),2) + pow(AA(0,1)-U(2),2)    - Rm(1)*Rm(1);
    // Anchor 1 to Anchor 2 constraint (Range 2)
    fvec(2) = pow(U(0)-U(1),2)    + pow(AA(1,1)-U(2),2)    - Rm(2)*Rm(2);

    return 0;
  }

  // Requires U=(x1,x2,y2y) and returns computed Jacobian matrix
  int df(const Eigen::VectorXd &U, Eigen::MatrixXd &fjac) const
  {
    fjac(0,0) = -2.0*(AA(0,0)-U(0));  fjac(0,1) = 0.0;                  fjac(0,2) = 0.0;
    fjac(1,0) =  0.0;                 fjac(1,1) = -2.0*(AA(0,0)-U(1));  fjac(1,2) = -2.0*(AA(0,1)-U(2));
    fjac(2,0) =  2.0*(U(0)-U(1));     fjac(2,1) = -2.0*(U(0)-U(1));     fjac(2,2) = -2.0*(AA(1,1)-U(2));
    return 0;
  }

  int values() const { return 3; } // number of constraints
};

void Fmin3A_autosurvey_test()
{
  std::cout << "\n Fmin3A_autosurvey_test: 3 Anchor Autosurvey (3 range eqns, 3 unknowns U=[x1,x2,y2]" << std::endl;

  // Declare truth AAt (Anchor Array)
  Eigen::MatrixXd AAt(3,2), AA(3,2);
  AAt <<   0.0,  0.0,
           0.0, 10.0,
          10.0, 10.0;
  std::cout << "AAt:\n" << AAt << std::endl;
  // Declare AA seed (initial condition)
  AA <<   0.0,  0.0,
          1.0,  1.0,
          1.0,  1.0;
  std::cout << "AA(seed):\n" << AA << std::endl;

  // Find Range Truths Rt(0-2) in this order: [r01, r02, r12] where r01 is the distance between Anchor 0 and Anchor 1
  Eigen::VectorXd Rt(3);
  Rt(0) = std::sqrt(pow(AAt(0,0)-AAt(1,0),2)+pow(AAt(0,1)-AAt(1,1),2));
  Rt(1) = std::sqrt(pow(AAt(0,0)-AAt(2,0),2)+pow(AAt(0,1)-AAt(2,1),2));
  Rt(2) = std::sqrt(pow(AAt(1,0)-AAt(2,0),2)+pow(AAt(1,1)-AAt(2,1),2));
  std::cout << "Rt: (" << Rt(0) << ", " << Rt(1) << ", " << Rt(2) << ")" << std::endl;

  // Simulate some noisy range measurements
  Eigen::VectorXd Rm(3);
  srand (time(NULL));
  Rm(0) = Rt(0) + 0.2*rand()/double(RAND_MAX) - 0.1;
  Rm(1) = Rt(1) + 0.2*rand()/double(RAND_MAX) - 0.1;
  Rm(2) = Rt(2) + 0.2*rand()/double(RAND_MAX) - 0.1;
  std::cout << "Rm: (" << Rm(0) << ", " << Rm(1) << ", " << Rm(2) << ")" << std::endl;

  /////  NLS starts here
  Fmin3A_autosurvey Fmin;
  Fmin.AA = AA;
  Fmin.Rm = Rt;
  Eigen::LevenbergMarquardt<Fmin3A_autosurvey, double> lm(Fmin);
  // I thought smaller tolerances would make a tighter solution, but it doesn't seem to have an effect?
  lm.parameters.maxfev = 1000;  // max number of Functor (Fmin) evaluations (default 400)
  lm.parameters.ftol = 1.0e-18;
  lm.parameters.xtol = 1.0e-18;
  lm.parameters.gtol = 1.0e-18;
  Eigen::VectorXd U(3);
  // U = [x1, y3, z2]  unknowns
  U(0) = AA(1,0);  U(1) = AA(2,0);  U(2) = AA(2,1);
  int ret = lm.minimize(U);
  std::cout << "U(optimal): (" << U(0) << ", " << U(1) << ", " << U(2) << ")" << std::endl;
  std::cout << "factor,  maxfev,   ftol,      xtol,     gtol,  epsfcn" << std::endl;
  std::cout << lm.parameters.factor << ",     " << lm.parameters.maxfev << ",     " << lm.parameters.ftol << ",     " << lm.parameters.xtol << ",    " << lm.parameters.gtol  << ",     " << lm.parameters.epsfcn << std::endl;
  std::cout << "iterations: " << lm.iter << ", retval: " << ret << "  " << ret_status[ret] << std::endl;
  AA(1,0) = U(0);  AA(2,0) = U(1);  AA(2,1) = U(2);
  std::cout << "AA(optimal) =" << std::endl;
  std::cout << AA << std::endl;
  Eigen::MatrixXd AAerr(3,2);
  AAerr = AA - AAt;
  std::cout << "AAerr = AAest - AAt =" << std::endl;
  std::cout << AAerr << std::endl;
  std::cout << "AAerr.norm() = " << AAerr.norm() << std::endl;
}

struct Fmin4A_autosurvey // 4 anchor autosurvey: 6 equations, 5 unknowns [x1, x2, y2, x3, y3]
{
  // User enters seed Anchor Array locations and Range Measurements Rm
  Eigen::MatrixXd AA;
  Eigen::VectorXd Rm;

  // operator() is called each iteration to calculate the constraints as fvec is forced to zero.
  int operator()(const Eigen::VectorXd &U, Eigen::VectorXd &fvec) const
  {
    // Define 3 constraint equations with U =[x1, x2, y2, x3, y3] as the vector of UNKNOWNS
    // Redefine for convenience and clarity
    double x0 = AA(0,0); double y0 = AA(0,1); double z0 = AA(0,2);
    double x1 = U(0);    double y1 = AA(1,1); double z1 = AA(1,2);
    double x2 = U(1);    double y2 = U(2);    double z2 = AA(2,2);
    double x3 = U(3);    double y3 = U(4);    double z3 = AA(3,2);

    // Anchor 0 to Anchor 1 constraint 0 using Rm 0
    fvec(0) = pow(x0-x1,2) + pow(y0-y1,2) + pow(z0-z1,2) - Rm(0)*Rm(0);
    // Anchor 0 to Anchor 2 constraint (range 1)
    fvec(1) = pow(x0-x2,2) + pow(y0-y2,2) + pow(z0-z2,2) - Rm(1)*Rm(1);
    // Anchor 0 to Anchor 3 constraint (range 2)
    fvec(2) = pow(x0-x3,2) + pow(y0-y3,2) + pow(z0-z3,2) - Rm(2)*Rm(2);
    // Anchor 1 to Anchor 2 constraint (range 3)
    fvec(3) = pow(x1-x2,2) + pow(y1-y2,2) + pow(z1-z2,2) - Rm(3)*Rm(3);
    // Anchor 1 to Anchor 3 constraint (range 4)
    fvec(4) = pow(x1-x3,2) + pow(y1-y3,2) + pow(z1-z3,2) - Rm(4)*Rm(4);
    // Anchor 2 to Anchor 3 constraint (range 5)
    fvec(5) = pow(x2-x3,2) + pow(y2-y3,2) + pow(z2-z3,2) - Rm(5)*Rm(5);

    return 0;
  }

  // df() is called in each iteration to calculate the multidimensional direction of descent (Jacobian matrix)
  int df(const Eigen::VectorXd &U, Eigen::MatrixXd &fjac) const
  {
    // Redefine for convenience and clarity
    double x0 = AA(0,0); double y0 = AA(0,1); double z0 = AA(0,2);
    double x1 = U(0);    double y1 = AA(1,1); double z1 = AA(1,2);
    double x2 = U(1);    double y2 = U(2);    double z2 = AA(2,2);
    double x3 = U(3);    double y3 = U(4);    double z3 = AA(3,2);

    // Jacobian is size #constraints x #unknowns (in this case 6x5)
    // df01/dx1                df01/dx2                   df01/dy2                   df01/dx3                   df01/dy3
    fjac(0,0) = -2.0*(x0-x1);  fjac(0,1) = 0.0;           fjac(0,2) = 0.0;           fjac(0,3) = 0.0;           fjac(0,4) = 0.0;
    // df02/dx1                df02/dx2                   df02/dy2                   df02/dx3                   df02/dy3
    fjac(1,0) = 0.0;           fjac(1,1) = -2.0*(x0-x2);  fjac(1,2) = -2.0*(y0-y2);  fjac(1,3) = 0.0;           fjac(1,4) = 0.0;
    // df03/dx1                df03/dx2                   df03/dy2                   df03/dx3                   df03/dy3
    fjac(2,0) = 0.0;           fjac(2,1) = 0.0;           fjac(2,2) = 0.0;           fjac(2,3) = -2.0*(x0-x3);  fjac(2,4) = -2.0*(y0-y3);
    // df12/dx1                df12/dx2                   df12/dy2                   df12/dx3                   df12/dy3
    fjac(3,0) =  2.0*(x1-x2);  fjac(3,1) = -2.0*(x1-x2);  fjac(3,2) = -2.0*(y1-y2);  fjac(3,3) = 0.0;           fjac(3,4) = 0.0;
    // df13/dx1                df13/dx2                   df13/dy2                   df13/dx3                   df13/dy3
    fjac(4,0) =  2.0*(x1-x3);  fjac(4,1) = 0.0;           fjac(4,2) = 0.0;           fjac(4,3) = -2.0*(x1-x3);  fjac(4,4) = -2.0*(y1-y3);
    // df23/dx1                df23/dx2                   df23/dy2                   df23/dx3                   df23/dy3
    fjac(5,0) = 0.0;           fjac(5,1) = 2.0*(x1-x2);   fjac(5,2) = 2.0*(y2-y3);   fjac(5,3) = -2.0*(x2-x3);  fjac(5,4) = -2.0*(y2-y3);

    return 0;
  }

  int values() const { return 6; } // number of constraints
};

void Fmin4A_autosurvey_test()
{
  std::cout << "\n Fmin4A_autosurvey_test: 4 Anchor Autosurvey (6 range eqns, 5 unknowns U=[x1,x2,y2,x3,y3]" << std::endl;

  // Declare truth AAt (Anchor Array) and AA seed
  Eigen::MatrixXd AAt(4,3);
  AAt <<   0.0,  0.0, 0.0,
          15.0,  0.0, 0.0,
          12.0, 10.0, 0.0,
          -4.0, 10.0, 0.0;
  std::cout << "AAt:\n" << AAt << std::endl;

  Eigen::MatrixXd AA(4,3);
  AA = AAt;
  Eigen::VectorXd U(5);
  U << 1, 1, 1, 1, 1;  // Init unknowns  x1,  x2,  y2,  x3,  y3
  AA(1,0) = U(0);
  AA(2,0) = U(1);
  AA(2,1) = U(2);
  AA(3,0) = U(3);
  AA(3,1) = U(4);
  std::cout << "AA(seed):\n" << AA << std::endl;

  // Find Range Truths in this order: [r01, r02, r03, r12, r13, r23] where r01 is the distance between Anchor 0 and Anchor 1
  Eigen::VectorXd Rt(6);
  Rt(0) = std::sqrt(pow(AAt(0,0)-AAt(1,0),2) + pow(AAt(0,1)-AAt(1,1),2) + pow(AAt(0,2)-AAt(1,2),2));
  Rt(1) = std::sqrt(pow(AAt(0,0)-AAt(2,0),2) + pow(AAt(0,1)-AAt(2,1),2) + pow(AAt(0,2)-AAt(2,2),2));
  Rt(2) = std::sqrt(pow(AAt(0,0)-AAt(3,0),2) + pow(AAt(0,1)-AAt(3,1),2) + pow(AAt(0,2)-AAt(3,2),2));
  Rt(3) = std::sqrt(pow(AAt(1,0)-AAt(2,0),2) + pow(AAt(1,1)-AAt(2,1),2) + pow(AAt(1,2)-AAt(2,2),2));
  Rt(4) = std::sqrt(pow(AAt(1,0)-AAt(3,0),2) + pow(AAt(1,1)-AAt(3,1),2) + pow(AAt(1,2)-AAt(3,2),2));
  Rt(5) = std::sqrt(pow(AAt(2,0)-AAt(3,0),2) + pow(AAt(2,1)-AAt(3,1),2) + pow(AAt(2,2)-AAt(3,2),2));
  std::cout << "Rt: (" << Rt(0) << ", " << Rt(1) << ", " << Rt(2) << ", " << Rt(3) << ", " << Rt(4) << ", " << Rt(5) << ")" << std::endl;

  // Simulate some noisy range measurements
  Eigen::VectorXd Rm(6);
  srand (time(NULL));
  Rm(0) = Rt(0) + 0.2*rand()/double(RAND_MAX) - 0.1;
  Rm(1) = Rt(1) + 0.2*rand()/double(RAND_MAX) - 0.1;
  Rm(2) = Rt(2) + 0.2*rand()/double(RAND_MAX) - 0.1;
  Rm(3) = Rt(3) + 0.2*rand()/double(RAND_MAX) - 0.1;
  Rm(4) = Rt(4) + 0.2*rand()/double(RAND_MAX) - 0.1;
  Rm(5) = Rt(5) + 0.2*rand()/double(RAND_MAX) - 0.1;
  std::cout << "Rm: (" << Rm(0) << ", " << Rm(1) << ", " << Rm(2) << ", " << Rm(3) << ", " << Rm(4) << ", " << Rm(5) << ")" << std::endl;

  /////  NLS starts here
  Fmin4A_autosurvey Fmin;
  Fmin.AA = AA;
  Fmin.Rm = Rt;
  Eigen::LevenbergMarquardt<Fmin4A_autosurvey, double> lm(Fmin);
  // I thought smaller tolerances would make a tighter solution, but it doesn't seem to have an effect?
  lm.parameters.maxfev = 1000;  // max number of Functor (Fmin) evaluations (default 400)
  lm.parameters.ftol = 1.0e-18;
  lm.parameters.xtol = 1.0e-18;
  lm.parameters.gtol = 1.0e-18;

  int ret = lm.minimize(U);
  std::cout << "U(optimal): (" << U(0) << ", " << U(1) << ", " << U(2) << ", " << U(3) << ", " << U(4) << ")" << std::endl;
  std::cout << "factor,  maxfev,   ftol,      xtol,     gtol,  epsfcn" << std::endl;
  std::cout << lm.parameters.factor << ",     " << lm.parameters.maxfev << ",     " << lm.parameters.ftol << ",     " << lm.parameters.xtol << ",    " << lm.parameters.gtol  << ",     " << lm.parameters.epsfcn << std::endl;
  std::cout << "iterations: " << lm.iter << ", retval: " << ret << "  " << ret_status[ret] << std::endl;
  AA(1,0) = U(0);  AA(2,0) = U(1);  AA(2,1) = U(2);  AA(3,0) = U(3);  AA(3,1) = U(4);
  std::cout << "AA(optimal) =" << std::endl;
  std::cout << AA << std::endl;
  Eigen::MatrixXd AAerr(4,3);
  AAerr = AA - AAt;
  std::cout << "AAerr = AAest - AAt =" << std::endl;
  std::cout << AAerr << std::endl;
  std::cout << "AAerr.norm() = " << AAerr.norm() << std::endl;
}

int main(int argc, char *argv[])
{
  Fmin2Ax1M_test();
  Fmin3Ax1M_test();
  Fmin4Ax1M_test();
  Fmin3A_autosurvey_test();
  Fmin4A_autosurvey_test();
  return 0;
}
