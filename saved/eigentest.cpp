#include <iostream>
#include <Eigen/Dense>

int main(int argc, char *argv[])
{
    Eigen::VectorXd x(2);
    x(0) = 2.0;
    x(1) = 3.0;
    std::cout << "x: " << x << std::endl;

    return 0;
}
