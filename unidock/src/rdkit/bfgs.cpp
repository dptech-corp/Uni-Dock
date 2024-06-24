#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <Eigen/Dense>
#include <unsupported/Eigen/NonLinearOptimization>
#include <unsupported/Eigen/NumericalDiff>

// 假设一个计算势能的函数
double calculatePotentialEnergy(const Eigen::Vector3d& position) {
    // 这里是一个简单的势能函数示例
    return std::sin(position.x()) * std::cos(position.y()) * std::exp(-position.z());
}

// 生成空间网格的势能面
std::vector<std::vector<std::vector<double>>> generatePotentialGrid(const Eigen::Vector3d& min, const Eigen::Vector3d& max, const Eigen::Vector3i& steps) {
    std::vector<std::vector<std::vector<double>>> grid(steps.x(), std::vector<std::vector<double>>(steps.y(), std::vector<double>(steps.z())));

    Eigen::Vector3d stepSize = (max - min).cwiseQuotient(steps.cast<double>());

    for (int i = 0; i < steps.x(); ++i) {
        for (int j = 0; j < steps.y(); ++j) {
            for (int k = 0; k < steps.z(); ++k) {
                Eigen::Vector3d position = min + Eigen::Vector3d(i, j, k).cwiseProduct(stepSize);
                grid[i][j][k] = calculatePotentialEnergy(position);
            }
        }
    }

    return grid;
}

// 假设一个根据torsion角度计算位置的函数
Eigen::Vector3d calculatePosition(double angle, int torsionIndex) {
    // 这里是一个简单的计算位置的函数示例，根据torsion角度和索引计算位置
    // 在实际应用中，这个函数应该根据分子的具体结构进行计算
    return Eigen::Vector3d(std::cos(angle), std::sin(angle), torsionIndex);
}

// 优化问题结构体
struct TorsionOptimizationFunctor {
    typedef double Scalar;
    typedef Eigen::VectorXd InputType;
    typedef Eigen::VectorXd ValueType;
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> JacobianType;

    enum { InputsAtCompileTime = Eigen::Dynamic };
    enum { ValuesAtCompileTime = Eigen::Dynamic };

    const Eigen::Vector3d& min;
    const Eigen::Vector3d& max;
    const Eigen::Vector3i& steps;
    const std::vector<std::vector<std::vector<double>>>& potentialGrid;

    TorsionOptimizationFunctor(const Eigen::Vector3d& min, const Eigen::Vector3d& max, const Eigen::Vector3i& steps, const std::vector<std::vector<std::vector<double>>>& grid)
        : min(min), max(max), steps(steps), potentialGrid(grid) {}

    int operator()(const InputType& x, ValueType& fvec) const {
        for (int i = 0; i < x.size(); ++i) {
            double angle = x[i];
            Eigen::Vector3d position = calculatePosition(angle, i);
            Eigen::Vector3i idx = ((position - min).cwiseQuotient((max - min).cwiseQuotient(steps.cast<double>()))).cast<int>();
            if (idx.x() < 0 || idx.y() < 0 || idx.z() < 0 || idx.x() >= steps.x() || idx.y() >= steps.y() || idx.z() >= steps.z()) {
                std::cerr << "Index out of bounds: (" << idx.x() << ", " << idx.y() << ", " << idx.z() << ")" << std::endl;
                return -1;
            }
            fvec[i] = potentialGrid[idx.x()][idx.y()][idx.z()];
        }
        return 0;
    }

    int inputs() const { return InputsAtCompileTime; }
    int values() const { return ValuesAtCompileTime; }
};

// 使用BFGS优化torsion角度，并保存优化过程
Eigen::VectorXd optimizeTorsionAngles(const Eigen::VectorXd& initialTorsions, const Eigen::Vector3d& min, const Eigen::Vector3d& max, const Eigen::Vector3i& steps, const std::vector<std::vector<std::vector<double>>>& potentialGrid) {
    TorsionOptimizationFunctor functor(min, max, steps, potentialGrid);
    Eigen::NumericalDiff<TorsionOptimizationFunctor> numDiff(functor);
    Eigen::LevenbergMarquardt<Eigen::NumericalDiff<TorsionOptimizationFunctor>, double> lm(numDiff);

    Eigen::VectorXd optimizedTorsions = initialTorsions;

    // 打开文件保存优化过程
    std::ofstream outfile("optimization_log.csv");
    outfile << "Iteration,Angle1,Angle2,Energy\n";

    // 定义一个Lambda函数来打印每次迭代的结果
    lm.parameters.maxfev = 100;  // 最大迭代次数
    lm.parameters.ftol = 1.0e-10; // 收敛容限

    int iter = 0;
    auto printProgress = [&](const Eigen::VectorXd& x, const Eigen::VectorXd& fvec) {
        double energy = fvec.sum(); // 简单求和示例，可以根据需要修改
        std::cout << "Iteration: " << iter << ", Angles: " << x.transpose() << ", Energy: " << energy << std::endl;
        outfile << iter << "," << x.transpose() << "," << energy << "\n";
        iter++;
    };

    // 设置LM算法的回调函数
    lm.minimizeInit(optimizedTorsions);
    do {
        lm.minimizeOneStep(optimizedTorsions);
        printProgress(optimizedTorsions, lm.fvec);
    } while (lm.nfev < lm.parameters.maxfev && lm.fnorm > lm.parameters.ftol);

    outfile.close();
    return optimizedTorsions;
}

int main() {
    std::cout << "Generating potential grid..." << std::endl;
    Eigen::Vector3d min(-10, -10, -10);
    Eigen::Vector3d max(10, 10, 10);
    Eigen::Vector3i steps(100, 100, 100);

    // 生成势能网格
    auto potentialGrid = generatePotentialGrid(min, max, steps);
    std::cout << "Potential grid generated." << std::endl;

    // 初始的torsion角度
    Eigen::VectorXd initialTorsions(2); // 假设有两个torsion角度
    initialTorsions << 0.0, 0.0;

    // 优化torsion角度
    std::cout << "Starting optimization..." << std::endl;
    auto optimizedTorsions = optimizeTorsionAngles(initialTorsions, min, max, steps, potentialGrid);

    std::cout << "Optimization finished." << std::endl;
    std::cout << "Optimized Torsion Angles: " << optimizedTorsions.transpose() << std::endl;

    return 0;
}
