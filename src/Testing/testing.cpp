#include <iostream>
#include <Eigen/Dense> // 需要先安裝 Eigen 庫

int main() {
    // 定義動態大小的矩陣
    Eigen::MatrixXd A(2, 3);
    Eigen::MatrixXd B(3, 2);

    A << 1, 2, 3,
         4, 5, 6;

    B << 7, 8,
         9, 10,
         11, 12;

    // 在 Eigen 中，這行可以直接編譯！而且速度極快（內建 SIMD 指令集優化）
    Eigen::MatrixXd C = A * B; 

    std::cout << C << std::endl;
    return 0;
}