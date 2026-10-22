#include "mapper/embedding.hpp"

namespace mapper {

Eigen::MatrixXd embedding_2d(const Dataset& data) {
    const Eigen::Index n = data.rows();
    Eigen::MatrixXd out = Eigen::MatrixXd::Zero(n, 2);

    if (data.cols() >= 1) {
        out.col(0) = data.col(0);
    }
    if (data.cols() >= 2) {
        out.col(1) = data.col(1);
    }
    return out;
}

} // namespace mapper