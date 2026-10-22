#include "mapper/filter.hpp"

#include <utility>

namespace mapper {

Projection::Projection(std::vector<int> columns)
    : columns_(std::move(columns)) {}

Eigen::VectorXd Projection::operator()(const Dataset& data) const {
    if (data.rows() == 0) {
        return Eigen::VectorXd();
    }

    // Default behavior: use the first column when no column is specified.
    if (columns_.empty()) {
        return data.col(0);
    }

    if (columns_.size() == 1) {
        return data.col(columns_[0]);
    }

    Eigen::VectorXd out = Eigen::VectorXd::Zero(data.rows());
    for (int c : columns_) {
        out += data.col(c);
    }
    out /= static_cast<double>(columns_.size());
    return out;
}

} // namespace mapper