#pragma once

#include "mapper/types.hpp"

#include <functional>
#include <vector>

namespace mapper {

// A filter (a.k.a. lens) maps a dataset to a low-dimensional real vector.
using FilterFunc = std::function<Eigen::VectorXd(const Dataset&)>;

// Projects each sample onto a selected set of columns.
// If a single column is provided, that column is returned as-is.
// If multiple columns are provided, their average is returned.
class Projection {
public:
    explicit Projection(std::vector<int> columns);

    Eigen::VectorXd operator()(const Dataset& data) const;

private:
    std::vector<int> columns_;
};

} // namespace mapper