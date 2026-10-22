#pragma once

#include <Eigen/Dense>
#include <cstddef>
#include <vector>

namespace mapper {

// A single data point represented as a column vector.
using Point   = Eigen::VectorXd;

// A dataset represented as a row-major matrix: rows are samples, columns are features.
using Dataset = Eigen::MatrixXd;

// Identifier used to index nodes inside a mapper graph.
using NodeId  = std::size_t;


} // namespace mapper