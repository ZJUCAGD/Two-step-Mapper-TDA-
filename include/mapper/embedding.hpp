#pragma once

#include "mapper/types.hpp"

namespace mapper {

// Returns an N x 2 matrix where each row corresponds to one sample.
// The first two features of the dataset are used directly as 2D coordinates.
// If the dataset has fewer than 2 columns, the missing column is filled
// with zeros.
Eigen::MatrixXd embedding_2d(const Dataset& data);

} // namespace mapper