#pragma once

#include "mapper/types.hpp" 

#include <limits>
#include <utility>
#include <vector>

namespace mapper {

// A one-dimensional cover made of uniformly spaced, overlapping intervals.
class OneDimensionalCover {
public:
    OneDimensionalCover(double min_val, double max_val,
                        int n_intervals, double overlap_frac);

    int size() const { return n_intervals_; }

    std::vector<int> which(double v) const;

    const std::pair<double, double>& interval(int i) const {
        return intervals_[i];
    }

    
    bool contains(int i, double v) const {
        return v > intervals_[i].first && v < intervals_[i].second;
    }

private:
    int n_intervals_;
    double overlap_frac_;
    std::vector<std::pair<double, double>> intervals_; 
};

// ---------------------------------------------------------------------------
// Adaptive cover utilities
// ---------------------------------------------------------------------------

// Modulus of continuity of the filter at scale eps:
//     omega(eps) = max { |f(x_i) - f(x_j)| : d(x_i, x_j) <= eps }.
double compute_eps_len(const Dataset& data,
                       const Eigen::VectorXd& func_val,
                       double eps);

// Cover interval length implied by omega(eps)
double compute_cover_len(const Dataset& data,
                         const Eigen::VectorXd& func_val,
                         double eps,
                         double overlap);

// Number of intervals implied by the cover length
int compute_cover_num(const Dataset& data,
                      const Eigen::VectorXd& func_val,
                      double eps,
                      double overlap);

} // namespace mapper