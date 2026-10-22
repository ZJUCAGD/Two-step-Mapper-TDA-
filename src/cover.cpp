#include "mapper/cover.hpp"

#include <nanoflann.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace mapper {

// ---------------------------------------------------------------------------
// nanoflann adaptor over an Eigen matrix (one point per row).
// ---------------------------------------------------------------------------
namespace {

struct PointCloudAdaptor {
    const Dataset& data;

    explicit PointCloudAdaptor(const Dataset& d) : data(d) {}

    inline size_t kdtree_get_point_count() const {
        return static_cast<size_t>(data.rows());
    }

    template <typename IndexType, typename DimType>
    inline double kdtree_get_pt(const IndexType idx, const DimType dim) const {
        return data(static_cast<Eigen::Index>(idx),
                    static_cast<Eigen::Index>(dim));
    }

    template <class BBOX>
    bool kdtree_get_bbox(BBOX& /*bb*/) const {
        return false;
    }
};

using DistanceAdaptor =
    nanoflann::L2_Simple_Adaptor<double, PointCloudAdaptor>;

using KDTree = nanoflann::KDTreeSingleIndexAdaptor<
    DistanceAdaptor, PointCloudAdaptor, -1, uint32_t>;

} // namespace

// ---------------------------------------------------------------------------
// OneDimensionalCover
// ---------------------------------------------------------------------------

OneDimensionalCover::OneDimensionalCover(double min_val, double max_val,
                                         int n_intervals, double overlap_frac)
    : n_intervals_(n_intervals < 1 ? 1 : n_intervals),
      overlap_frac_(overlap_frac) {

    constexpr double kInf = std::numeric_limits<double>::infinity();

    // Reference semantics: overlap must lie strictly inside (0, 0.5).
    if (overlap_frac_ <= 0.0) {
        overlap_frac_ = 0.25;
    }
    if (overlap_frac_ >= 0.5) {
        overlap_frac_ = 0.49;
    }

    double range = max_val - min_val;
    if (range <= 0.0) {
        // Degenerate range: every point shares the same filter value.
        n_intervals_ = 1;
        intervals_.clear();
        intervals_.emplace_back(-kInf, kInf);
        return;
    }

    // Let L be the interval length and s = L * (1 - overlap) the step between
    // consecutive intervals. Imposing that the first interval starts at
    // min_val and the last one ends at max_val gives:
    //     L * ((n - 1) * (1 - overlap) + 1) = range
    const double denom = static_cast<double>(n_intervals_ - 1) *
                         (1.0 - overlap_frac_) + 1.0;
    const double L = range / denom;
    const double s = L * (1.0 - overlap_frac_);

    intervals_.resize(n_intervals_);
    for (int i = 0; i < n_intervals_; ++i) {
        const double lo = min_val + static_cast<double>(i) * s;
        intervals_[i] = {lo, lo + L};
    }

    // Extend the two extremes to infinity so that min_val and max_val are
    // strictly inside the first and last intervals.
    intervals_.front().first = -kInf;
    intervals_.back().second =  kInf;
}

std::vector<int> OneDimensionalCover::which(double v) const {
    std::vector<int> out;
    for (int i = 0; i < n_intervals_; ++i) {
        if (contains(i, v)) out.push_back(i);
    }
    return out;
}

// ---------------------------------------------------------------------------
// Adaptive cover utilities
// ---------------------------------------------------------------------------

double compute_eps_len(const Dataset& data,
                       const Eigen::VectorXd& func_val,
                       double eps) {
    const int n = static_cast<int>(data.rows());
    if (n <= 1 || eps <= 0.0) {
        return 0.0;
    }

    PointCloudAdaptor adaptor(data);
    KDTree tree(static_cast<int>(data.cols()), adaptor,
                nanoflann::KDTreeSingleIndexAdaptorParams(10));

    const double squared_radius = eps * eps;
    const double f_global_max   = func_val.maxCoeff();
    const double f_global_min   = func_val.minCoeff();
    const double theoretical_max = f_global_max - f_global_min;

    using ResultItem = nanoflann::ResultItem<uint32_t, double>;

    std::vector<ResultItem> matches;
    nanoflann::SearchParameters params;
    Eigen::VectorXd query(data.cols());

    double omega = 0.0;

    for (int i = 0; i < n; ++i) {
        query = data.row(i);

        matches.clear();
        tree.radiusSearch(query.data(), squared_radius, matches, params);

        const double f_i = func_val[i];
        double local_max = f_i;
        double local_min = f_i;
        for (const auto& m : matches) {
            const double f_j = func_val[m.first];
            if (f_j > local_max) local_max = f_j;
            if (f_j < local_min) local_min = f_j;
        }

        const double diff = std::max(local_max - f_i, f_i - local_min);
        if (diff > omega) {
            omega = diff;
        }
        if (omega >= theoretical_max) {
            break;
        }
    }

    return omega;
}

double compute_cover_len(const Dataset& data,
                         const Eigen::VectorXd& func_val,
                         double eps,
                         double overlap) {
    if (overlap <= 0.0) {
        return 0.0;
    }
    const double omega = compute_eps_len(data, func_val, eps);
    return 1.01 * omega / overlap;
}

int compute_cover_num(const Dataset& data,
                      const Eigen::VectorXd& func_val,
                      double eps,
                      double overlap) {
    const double f_max = func_val.maxCoeff();
    const double f_min = func_val.minCoeff();
    const double cover_len = compute_cover_len(data, func_val, eps, overlap);

    if (cover_len <= 0.0 || overlap >= 0.5) {
        return 1;
    }

    const double num =
        (f_max - f_min - cover_len * overlap) / (cover_len * (1.0 - overlap));

    return std::max(1, static_cast<int>(num));
}

} // namespace mapper