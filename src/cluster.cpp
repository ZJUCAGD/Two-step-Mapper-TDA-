#include "mapper/cluster.hpp"

#include <nanoflann.hpp>

#include <algorithm>
#include <functional>
#include <numeric>
#include <unordered_map>
#include <vector>

namespace mapper {

// ===========================================================================
// Shared nanoflann adaptor over an Eigen matrix.
// ===========================================================================
namespace {

struct PointCloudAdaptor {
    const Eigen::MatrixXd& data;

    explicit PointCloudAdaptor(const Eigen::MatrixXd& d) : data(d) {}

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

using ResultItem = nanoflann::ResultItem<uint32_t, double>;

} // namespace

// ===========================================================================
// DBSCAN
// ===========================================================================

struct DBSCAN::Impl {
    // Declaration order matters: the tree holds a reference to the adaptor,
    // so the adaptor must be destroyed last.
    std::unique_ptr<PointCloudAdaptor> adaptor;
    std::unique_ptr<KDTree>            tree;
};

DBSCAN::DBSCAN(double eps, int min_samples)
    : eps_(eps), min_samples_(min_samples), impl_(std::make_unique<Impl>()) {}

DBSCAN::~DBSCAN() = default;
DBSCAN::DBSCAN(DBSCAN&&) noexcept = default;
DBSCAN& DBSCAN::operator=(DBSCAN&&) noexcept = default;

std::vector<int> DBSCAN::region_query(const Dataset& data, int i) const {
    std::vector<int> neighbors;

    if (!impl_->tree) {
        const Eigen::VectorXd pi = data.row(i);
        const double eps2 = eps_ * eps_;
        for (int j = 0; j < data.rows(); ++j) {
            if ((pi - data.row(j)).squaredNorm() <= eps2) {
                neighbors.push_back(j);
            }
        }
        return neighbors;
    }

    const double squared_radius = eps_ * eps_;
    const Eigen::VectorXd query = data.row(i);

    std::vector<ResultItem> matches;
    nanoflann::SearchParameters params;

    const size_t n_found = impl_->tree->radiusSearch(
        query.data(), squared_radius, matches, params);

    neighbors.reserve(n_found);
    for (const auto& m : matches) {
        neighbors.push_back(static_cast<int>(m.first));
    }
    return neighbors;
}

std::vector<int> DBSCAN::fit_predict(const Dataset& data) const {
    const int n = static_cast<int>(data.rows());

    impl_->adaptor = std::make_unique<PointCloudAdaptor>(data);
    impl_->tree = std::make_unique<KDTree>(
        static_cast<int>(data.cols()),
        *impl_->adaptor,
        nanoflann::KDTreeSingleIndexAdaptorParams(10));

    // Label convention: -2 = unclassified, -1 = noise, >=0 = cluster id.
    std::vector<int>  labels(n, -2);
    std::vector<bool> visited(n, false);
    int cluster_id = -1;

    for (int i = 0; i < n; ++i) {
        if (visited[i]) continue;
        visited[i] = true;

        std::vector<int> neighbors = region_query(data, i);
        if (static_cast<int>(neighbors.size()) < min_samples_) {
            labels[i] = -1;
            continue;
        }

        cluster_id += 1;
        labels[i] = cluster_id;

        std::vector<int> seeds = neighbors;
        seeds.erase(std::remove(seeds.begin(), seeds.end(), i), seeds.end());

        for (std::size_t k = 0; k < seeds.size(); ++k) {
            const int p = seeds[k];

            if (!visited[p]) {
                visited[p] = true;
                std::vector<int> nbrs = region_query(data, p);
                if (static_cast<int>(nbrs.size()) >= min_samples_) {
                    for (int nb : nbrs) {
                        if (std::find(seeds.begin(), seeds.end(), nb) ==
                            seeds.end()) {
                            seeds.push_back(nb);
                        }
                    }
                }
            }

            if (labels[p] == -1 || labels[p] == -2) {
                labels[p] = cluster_id;
            }
        }
    }

    for (auto& l : labels) {
        if (l == -2) l = -1;
    }
    return labels;
}

// ===========================================================================
// EpsGraphClusterer
// ===========================================================================

EpsGraphClusterer::EpsGraphClusterer(double eps) : eps_(eps) {}

std::vector<int> EpsGraphClusterer::fit_predict(const Dataset& data) const {
    const int n = static_cast<int>(data.rows());
    if (n == 0) return {};
    if (n == 1) return {0};

    // Disjoint-set union over the points.
    std::vector<int> parent(n);
    std::iota(parent.begin(), parent.end(), 0);
    std::vector<int> rank(n, 0);

    std::function<int(int)> find = [&](int x) -> int {
        while (parent[x] != x) {
            parent[x] = parent[parent[x]];
            x = parent[x];
        }
        return x;
    };
    auto unite = [&](int a, int b) {
        a = find(a);
        b = find(b);
        if (a == b) return;
        if (rank[a] < rank[b]) std::swap(a, b);
        parent[b] = a;
        if (rank[a] == rank[b]) ++rank[a];
    };

    // KD-tree over the input points.
    PointCloudAdaptor adaptor(data);
    KDTree tree(static_cast<int>(data.cols()), adaptor,
                nanoflann::KDTreeSingleIndexAdaptorParams(10));

    const double r2 = eps_ * eps_;

    std::vector<ResultItem> matches;
    nanoflann::SearchParameters params;
    Eigen::VectorXd query(data.cols());

    for (int i = 0; i < n; ++i) {
        query = data.row(i);
        matches.clear();
        tree.radiusSearch(query.data(), r2, matches, params);
        for (const auto& m : matches) {
            const int j = static_cast<int>(m.first);
            // Only unite with larger indices to halve the number of unions.
            if (j > i) unite(i, j);
        }
    }

    // Map each root to a compact label 0, 1, 2, ...
    std::unordered_map<int, int> root_to_label;
    root_to_label.reserve(static_cast<std::size_t>(n));

    std::vector<int> labels(n);
    int next = 0;
    for (int i = 0; i < n; ++i) {
        const int r = find(i);
        auto it = root_to_label.find(r);
        if (it == root_to_label.end()) {
            root_to_label.emplace(r, next);
            labels[i] = next;
            ++next;
        } else {
            labels[i] = it->second;
        }
    }
    return labels;
}

} // namespace mapper