#include "mapper/mapper.hpp"

#include <algorithm>
#include <unordered_map>

namespace mapper {

Mapper::Mapper(MapperParams params, FilterFunc filter)
    : params_(std::move(params)), filter_(std::move(filter)) {}

MapperGraph Mapper::fit_transform(const Dataset& data) const {
    MapperGraph graph;
    if (data.rows() == 0) {
        return graph;
    }

    // Step 1: evaluate the filter on the whole dataset.
    Eigen::VectorXd f = filter_(data);
    const double f_min = f.minCoeff();
    const double f_max = f.maxCoeff();

    // Step 2: build a one-dimensional uniform cover over the filter range.
    OneDimensionalCover cover(f_min, f_max, params_.cover_num, params_.overlap);

    // Step 3: resolve the clusterer once, outside the per-interval loop.
    ClustererPtr clusterer = params_.clusterer;
    if (!clusterer) {
        switch (params_.method) {
            case ClusterMethod::DBSCAN:
                clusterer = std::make_shared<DBSCAN>(params_.eps,
                                                     params_.min_samples);
                break;
            case ClusterMethod::EpsGraph:
            default:
                clusterer = std::make_shared<EpsGraphClusterer>(params_.eps);
                break;
        }
    }

    const int n = static_cast<int>(data.rows());

    for (int i = 0; i < cover.size(); ++i) {
        // Collect all point indices whose filter value falls into interval i.
        std::vector<int> idx;
        idx.reserve(static_cast<std::size_t>(n));
        for (int j = 0; j < n; ++j) {
            if (cover.contains(i, f[j])) {
                idx.push_back(j);
            }
        }
        if (idx.empty()) {
            continue;
        }

        // Build the sub-dataset corresponding to this interval.
        Dataset sub(static_cast<Eigen::Index>(idx.size()), data.cols());
        for (std::size_t k = 0; k < idx.size(); ++k) {
            sub.row(static_cast<Eigen::Index>(k)) = data.row(idx[k]);
        }

        // Run the selected clusterer inside this interval.
        std::vector<int> labels = clusterer->fit_predict(sub);

        // Group point indices by local cluster id; discard noise (label < 0).
        std::unordered_map<int, std::vector<int>> groups;
        for (std::size_t k = 0; k < labels.size(); ++k) {
            const int lab = labels[k];
            if (lab < 0) {
                continue;
            }
            groups[lab].push_back(idx[k]);
        }

        // Create one graph node per cluster.
        for (auto& kv : groups) {
            Node node;
            node.elements    = std::move(kv.second);
            node.interval_id = i;
            node.cluster_id  = kv.first;
            graph.add_node(std::move(node));
        }
    }

    // Step 4: connect nodes that share at least min_intersection points.
    graph.build_edges(params_.min_intersection);

    // Step 5: optional contraction of strict-subset nodes.
    if (params_.contract_nodes) {
        graph.contract_nodes();
    }

    return graph;
}

} // namespace mapper