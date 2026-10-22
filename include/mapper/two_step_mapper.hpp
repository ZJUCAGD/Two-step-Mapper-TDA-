#pragma once

#include "mapper/cluster.hpp"
#include "mapper/graph.hpp"
#include "mapper/mapper.hpp"
#include "mapper/types.hpp"

#include <Eigen/Dense>

#include <cstddef>
#include <vector>

namespace mapper {

// Parameters controlling the two-step Mapper pipeline.
struct TwoStepMapperParams {
    double overlap         = 0.2;   // Cover overlap fraction.
    double eps             = 1.0;   // DBSCAN neighbourhood radius.
    int    min_samples     = 4;
    int    split_threshold = 2;     // S_i >= split_threshold triggers a split.

    // Which clustering algorithm to use. Null means: follow
    // ClusterMethod::EpsGraph using the eps field.
    ClusterMethod method    = ClusterMethod::EpsGraph;
    ClustererPtr  clusterer = nullptr;
};

// Result of the two-step Mapper, exposing both stages for inspection.
struct TwoStepMapperResult {
    MapperGraph first_stage;    // Initial Mapper graph on the main direction.
    MapperGraph final_graph;    // Graph after the orthogonal split step.
};

// Returns the first principal direction of a 2D / 3D point set.
Eigen::VectorXd first_principal_direction(const Dataset& data);

// Returns a unit vector orthogonal to w.
Eigen::VectorXd orthogonal_direction(const Eigen::VectorXd& w);

// Two-step Mapper algorithm.
class TwoStepMapper {
public:
    explicit TwoStepMapper(TwoStepMapperParams params);

    TwoStepMapperResult fit_transform(const Dataset& data) const;

private:
    TwoStepMapperParams params_;
};

// ---------------------------------------------------------------------------
// Maximum local radius derived from strip geometry.
//
// The routine estimates a global scale r_max that characterises the largest
// "bend" present in a Mapper graph, using the geometry of node strips:
//
//   * For every node v with degree <= 2, a strip width is the maximum
//     distance from v's points to the local tangent line through the shared
//     centroids of its incident edges (or, for degree 1, through the node
//     centroid itself and the shared centroid with its only neighbour).
//
//   * For every singular node v with degree > 2, the smallest angle alpha
//     between adjacent strips is used to turn the neighbouring strip widths
//     into a local radius via
//
//         r = 0.5 * sqrt( eps^2 + ( d / sin(alpha/2)
//                                   + eps / tan(alpha/2) )^2 ),
//
//     where d is twice the width of the widest of the two adjacent strips.
//     Angles below pi/10 are rejected for stability.
//
// The maximum over all such radii is returned as r_max.
// ---------------------------------------------------------------------------

struct StripRadiusResult {
    // Per-node local strip radius. Zero when no radius could be derived.
    std::vector<double> radii;
    // Per-node local tangent direction. Empty for singular nodes.
    std::vector<Eigen::VectorXd> directions;
    // Maximum radius over all nodes; 0.0 when no node yields a value.
    double r_max = 0.0;
};

// Computes the maximum local radius of `graph` given the original `data`.
double compute_graph_radius_by_strips(const MapperGraph& graph,
                                      const Dataset&     data,
                                      double             eps,
                                      StripRadiusResult* out = nullptr,
                                      bool               singular_only = false);

// ---------------------------------------------------------------------------
// Final cover overlap derived from the maximum local radius.
// ---------------------------------------------------------------------------
double compute_overlap(const Dataset&         data,
                       const Eigen::VectorXd& func_val,
                       double                 strip_width,
                       double                 eps);

// ---------------------------------------------------------------------------
// Automatic cover overlap estimated from the maximum local radius.
// ---------------------------------------------------------------------------
double estimate_overlap_from_r_max(const Dataset& data,
                                   double         eps,
                                   int            min_samples,
                                   int            split_threshold,
                                   double         fallback_overlap,
                                   double*        out_r_max = nullptr);

} // namespace mapper