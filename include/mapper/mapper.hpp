#pragma once

#include "mapper/cluster.hpp"
#include "mapper/cover.hpp"
#include "mapper/filter.hpp"
#include "mapper/graph.hpp"
#include "mapper/types.hpp"

#include <memory>

namespace mapper {

// Which clustering algorithm to use when no explicit clusterer is supplied.
// The default is EpsGraph (connected components of the eps-ball graph).
enum class ClusterMethod {
    EpsGraph,
    DBSCAN,
};

struct MapperParams {
    int    cover_num        = 10;      // Number of cover intervals.
    double overlap          = 0.2;     // Overlap fraction between intervals.
    int    min_intersection = 1;       // Minimum shared points required for an edge.
    bool   contract_nodes   = false;   // If true, remove strict-subset nodes.

    // Which algorithm to use when `clusterer` is null.
    ClusterMethod method = ClusterMethod::EpsGraph;

    // Explicit clusterer. Takes precedence over `method` when non-null.
    ClustererPtr clusterer = nullptr;

    // Parameters used by the built-in clusterers when `clusterer` is null.
    double eps      = 1.0;   // Neighbourhood radius for both built-ins.
    int    min_samples = 4;  // used only by ClusterMethod::DBSCAN.
};

class Mapper {
public:
    Mapper(MapperParams params, FilterFunc filter);

    MapperGraph fit_transform(const Dataset& data) const;

private:
    MapperParams params_;
    FilterFunc   filter_;
};

} // namespace mapper