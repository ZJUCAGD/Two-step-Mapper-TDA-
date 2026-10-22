#pragma once

#include <Eigen/Dense>

#include <cstddef>
#include <ostream>
#include <string>
#include <utility>
#include <vector>

namespace mapper {

using NodeId = std::size_t;

// A node of the mapper graph represents a cluster found inside one cover
// interval. Its elements are indices into the original dataset.
struct Node {
    std::vector<int> elements;   // Original point indices contained in this cluster.
    int interval_id = -1;        // Index of the cover interval this node belongs to.
    int cluster_id  = -1;        // Local cluster id within that interval.
};

// An undirected graph describing the mapper output.
class MapperGraph {
public:
    // 2D position of a node, computed as the centroid of its elements in a
    // given 2D embedding (typically the first two features of the dataset).
    struct NodePos2D {
        NodeId id;
        double x;
        double y;
        int    size;   // number of original points inside the node
    };

    void add_node(Node n);
    void add_edge(NodeId a, NodeId b);

    // Connects two nodes when their element sets share at least
    // min_intersection common points.
    void build_edges(int min_intersection = 1);

    // Removes every node whose element set is a strict subset of another
    // node's element set.
    void contract_nodes();

    // ----------------------------------------------------------------------
    // Merges the nodes listed in `subset` that belong to the same connected
    // component of the current graph. Each merged group is replaced by a
    // single node whose elements are the union of the originals. Edges are
    // rebuilt from element intersections (min_intersection = 1).
    // ----------------------------------------------------------------------
    MapperGraph merge_connected_nodes(const std::vector<NodeId>& subset) const;

    // ----------------------------------------------------------------------
    // Merges pairs of adjacent high-degree nodes.
    //
    // For every node v with degree >= min_degree, if v has a neighbour u
    // with degree >= min_degree, then both v and u are queued. Every
    // connected group of queued nodes is replaced by a single node whose
    // elements are the union of the originals. Edges are then rebuilt from
    // element intersections (min_intersection = 1).
    // ----------------------------------------------------------------------
    void merge_neighbor_high_nodes(int min_degree = 3);

    std::size_t num_nodes() const { return nodes_.size(); }
    std::size_t num_edges() const { return edges_.size(); }

    const std::vector<Node>& nodes() const { return nodes_; }

    // Moves the node list out of the graph, leaving it empty.
    std::vector<Node> take_nodes() { return std::move(nodes_); }

    const std::vector<std::pair<NodeId, NodeId>>& edges() const {
        return edges_;
    }

    // Returns the 2D centroid of each node's elements, using the provided
    // embedding (an N x 2 matrix with one row per original sample).
    std::vector<NodePos2D>
    compute_node_positions(const Eigen::MatrixXd& embedding) const;

    // Emits a self-contained HTML page that renders the mapper graph.
    void write_html(std::ostream& os,
                    const Eigen::MatrixXd& embedding,
                    const std::string& title = "Mapper Graph") const;

    // Same layout as write_html, but node colours are driven by per-point
    // function values:
    //
    //   * the position of each node is the centroid of its elements in
    //     `embedding` (an N x 2 matrix), exactly like write_html;
    //   * the colour of each node is the average of `point_values` over the
    //     node's elements, mapped through the "jet" colormap;
    //   * a vertical colourbar on the right shows the value range and the
    //     supplied label.
    //
    // `point_values` must have one entry per row of `embedding`.
    void write_html_with_values(std::ostream& os,
                                const Eigen::MatrixXd& embedding,
                                const Eigen::VectorXd& point_values,
                                const std::string& title = "Mapper Graph",
                                const std::string& colorbar_label =
                                    "Function value") const;


    
    // Serialization helpers.
    void write_json(std::ostream& os) const;
    void write_dot(std::ostream& os) const;

private:
    std::vector<Node> nodes_;
    std::vector<std::pair<NodeId, NodeId>> edges_;
};

    // ---------------------------------------------------------------------------
// Standalone point-cloud renderer.
//
// Writes a self-contained HTML page that shows only `points`. The first two
// columns of `points` are used as (x, y). 
// ---------------------------------------------------------------------------
void write_points_html(std::ostream& os,
                    const Eigen::MatrixXd& points,
                    const std::string& title = "Point Cloud",
                    double point_alpha = -1.0);



} // namespace mapper