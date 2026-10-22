#include "mapper/graph.hpp"

#include <algorithm>
#include <cmath>
#include <functional>
#include <iomanip>
#include <numeric>
#include <sstream>
#include <unordered_map>
#include <unordered_set>

namespace mapper {

void MapperGraph::add_node(Node n) {
    nodes_.push_back(std::move(n));
}

void MapperGraph::add_edge(NodeId a, NodeId b) {
    if (a > b) {
        std::swap(a, b);
    }
    edges_.emplace_back(a, b);
}

void MapperGraph::build_edges(int min_intersection) {
    edges_.clear();
    const std::size_t n = nodes_.size();

    for (std::size_t i = 0; i < n; ++i) {
        std::unordered_set<int> si(nodes_[i].elements.begin(),
                                   nodes_[i].elements.end());
        for (std::size_t j = i + 1; j < n; ++j) {
            int inter = 0;
            for (int e : nodes_[j].elements) {
                if (si.count(e)) {
                    ++inter;
                    if (inter >= min_intersection) {
                        break;
                }
            }
        }
            if (inter >= min_intersection) {
                add_edge(i, j);
            }
        }
    }
}

void MapperGraph::contract_nodes() {
    const std::size_t n = nodes_.size();

    // Convert each node's element list into a hash set for fast subset tests.
    std::vector<std::unordered_set<int>> sets(n);
    for (std::size_t i = 0; i < n; ++i) {
        sets[i] = std::unordered_set<int>(nodes_[i].elements.begin(),
                                          nodes_[i].elements.end());
    }

    // Mark nodes whose element set is a strict subset of another node's set.
    std::vector<bool> remove(n, false);
    for (std::size_t u = 0; u < n; ++u) {
        for (std::size_t v = 0; v < n; ++v) {
            if (u == v) {
                continue;
            }
            if (sets[u].size() >= sets[v].size()) {
                continue;
            }

            bool subset = true;
            for (int e : sets[u]) {
                if (!sets[v].count(e)) {
                    subset = false;
                    break;
                }
            }
            if (subset) {
                remove[u] = true;
                break;
            }
        }
    }

    // Rebuild the node list, remembering the index mapping.
    std::vector<int> old2new(n, -1);
    std::vector<Node> new_nodes;
    new_nodes.reserve(n);
    for (std::size_t i = 0; i < n; ++i) {
        if (!remove[i]) {
            old2new[i] = static_cast<int>(new_nodes.size());
            new_nodes.push_back(std::move(nodes_[i]));
        }
    }

    // Rebuild the edge list and remove duplicates.
    std::vector<std::pair<NodeId, NodeId>> new_edges;
    for (const auto& e : edges_) {
        if (remove[e.first] || remove[e.second]) {
            continue;
        }
        NodeId a = old2new[e.first];
        NodeId b = old2new[e.second];
        if (a > b) {
            std::swap(a, b);
        }
        new_edges.emplace_back(a, b);
    }
    std::sort(new_edges.begin(), new_edges.end());
    new_edges.erase(std::unique(new_edges.begin(), new_edges.end()),
                    new_edges.end());

    nodes_ = std::move(new_nodes);
    edges_ = std::move(new_edges);
}

// ---------------------------------------------------------------------------
// merge_connected_nodes
// ---------------------------------------------------------------------------
MapperGraph MapperGraph::merge_connected_nodes(
    const std::vector<NodeId>& subset) const {
    const std::size_t n = nodes_.size();
    if (subset.empty() || n == 0) {
        return *this;
    }

    // Mark which nodes belong to the subset. 
    std::vector<bool> in_subset(n, false);
    for (NodeId idx : subset) {
        if (idx < n) in_subset[idx] = true;
    }

    // Disjoint-set union.
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

    // Union over the INDUCED subgraph on `subset` only. 
    for (const auto& e : edges_) {
        if (e.first >= n || e.second >= n) continue;
        if (!in_subset[e.first] || !in_subset[e.second]) continue;
        unite(static_cast<int>(e.first), static_cast<int>(e.second));
    }

    // Group subset nodes by their component root.
    std::unordered_map<int, std::vector<NodeId>> groups;
    groups.reserve(subset.size());
    for (NodeId idx : subset) {
        if (idx >= n) continue;
        groups[find(static_cast<int>(idx))].push_back(idx);
    }

    // Merge groups with two or more members.
    std::vector<bool> removed(n, false);
    std::vector<Node> merged_nodes;

    for (const auto& kv : groups) {
        const auto& indices = kv.second;
        if (indices.size() < 2) continue;

        Node merged;
        std::unordered_set<int> seen;
        for (NodeId idx : indices) {
            removed[idx] = true;
            for (int e : nodes_[idx].elements) {
                if (seen.insert(e).second) {
                    merged.elements.push_back(e);
                }
            }
        }
        std::sort(merged.elements.begin(), merged.elements.end());
        merged_nodes.push_back(std::move(merged));
    }

    if (merged_nodes.empty()) {
        return *this;
    }

    // Build a fresh graph: surviving nodes plus merged replacements.
    MapperGraph out;
    for (std::size_t i = 0; i < n; ++i) {
        if (!removed[i]) {
            out.add_node(nodes_[i]);
        }
    }
    for (auto& node : merged_nodes) {
        out.add_node(std::move(node));
    }
    out.build_edges(1);
    return out;
}

// ---------------------------------------------------------------------------
// merge_neighbor_high_nodes
// ---------------------------------------------------------------------------
void MapperGraph::merge_neighbor_high_nodes(int min_degree) {
    const std::size_t n = nodes_.size();
    if (n == 0) return;

    // Build the undirected adjacency list.
    std::vector<std::vector<NodeId>> adjacency(n);
    for (const auto& e : edges_) {
        if (e.first >= n || e.second >= n) continue;
        adjacency[e.first].push_back(e.second);
        adjacency[e.second].push_back(e.first);
    }

    // Identify the high-degree nodes.
    const std::size_t degree_threshold =
        static_cast<std::size_t>(std::max(0, min_degree));
    std::vector<bool> is_high(n, false);
    for (std::size_t i = 0; i < n; ++i) {
        if (adjacency[i].size() >= degree_threshold) {
            is_high[i] = true;
        }
    }

    // Queue every high-degree node that has at least one high-degree
    // neighbour. This matches the reference semantics: v is queued as soon
    // as one of its neighbours is also high-degree.
    std::vector<NodeId> candidates;
    for (std::size_t i = 0; i < n; ++i) {
        if (!is_high[i]) continue;
        for (NodeId nb : adjacency[i]) {
            if (is_high[nb]) {
                candidates.push_back(static_cast<NodeId>(i));
                break;
            }
        }
    }

    if (!candidates.empty()) {
        MapperGraph merged = merge_connected_nodes(candidates);
        *this = std::move(merged);
    }

    // Mirror the reference's final step: rebuild edges from element
    // intersections. Idempotent when nothing was merged.
    build_edges(1);
}

std::vector<MapperGraph::NodePos2D>
MapperGraph::compute_node_positions(const Eigen::MatrixXd& embedding) const {
    std::vector<NodePos2D> out;
    out.reserve(nodes_.size());

    for (std::size_t i = 0; i < nodes_.size(); ++i) {
        const Node& n = nodes_[i];

        double sx = 0.0;
        double sy = 0.0;
        if (!n.elements.empty()) {
            for (int e : n.elements) {
                sx += embedding(e, 0);
                sy += embedding(e, 1);
            }
            const double k = static_cast<double>(n.elements.size());
            sx /= k;
            sy /= k;
        }

        out.push_back(NodePos2D{
            static_cast<NodeId>(i),
            sx,
            sy,
            static_cast<int>(n.elements.size())
        });
    }
    return out;
}

void MapperGraph::write_json(std::ostream& os) const {
    os << "{\n  \"nodes\": [\n";
    for (std::size_t i = 0; i < nodes_.size(); ++i) {
        os << "    {\"id\":"       << i
           << ",\"size\":"         << nodes_[i].elements.size()
           << ",\"interval\":"     << nodes_[i].interval_id
           << ",\"cluster\":"      << nodes_[i].cluster_id
           << ",\"elements\":[";
        for (std::size_t k = 0; k < nodes_[i].elements.size(); ++k) {
            if (k) {
                os << ",";
            }
            os << nodes_[i].elements[k];
        }
        os << "]}";
        if (i + 1 < nodes_.size()) {
            os << ",";
        }
        os << "\n";
    }
    os << "  ],\n  \"edges\": [\n";
    for (std::size_t i = 0; i < edges_.size(); ++i) {
        os << "    [" << edges_[i].first << "," << edges_[i].second << "]";
        if (i + 1 < edges_.size()) {
            os << ",";
        }
        os << "\n";
    }
    os << "  ]\n}\n";
}

void MapperGraph::write_dot(std::ostream& os) const {
    os << "graph Mapper {\n";
    for (std::size_t i = 0; i < nodes_.size(); ++i) {
        os << "  n" << i
           << " [label=\"" << nodes_[i].elements.size() << "\"];\n";
    }
    for (const auto& e : edges_) {
        os << "  n" << e.first << " -- " << e.second << ";\n";
    }
    os << "}\n";
}

void MapperGraph::write_html(std::ostream& os,
                             const Eigen::MatrixXd& embedding,
                             const std::string& title) const {
    const std::vector<NodePos2D> pos = compute_node_positions(embedding);

    os << "<!DOCTYPE html>\n<html>\n<head>\n"
       << "<meta charset=\"utf-8\">\n"
       << "<title>" << title << "</title>\n"
       << "<style>\n"
       << "html,body{margin:0;height:100%;background:#111;color:#eee;"
          "font-family:sans-serif;}\n"
       << "canvas{display:block;}\n"
       << "#info{position:fixed;top:8px;left:8px;padding:6px 10px;"
          "background:rgba(0,0,0,0.5);border-radius:4px;font-size:13px;}\n"
       << "</style>\n</head>\n<body>\n"
       << "<canvas id=\"c\"></canvas>\n"
       << "<div id=\"info\">nodes: " << nodes_.size()
       << " &nbsp; edges: " << edges_.size() << "</div>\n"
       << "<script>\n";

    os << "const nodes = [";
    os << std::setprecision(15);
    for (std::size_t i = 0; i < pos.size(); ++i) {
        if (i) {
            os << ",";
        }
        os << "{\"id\":"    << pos[i].id
           << ",\"x\":"     << pos[i].x
           << ",\"y\":"     << pos[i].y
           << ",\"size\":"  << pos[i].size
           << "}";
    }
    os << "];\n";

    os << "const edges = [";
    for (std::size_t i = 0; i < edges_.size(); ++i) {
        if (i) {
            os << ",";
        }
        os << "[" << edges_[i].first << "," << edges_[i].second << "]";
    }
    os << "];\n";

    // Renderer.
    os << R"JS(
const canvas = document.getElementById('c');
const ctx = canvas.getContext('2d');

function resize() {
    canvas.width  = window.innerWidth;
    canvas.height = window.innerHeight;
    draw();
}

function draw() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    if (nodes.length === 0) return;

    let minx = Infinity, maxx = -Infinity;
    let miny = Infinity, maxy = -Infinity;
    for (const n of nodes) {
        if (n.x < minx) minx = n.x;
        if (n.x > maxx) maxx = n.x;
        if (n.y < miny) miny = n.y;
        if (n.y > maxy) maxy = n.y;
    }

    const pad = 60;
    const W = canvas.width  - 2 * pad;
    const H = canvas.height - 2 * pad;
    const spanX = (maxx - minx) || 1;
    const spanY = (maxy - miny) || 1;
    const scale = Math.min(W / spanX, H / spanY);
    const ox = (canvas.width  - spanX * scale) / 2 - minx * scale;
    const oy = (canvas.height - spanY * scale) / 2 - miny * scale;

    const tx = x => x * scale + ox;
    const ty = y => canvas.height - (y * scale + oy); // flip Y

    // Edges first (under the nodes).
    ctx.strokeStyle = '#666';
    ctx.lineWidth = 1;
    for (const [a, b] of edges) {
        ctx.beginPath();
        ctx.moveTo(tx(nodes[a].x), ty(nodes[a].y));
        ctx.lineTo(tx(nodes[b].x), ty(nodes[b].y));
        ctx.stroke();
    }

    // Node radius scales with sqrt(size) so area roughly tracks point count.
    let maxSize = 1;
    for (const n of nodes) if (n.size > maxSize) maxSize = n.size;

    for (const n of nodes) {
        const r = 4 + 16 * Math.sqrt(n.size / maxSize);
        ctx.beginPath();
        ctx.arc(tx(n.x), ty(n.y), r, 0, 2 * Math.PI);
        ctx.fillStyle   = 'rgba(80,180,255,0.75)';
        ctx.fill();
        ctx.lineWidth   = 1;
        ctx.strokeStyle = '#fff';
        ctx.stroke();
    }
}

window.addEventListener('resize', resize);
resize();
)JS"
       << "</script>\n</body>\n</html>\n";
}

void write_points_html(std::ostream& os,
                       const Eigen::MatrixXd& points,
                       const std::string& title,
                       double point_alpha) {
    const Eigen::Index n = points.rows();
    if (n == 0) {
        os << "<!DOCTYPE html><html><body>(empty)</body></html>\n";
        return;
    }

    // 1. Shared isotropic normalization to [0,1]^2.
    const double minx = points.col(0).minCoeff();
    const double maxx = points.col(0).maxCoeff();
    double miny = 0.0, maxy = 0.0;
    if (points.cols() >= 2) {
        miny = points.col(1).minCoeff();
        maxy = points.col(1).maxCoeff();
    }
    const double cx = 0.5 * (minx + maxx);
    const double cy = 0.5 * (miny + maxy);
    double scale = std::max(maxx - minx, maxy - miny);
    if (scale <= 1e-12) scale = 1.0;

    auto norm = [&](double x, double y) -> std::pair<double, double> {
        return { (x - cx) / scale + 0.5, (y - cy) / scale + 0.5 };
    };

    // 2. Flatten the normalized coordinates into [x0, y0, x1, y1, ...].
    std::vector<double> flat;
    flat.reserve(static_cast<std::size_t>(n) * 2);
    for (Eigen::Index i = 0; i < n; ++i) {
        const double x = points(i, 0);
        const double y = (points.cols() >= 2) ? points(i, 1) : 0.0;
        const auto p = norm(x, y);
        flat.push_back(p.first);
        flat.push_back(p.second);
    }

    // 3. Choose opacity from the point count unless supplied.
    double alpha = point_alpha;
    if (alpha <= 0.0) {
        const double np = static_cast<double>(std::max<Eigen::Index>(1, n));
        alpha = 1.0 / (std::log10(np + 1.0) * 0.5);
        alpha = std::clamp(alpha, 0.2, 1.0);
    }

    // 4. Emit the HTML.
    os << "<!DOCTYPE html>\n<html>\n<head>\n"
       << "<meta charset=\"utf-8\">\n"
       << "<title>" << title << "</title>\n"
       << "<style>\n"
       << "html,body{margin:0;height:100%;background:#fff;}\n"
       << "canvas{display:block;}\n"
       << "#info{position:fixed;top:8px;left:8px;padding:6px 10px;"
          "background:rgba(0,0,0,0.6);color:#fff;border-radius:4px;"
          "font-size:13px;font-family:sans-serif;}\n"
       << "</style>\n</head>\n<body>\n"
       << "<canvas id=\"c\"></canvas>\n"
       << "<div id=\"info\">points: " << n << "</div>\n"
       << "<script>\n";

    os << "const rawPoints = [";
    os << std::setprecision(15);
    for (std::size_t i = 0; i < flat.size(); ++i) {
        if (i) os << ",";
        os << flat[i];
    }
    os << "];\n";
    os << "const pointAlpha = " << alpha << ";\n";

    os << R"JS(
const canvas = document.getElementById('c');
const ctx = canvas.getContext('2d');

function draw() {
    const W = canvas.width;
    const H = canvas.height;
    ctx.clearRect(0, 0, W, H);

    // Square viewport, centered, with a small margin.
    const pad  = 40;
    const side = Math.min(W, H) - 2 * pad;
    const ox   = (W - side) / 2;
    const oy   = (H - side) / 2;
    const tx   = x => ox + x * side;
    const ty   = y => oy + (1.0 - y) * side;   // flip y: origin at bottom-left

    // 1. Data points.
    ctx.globalAlpha = pointAlpha;
    ctx.fillStyle   = '#1a7f3c';
    const r = 1.3;
    for (let i = 0; i + 1 < rawPoints.length; i += 2) {
        ctx.beginPath();
        ctx.arc(tx(rawPoints[i]), ty(rawPoints[i + 1]), r, 0, 2 * Math.PI);
        ctx.fill();
    }
    ctx.globalAlpha = 1.0;

    // 2. Border of the unit square.
    ctx.strokeStyle = '#000';
    ctx.lineWidth   = 1.2;
    ctx.strokeRect(tx(0), ty(1), side, side);
}

function resize() {
    canvas.width  = window.innerWidth;
    canvas.height = window.innerHeight;
    draw();
}

window.addEventListener('resize', resize);
resize();
)JS"
       << "</script>\n</body>\n</html>\n";
}

void MapperGraph::write_html_with_values(std::ostream& os,
                                         const Eigen::MatrixXd& embedding,
                                         const Eigen::VectorXd& point_values,
                                         const std::string& title,
                                         const std::string& colorbar_label) const {
    if (embedding.rows() != point_values.size()) {
        os << "<!-- ERROR: embedding.rows() != point_values.size() -->\n";
        return;
    }

    // 2D centroids of each node's elements (same layout as write_html).
    const std::vector<NodePos2D> pos = compute_node_positions(embedding);

    // Per-node average function value.
    std::vector<double> node_values(nodes_.size(), 0.0);
    for (std::size_t i = 0; i < nodes_.size(); ++i) {
        const auto& elems = nodes_[i].elements;
        if (elems.empty()) continue;
        double s = 0.0;
        for (int e : elems) s += point_values[e];
        node_values[i] = s / static_cast<double>(elems.size());
    }

    // Value range across nodes.
    double vmin = 0.0, vmax = 0.0;
    bool first = true;
    for (double v : node_values) {
        if (first) { vmin = vmax = v; first = false; }
        else {
            if (v < vmin) vmin = v;
            if (v > vmax) vmax = v;
        }
    }

    os << "<!DOCTYPE html>\n<html>\n<head>\n"
       << "<meta charset=\"utf-8\">\n"
       << "<title>" << title << "</title>\n"
       << "<style>\n"
       << "html,body{margin:0;height:100%;background:#111;color:#eee;"
          "font-family:sans-serif;}\n"
       << "canvas{display:block;}\n"
       << "#info{position:fixed;top:8px;left:8px;padding:6px 10px;"
          "background:rgba(0,0,0,0.5);border-radius:4px;font-size:13px;}\n"
       << "</style>\n</head>\n<body>\n"
       << "<canvas id=\"c\"></canvas>\n"
       << "<div id=\"info\">nodes: " << nodes_.size()
       << " &nbsp; edges: " << edges_.size()
       << " &nbsp; value range: ["
       << vmin << ", " << vmax << "]</div>\n"
       << "<script>\n";

    // Nodes (positions + sizes).
    os << "const nodes = [";
    os << std::setprecision(15);
    for (std::size_t i = 0; i < pos.size(); ++i) {
        if (i) os << ",";
        os << "{\"id\":"    << pos[i].id
           << ",\"x\":"     << pos[i].x
           << ",\"y\":"     << pos[i].y
           << ",\"size\":"  << pos[i].size
           << "}";
    }
    os << "];\n";

    // Per-node average value (drives the colour).
    os << "const nodeValues = [";
    for (std::size_t i = 0; i < node_values.size(); ++i) {
        if (i) os << ",";
        os << node_values[i];
    }
    os << "];\n";

    // Edges.
    os << "const edges = [";
    for (std::size_t i = 0; i < edges_.size(); ++i) {
        if (i) os << ",";
        os << "[" << edges_[i].first << "," << edges_[i].second << "]";
    }
    os << "];\n";

    // Scale / label for the colourbar.
    os << "const vmin = " << vmin << ";\n";
    os << "const vmax = " << vmax << ";\n";
    os << "const colorbarLabel = \"" << colorbar_label << "\";\n";

    os << R"JS(
const canvas = document.getElementById('c');
const ctx = canvas.getContext('2d');

// Matplotlib-style "jet" colormap, approximated by piecewise linear
// interpolation of six anchor points.
function jet(t) {
    t = Math.max(0.0, Math.min(1.0, t));
    const pts = [
        [0.000, 0.000, 0.000, 0.500],
        [0.125, 0.000, 0.000, 1.000],
        [0.375, 0.000, 1.000, 1.000],
        [0.625, 1.000, 1.000, 0.000],
        [0.875, 1.000, 0.000, 0.000],
        [1.000, 0.500, 0.000, 0.000],
    ];
    for (let i = 0; i < pts.length - 1; ++i) {
        if (t >= pts[i][0] && t <= pts[i + 1][0]) {
            const u = (t - pts[i][0]) / (pts[i + 1][0] - pts[i][0]);
            return [
                pts[i][1] + u * (pts[i + 1][1] - pts[i][1]),
                pts[i][2] + u * (pts[i + 1][2] - pts[i][2]),
                pts[i][3] + u * (pts[i + 1][3] - pts[i][3]),
            ];
        }
    }
    return [0, 0, 0];
}

function valueToColor(v) {
    const span = (vmax - vmin) > 1e-30 ? (vmax - vmin) : 1.0;
    const c = jet((v - vmin) / span);
    return 'rgb(' + Math.round(c[0] * 255) + ','
                  + Math.round(c[1] * 255) + ','
                  + Math.round(c[2] * 255) + ')';
}

function draw() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    if (nodes.length === 0) return;

    let minx = Infinity, maxx = -Infinity;
    let miny = Infinity, maxy = -Infinity;
    for (const n of nodes) {
        if (n.x < minx) minx = n.x;
        if (n.x > maxx) maxx = n.x;
        if (n.y < miny) miny = n.y;
        if (n.y > maxy) maxy = n.y;
    }

    const pad = 60;
    const W = canvas.width  - 2 * pad;
    const H = canvas.height - 2 * pad;
    const spanX = (maxx - minx) || 1;
    const spanY = (maxy - miny) || 1;
    const scale = Math.min(W / spanX, H / spanY);
    const ox = (canvas.width  - spanX * scale) / 2 - minx * scale;
    const oy = (canvas.height - spanY * scale) / 2 - miny * scale;

    const tx = x => x * scale + ox;
    const ty = y => canvas.height - (y * scale + oy);  // flip Y

    // Edges first (under the nodes).
    ctx.strokeStyle = '#666';
    ctx.lineWidth = 1;
    for (const [a, b] of edges) {
        ctx.beginPath();
        ctx.moveTo(tx(nodes[a].x), ty(nodes[a].y));
        ctx.lineTo(tx(nodes[b].x), ty(nodes[b].y));
        ctx.stroke();
    }

    // Nodes, coloured by their average value.
    let maxSize = 1;
    for (const n of nodes) if (n.size > maxSize) maxSize = n.size;

    for (let i = 0; i < nodes.length; ++i) {
        const n = nodes[i];
        const r = 4 + 16 * Math.sqrt(n.size / maxSize);
        ctx.beginPath();
        ctx.arc(tx(n.x), ty(n.y), r, 0, 2 * Math.PI);
        ctx.fillStyle = valueToColor(nodeValues[i]);
        ctx.fill();
        ctx.lineWidth = 1;
        ctx.strokeStyle = '#fff';
        ctx.stroke();
    }

    drawColorbar();
}

function drawColorbar() {
    const cw = 22;
    const ch = 300;
    const cx = canvas.width - 90;
    const cy = (canvas.height - ch) / 2;

    // Vertical gradient from vmax (top) down to vmin (bottom).
    for (let y = 0; y < ch; ++y) {
        const t = 1.0 - y / (ch - 1);
        const c = jet(t);
        ctx.fillStyle = 'rgb(' + Math.round(c[0] * 255) + ','
                              + Math.round(c[1] * 255) + ','
                              + Math.round(c[2] * 255) + ')';
        ctx.fillRect(cx, cy + y, cw, 1);
    }

    // Border.
    ctx.strokeStyle = '#fff';
    ctx.lineWidth = 1;
    ctx.strokeRect(cx, cy, cw, ch);

    // Tick labels.
    ctx.fillStyle = '#fff';
    ctx.font = '12px sans-serif';
    ctx.textAlign = 'left';
    ctx.fillText(vmax.toFixed(4), cx + cw + 6, cy + 10);
    ctx.fillText(vmin.toFixed(4), cx + cw + 6, cy + ch);

    // Vertical label to the left of the colourbar.
    ctx.save();
    ctx.translate(cx - 10, cy + ch / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.textAlign = 'center';
    ctx.fillText(colorbarLabel, 0, 0);
    ctx.restore();
}

function resize() {
    canvas.width  = window.innerWidth;
    canvas.height = window.innerHeight;
    draw();
}

window.addEventListener('resize', resize);
resize();
)JS"
       << "</script>\n</body>\n</html>\n";
}


} // namespace mapper