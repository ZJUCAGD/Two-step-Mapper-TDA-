#include "mapper/two_step_mapper.hpp"

#include "mapper/cover.hpp"

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <numeric>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace mapper {

// ---------------------------------------------------------------------------
// Direction helpers
// ---------------------------------------------------------------------------

Eigen::VectorXd first_principal_direction(const Dataset& data) {
    const Eigen::Index n = data.rows();
    const Eigen::Index d = data.cols();

    if (n < 2 || d < 1) {
        Eigen::VectorXd w = Eigen::VectorXd::Zero(std::max<Eigen::Index>(d, 1));
        w[0] = 1.0;
        return w;
    }

    const Eigen::RowVectorXd mean = data.colwise().mean();
    const Eigen::MatrixXd    Xc   = data.rowwise() - mean;
    const Eigen::MatrixXd    C    = (Xc.transpose() * Xc) /
                                    static_cast<double>(n - 1);

    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(C);
    Eigen::VectorXd w = es.eigenvectors().col(d - 1);

    // Fix the sign so the result is deterministic across runs.
    for (Eigen::Index i = 0; i < w.size(); ++i) {
        if (std::abs(w[i]) > 1e-12) {
            if (w[i] < 0.0) w = -w;
            break;
        }
    }
    return w;
}

Eigen::VectorXd orthogonal_direction(const Eigen::VectorXd& w) {
    const Eigen::Index d = w.size();
    Eigen::VectorXd    o = Eigen::VectorXd::Zero(d);

    if (d == 2) {
        o[0] = -w[1];
        o[1] =  w[0];
        return o;
    }
    if (d == 0) return o;

    Eigen::Index axis = 0;
    for (Eigen::Index i = 1; i < d; ++i) {
        if (std::abs(w[i]) < std::abs(w[axis])) axis = i;
    }
    o[axis] = 1.0;
    o -= (o.dot(w) / w.dot(w)) * w;
    const double nrm = o.norm();
    if (nrm > 1e-12) o /= nrm;
    return o;
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

namespace {

FilterFunc make_projection(const Eigen::VectorXd& direction) {
    return [direction](const Dataset& d) -> Eigen::VectorXd {
        return d * direction;
    };
}

// Number of orthogonal-cover intervals that a node's point subset spans.
int compute_split_num(const std::vector<int>& elements,
                      const Eigen::VectorXd&  fvals,
                      double                  cover_len,
                      double                  overlap) {
    if (elements.empty() || cover_len <= 0.0 || overlap >= 0.5) return 0;

    double y_min = fvals[elements[0]];
    double y_max = y_min;
    for (int e : elements) {
        const double v = fvals[e];
        if (v < y_min) y_min = v;
        if (v > y_max) y_max = v;
    }
    const double num = (y_max - y_min - cover_len * overlap) /
                       cover_len / (1.0 - overlap);
    return static_cast<int>(num);
}

// Nodes in `g` whose element set intersects `target`.
std::vector<NodeId> find_intersecting_nodes(
    const MapperGraph& g, const std::vector<int>& target) {
    std::vector<NodeId> out;
    if (target.empty()) return out;

    std::unordered_set<int> target_set(target.begin(), target.end());
    for (std::size_t i = 0; i < g.num_nodes(); ++i) {
        for (int e : g.nodes()[i].elements) {
            if (target_set.count(e)) {
                out.push_back(i);
                break;
            }
        }
    }
    return out;
}

// Merge a group of nodes into one (element union); rebuild edges.
MapperGraph merge_node_group(MapperGraph g,
                             const std::vector<NodeId>& group) {
    if (group.size() < 2) return g;

    std::unordered_set<NodeId> group_set(group.begin(), group.end());

    Node merged;
    std::unordered_set<int> seen;
    for (NodeId idx : group) {
        for (int e : g.nodes()[idx].elements) {
            if (seen.insert(e).second) merged.elements.push_back(e);
        }
    }
    std::sort(merged.elements.begin(), merged.elements.end());

    MapperGraph out;
    for (std::size_t i = 0; i < g.num_nodes(); ++i) {
        if (!group_set.count(i)) out.add_node(g.nodes()[i]);
    }
    out.add_node(std::move(merged));
    out.build_edges(1);
    return out;
}

} // namespace

// ---------------------------------------------------------------------------
// TwoStepMapper
// ---------------------------------------------------------------------------

TwoStepMapper::TwoStepMapper(TwoStepMapperParams params)
    : params_(std::move(params)) {}

TwoStepMapperResult TwoStepMapper::fit_transform(const Dataset& data) const {
    TwoStepMapperResult result;
    if (data.rows() == 0) return result;

    const int d = static_cast<int>(data.cols());

    // 1. Directions and filters.
    const Eigen::VectorXd w_main = first_principal_direction(data);
    const Eigen::VectorXd w_perp = orthogonal_direction(w_main);

    FilterFunc f_main = make_projection(w_main);
    FilterFunc f_perp = make_projection(w_perp);

    const Eigen::VectorXd fvals_main = f_main(data);
    const Eigen::VectorXd fvals_perp = f_perp(data);

    // 2. First-stage Mapper on the main direction.
    MapperParams mp1;
    mp1.cover_num        = compute_cover_num(data, fvals_main,
                                             params_.eps, params_.overlap);
    mp1.overlap          = params_.overlap;
    mp1.min_intersection = 1;
    mp1.contract_nodes   = true;
    // Propagate clustering choice.
    mp1.method           = params_.method;
    mp1.clusterer        = params_.clusterer;
    mp1.eps              = params_.eps;
    mp1.min_samples         = params_.min_samples;

    Mapper      mapper1(mp1, f_main);
    MapperGraph g = mapper1.fit_transform(data);
    result.first_stage = g;

    // 3. Fixed global orthogonal cover length used by every split test.
    const double cover_len_perp =
        compute_cover_len(data, fvals_perp, params_.eps, params_.overlap);
    if (cover_len_perp <= 0.0) {
        result.final_graph = std::move(g);
        return result;
    }

    // 4. Collect V_split on the current graph.
    auto collect_v_split = [&](const MapperGraph& gr) {
        std::vector<NodeId> v;
        for (std::size_t i = 0; i < gr.num_nodes(); ++i) {
            if (compute_split_num(gr.nodes()[i].elements, fvals_perp,
                                  cover_len_perp, params_.overlap) >=
                params_.split_threshold) {
                v.push_back(static_cast<NodeId>(i));
            }
        }
        return v;
    };

    const std::vector<NodeId> v_split0 = collect_v_split(g);

    // 5. Merge V_split nodes that lie in the same connected component.
    g = g.merge_connected_nodes(v_split0);

    // 6. Snapshot nodes and adjacency before the split loop mutates `g`.
    const std::size_t n_snapshot = g.num_nodes();

    std::vector<std::vector<int>> orig_elements;
    orig_elements.reserve(n_snapshot);
    for (const auto& node : g.nodes()) orig_elements.push_back(node.elements);

    std::vector<std::vector<NodeId>> neighbours(n_snapshot);
    {
        std::vector<std::unordered_set<NodeId>> adj(n_snapshot);
        for (const auto& e : g.edges()) {
            adj[e.first].insert(e.second);
            adj[e.second].insert(e.first);
        }
        for (std::size_t i = 0; i < n_snapshot; ++i) {
            neighbours[i].assign(adj[i].begin(), adj[i].end());
        }
    }

    // 7. Split loop.
    std::vector<bool> delete_orig(n_snapshot, false);
    std::vector<Node> appended_nodes;

    for (std::size_t i = 0; i < n_snapshot; ++i) {
        const int S_i = compute_split_num(orig_elements[i], fvals_perp,
                                          cover_len_perp, params_.overlap);
        if (S_i < params_.split_threshold) continue;
        
        delete_orig[i] = true;

        const auto& xi = orig_elements[i];
        const Eigen::Index m = static_cast<Eigen::Index>(xi.size());
        if (m == 0) continue;

        Dataset sub_data(m, d);
        Eigen::VectorXd sub_fperp(m);
        for (Eigen::Index k = 0; k < m; ++k) {
            sub_data.row(k) = data.row(xi[static_cast<std::size_t>(k)]);
            sub_fperp[k]    = fvals_perp[xi[static_cast<std::size_t>(k)]];
        }

        // Sub-Mapper built on the orthogonal direction.
        MapperParams mp2;
        mp2.cover_num        = std::max(1, S_i);
        mp2.overlap          = params_.overlap;
        mp2.min_intersection = 1;
        mp2.contract_nodes   = true;
        // Propagate clustering choice.
        mp2.method           = params_.method;
        mp2.clusterer        = params_.clusterer;
        mp2.eps              = params_.eps;
        mp2.min_samples         = params_.min_samples;

        Mapper      sub_mapper(mp2, f_perp);
        MapperGraph sub_g = sub_mapper.fit_transform(sub_data);

        // Remap local indices (positions in sub_data) to global ids.
        MapperGraph remapped;
        for (const auto& node : sub_g.nodes()) {
            Node nn;
            nn.interval_id = node.interval_id;
            nn.cluster_id  = node.cluster_id;
            nn.elements.reserve(node.elements.size());
            for (int local_idx : node.elements) {
                nn.elements.push_back(xi[static_cast<std::size_t>(local_idx)]);
            }
            std::sort(nn.elements.begin(), nn.elements.end());
            remapped.add_node(std::move(nn));
        }
        remapped.build_edges(1);

        // Merge sub-nodes that intersect each neighbouring point set X_j.
        for (NodeId j : neighbours[i]) {
            const auto& xj = orig_elements[j];
            if (xj.empty()) continue;

            const std::vector<NodeId> to_merge =
                find_intersecting_nodes(remapped, xj);
            if (to_merge.size() >= 2) {
                remapped = merge_node_group(std::move(remapped), to_merge);
            }
        }

        for (auto& node : remapped.take_nodes()) {
            appended_nodes.push_back(std::move(node));
        }
    }

    // 8. Assemble the final graph.
    MapperGraph final_g;
    for (std::size_t i = 0; i < n_snapshot; ++i) {
        if (!delete_orig[i]) final_g.add_node(g.nodes()[i]);
    }
    for (auto& node : appended_nodes) final_g.add_node(std::move(node));

    final_g.build_edges(1);

    // 9. Merge adjacent high-degree nodes.
    final_g.merge_neighbor_high_nodes(3);

    result.final_graph = std::move(final_g);

    return result;
}

// ---------------------------------------------------------------------------
// Maximum local radius
// ---------------------------------------------------------------------------

namespace {

constexpr double kPi = 3.14159265358979323846;

Eigen::VectorXd centroid_of_elements(const std::vector<int>& elements,
                                     const Dataset&          data) {
    if (elements.empty()) return Eigen::VectorXd::Zero(data.cols());
    Eigen::VectorXd c = Eigen::VectorXd::Zero(data.cols());
    for (int idx : elements) c += data.row(idx).transpose();
    c /= static_cast<double>(elements.size());
    return c;
}

Eigen::VectorXd shared_centroid(const std::vector<int>& a,
                                const std::vector<int>& b,
                                const Dataset&          data) {
    std::unordered_set<int> sa(a.begin(), a.end());
    std::vector<int> shared;
    shared.reserve(std::min(a.size(), b.size()));
    for (int e : b) {
        if (sa.count(e)) shared.push_back(e);
    }
    if (shared.empty()) return centroid_of_elements(a, data);
    return centroid_of_elements(shared, data);
}

double point_line_distance(const Eigen::VectorXd& p,
                           const Eigen::VectorXd& c1,
                           const Eigen::VectorXd& c2) {
    const Eigen::VectorXd line_vec  = c2 - c1;
    const double          line_norm = line_vec.norm();
    if (line_norm < 1e-12) return (p - c1).norm();

    if (p.size() == 2) {
        const double num = std::abs(line_vec[0] * (p[1] - c1[1]) -
                                    line_vec[1] * (p[0] - c1[0]));
        return num / line_norm;
    }

    const double          t    = (p - c1).dot(line_vec) / (line_norm * line_norm);
    const Eigen::VectorXd proj = c1 + t * line_vec;
    return (p - proj).norm();
}

} // namespace

double compute_graph_radius_by_strips(const MapperGraph& graph,
                                      const Dataset&     data,
                                      double             eps,
                                      StripRadiusResult* out,
                                      bool               singular_only) {
    const std::size_t n = graph.num_nodes();
    if (n == 0) {
        if (out) *out = StripRadiusResult{};
        return 0.0;
    }

    // Per-node centroid and per-edge shared centroid.
    std::vector<Eigen::VectorXd> midpoints(n);
    for (std::size_t i = 0; i < n; ++i) {
        midpoints[i] = centroid_of_elements(graph.nodes()[i].elements, data);
    }

    std::vector<std::vector<std::size_t>> adjacency(n);
    std::vector<std::unordered_map<std::size_t, Eigen::VectorXd>> shared_center(n);

    for (const auto& e : graph.edges()) {
        const std::size_t a = e.first;
        const std::size_t b = e.second;
        if (a >= n || b >= n) continue;
        adjacency[a].push_back(b);
        adjacency[b].push_back(a);

        Eigen::VectorXd c = shared_centroid(graph.nodes()[a].elements,
                                            graph.nodes()[b].elements,
                                            data);
        shared_center[a][b] = c;
        shared_center[b][a] = c;
    }

    std::vector<double>          radii(n, 0.0);
    std::vector<bool>            radius_valid(n, false);
    std::vector<Eigen::VectorXd> directions(n);

    // Strip width of a non-singular node (degree <= 2): the maximum distance
    // from its points to the local tangent line, plus that tangent direction.
    // `valid` is false for nodes without a well-defined strip (degree 0 or
    // degree > 2, missing shared centroids, or no elements).
    struct Strip {
        double          width = 0.0;
        bool            valid = false;
        Eigen::VectorXd direction;
    };
    auto strip_of = [&](std::size_t vid) -> Strip {
        const auto& nb_list = adjacency[vid];
        const std::size_t deg = nb_list.size();
        if (deg == 0 || deg > 2) return Strip{};

        Eigen::VectorXd c1, c2;
        if (deg == 2) {
            auto it1 = shared_center[vid].find(nb_list[0]);
            auto it2 = shared_center[vid].find(nb_list[1]);
            if (it1 == shared_center[vid].end() ||
                it2 == shared_center[vid].end()) return Strip{};
            c1 = it1->second;
            c2 = it2->second;
        } else { // deg == 1
            c1 = midpoints[vid];
            auto it = shared_center[vid].find(nb_list[0]);
            if (it == shared_center[vid].end()) return Strip{};
            c2 = it->second;
        }

        const auto& elements = graph.nodes()[vid].elements;
        if (elements.empty()) return Strip{};

        double max_dist = 0.0;
        for (int idx : elements) {
            const Eigen::VectorXd p = data.row(idx).transpose();
            const double d = point_line_distance(p, c1, c2);
            if (d > max_dist) max_dist = d;
        }
        return Strip{max_dist, true, c2 - c1};
    };

    // Non-singular nodes: strip width from the local tangent line.
    if (!singular_only) {
        for (std::size_t vid = 0; vid < n; ++vid) {
            const Strip s = strip_of(vid);
            if (!s.valid) continue;
            radii[vid]        = s.width;
            radius_valid[vid] = true;
            directions[vid]   = s.direction;
        }
    }

    // Singular nodes (degree > 2): smallest angle between adjacent strips.
    for (std::size_t vid = 0; vid < n; ++vid) {
        const auto& nb_list = adjacency[vid];
        if (nb_list.size() <= 2) continue;

        std::vector<Eigen::Vector2d> dirs;
        std::vector<double>          nb_radii;

        for (std::size_t nb : nb_list) {
            auto it_c1 = shared_center[vid].find(nb);
            if (it_c1 == shared_center[vid].end()) continue;
            const Eigen::VectorXd& c1 = it_c1->second;

            const Eigen::VectorXd* c2_ptr = nullptr;
            for (std::size_t other : adjacency[nb]) {
                if (other == vid) continue;
                auto it_c2 = shared_center[nb].find(other);
                if (it_c2 == shared_center[nb].end()) continue;
                if ((it_c2->second - c1).norm() > 0.0) {
                    c2_ptr = &it_c2->second;
                    break;
                }
            }
            if (!c2_ptr) continue;

            // Neighbour strip width
            const Strip nb_strip = strip_of(nb);
            if (!nb_strip.valid) continue;

            const Eigen::VectorXd d = *c2_ptr - c1;
            if (d.norm() < 1e-12) continue;
            if (d.size() < 2) continue;

            dirs.emplace_back(d[0], d[1]);
            nb_radii.push_back(nb_strip.width);
        }

        if (dirs.size() < 2) continue;

        std::vector<std::size_t> order(dirs.size());
        std::iota(order.begin(), order.end(), 0);
        std::sort(order.begin(), order.end(),
                  [&](std::size_t a, std::size_t b) {
                      const double aa = std::atan2(dirs[a][1], dirs[a][0]);
                      const double ab = std::atan2(dirs[b][1], dirs[b][0]);
                      return aa < ab;
                  });

        std::vector<double> angles(order.size());
        for (std::size_t k = 0; k < order.size(); ++k) {
            const auto& d = dirs[order[k]];
            angles[k] = std::atan2(d[1], d[0]);
        }

        std::vector<double> gaps(order.size());
        for (std::size_t k = 0; k + 1 < order.size(); ++k) {
            gaps[k] = angles[k + 1] - angles[k];
        }
        gaps.back() = 2.0 * kPi - angles.back() + angles.front();

        std::size_t min_idx = 0;
        for (std::size_t k = 1; k < gaps.size(); ++k) {
            if (gaps[k] < gaps[min_idx]) min_idx = k;
        }
        const double min_angle = gaps[min_idx];

        const std::size_t i = order[min_idx];
        const std::size_t j = order[(min_idx + 1) % order.size()];

        constexpr double kAngleTol = kPi / 10.0;
        if (min_angle < kAngleTol) continue;

        const double sin_half = std::sin(min_angle / 2.0);
        const double tan_half = std::tan(min_angle / 2.0);
        if (sin_half < 1e-12 || tan_half < 1e-12) continue;

        const double strip_radius = 2.0 * std::max(nb_radii[i], nb_radii[j]);

        const double r = 0.5 * std::sqrt(
            eps * eps +
            std::pow(strip_radius / sin_half + eps / tan_half, 2.0));

        radii[vid]        = r;
        radius_valid[vid] = true;
        directions[vid]   = Eigen::VectorXd();
    }

    double r_max = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        if (radius_valid[i] && radii[i] > r_max) r_max = radii[i];
    }

    if (out) {
        out->radii      = std::move(radii);
        out->directions = std::move(directions);
        out->r_max      = r_max;
    }
    return r_max;
};

// ---------------------------------------------------------------------------
// Final cover overlap derived from r_max
// ---------------------------------------------------------------------------

double compute_overlap(const Dataset&         data,
                       const Eigen::VectorXd& func_val,
                       double                 strip_width,
                       double                 eps) {
    if (strip_width <= 0.0 || eps <= 0.0) {
        return 0.0;
    }

    const double omega1 = compute_eps_len(data, func_val, eps);
    const double omega2 = compute_eps_len(data, func_val, strip_width / 2.0);

    const double denom = omega1 + omega2;
    if (denom <= 0.0) {
        return 0.0;
    }

    return 0.5 * omega1 / denom;
}

// ---------------------------------------------------------------------------
// Automatic cover overlap estimated from r_max (see header)
// ---------------------------------------------------------------------------

double estimate_overlap_from_r_max(const Dataset& data,
                                   double         eps,
                                   int            min_samples,
                                   int            split_threshold,
                                   double         fallback_overlap,
                                   double*        out_r_max) {
    double r_max = 0.0;

    TwoStepMapperParams tp;
    tp.overlap         = fallback_overlap;
    tp.eps             = eps;
    tp.min_samples        = min_samples;
    tp.split_threshold = split_threshold;
    tp.method          = ClusterMethod::EpsGraph;

    TwoStepMapper             tsm(tp);
    const TwoStepMapperResult initial = tsm.fit_transform(data);

    // If the initial graph has no singular node (degree >= 3), keep the default overlap.
    {
        const std::size_t n = initial.final_graph.num_nodes();
        std::vector<std::size_t> degree(n, 0);
        for (const auto& e : initial.final_graph.edges()) {
            if (e.first < n) ++degree[e.first];
            if (e.second < n) ++degree[e.second];
        }

        bool has_singular = false;
        for (std::size_t d : degree) {
            if (d >= 3) {
                has_singular = true;
                break;
            }
        }

        if (!has_singular) {
            if (out_r_max) *out_r_max = r_max;   // r_max == 0.0 here
            return fallback_overlap;
        }
    }

    StripRadiusResult sr;
    r_max = compute_graph_radius_by_strips(initial.final_graph, data, eps, &sr,
                                           /*singular_only=*/true);

    const double strip_width = 2.0 * r_max;

    double overlap = fallback_overlap;
    if (strip_width > 0.0) {
        // Overlap is computed against the main projection.
        const Eigen::VectorXd w_main = first_principal_direction(data);
        const Eigen::VectorXd fvals_main = data * w_main;
        const double computed = compute_overlap(data, fvals_main,
                                                strip_width, eps);

        if (computed > 0.0 && computed < 0.5) {
            overlap = computed;
        }
    }

    if (out_r_max) *out_r_max = r_max;
    return overlap;
}

} // namespace mapper