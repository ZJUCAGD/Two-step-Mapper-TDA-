#pragma once

#include "mapper/types.hpp"

#include <memory>
#include <vector>

namespace mapper {

// ---------------------------------------------------------------------------
// Abstract clustering interface.
//
// Every concrete clusterer returns one label per row of the input dataset.
// Labels < 0 mean "noise"; all
// non-negative labels are treated as cluster ids.
// ---------------------------------------------------------------------------
class Clusterer {
public:
    virtual ~Clusterer() = default;

    virtual std::vector<int> fit_predict(const Dataset& data) const = 0;

    // Neighbourhood radius used by this clusterer, if any. Returns 0 for
    // clusterers that do not have a natural eps scale.
    virtual double eps() const { return 0.0; }
};

using ClustererPtr = std::shared_ptr<const Clusterer>;

// ---------------------------------------------------------------------------
// Density-based spatial clustering of applications with noise (DBSCAN).
//
// A point is a core point if at least min_samples points lie within eps of
// it. Clusters are formed by connecting core points and their
// density-reachable neighbours. Points that are neither core nor reachable
// are labelled as noise (-1).
// ---------------------------------------------------------------------------
class DBSCAN : public Clusterer {
public:
    DBSCAN(double eps, int min_samples);
    ~DBSCAN() override;

    DBSCAN(const DBSCAN&) = delete;
    DBSCAN& operator=(const DBSCAN&) = delete;
    DBSCAN(DBSCAN&&) noexcept;
    DBSCAN& operator=(DBSCAN&&) noexcept;

    std::vector<int> fit_predict(const Dataset& data) const override;
    double eps() const override { return eps_; }

private:
    double eps_;
    int    min_samples_;

    struct Impl;
    mutable std::unique_ptr<Impl> impl_;

    std::vector<int> region_query(const Dataset& data, int i) const;
};

// ---------------------------------------------------------------------------
// Connectivity clustering on the eps-neighbourhood graph.
//
// Two points belong to the same cluster iff there is a chain of points
// p0, p1, ..., pk such that every consecutive pair is within eps. This is
// the connected-component decomposition of the eps-ball graph, equivalent
// to single-linkage clustering at threshold eps.
// ---------------------------------------------------------------------------
class EpsGraphClusterer : public Clusterer {
public:
    explicit EpsGraphClusterer(double eps);

    std::vector<int> fit_predict(const Dataset& data) const override;
    double eps() const override { return eps_; }

private:
    double eps_;
};

} // namespace mapper