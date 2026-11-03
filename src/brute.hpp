#pragma once

#include "spatial_index.hpp"

// O(n) scan over every point. This is the correctness reference: every test
// compares a tree's output against it.
class BruteForce : public SpatialIndex {
public:
    void build(const std::vector<Location>& pts) override;
    std::vector<Result> kNearest(double lat, double lon, int k) const override;
    std::vector<Result> withinRadius(double lat, double lon, double r_km) const override;
    size_t memoryBytes() const override;
    std::string name() const override;

private:
    const std::vector<Location>* points_ = nullptr;
};
