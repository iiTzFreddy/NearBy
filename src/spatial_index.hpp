#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include "geo.hpp"

// The interface shared by BruteForce, KdTree and QuadTree, so the tests, the
// benchmark and the CLI can treat every structure the same way.
//
// Rules for every implementation:
//  - build() keeps a reference to `pts`, which must outlive the index. Store
//    uint32_t indices into it, never copies of the coordinates.
//  - build() can be called again and replaces everything from the previous call.
//  - Results are sorted with ResultOrder and distances come from haversineKm.
//  - kNearest returns min(k, n) results, or none when k <= 0.
//  - withinRadius includes points at exactly r_km, and returns none when r_km < 0.
class SpatialIndex {
public:
    virtual void build(const std::vector<Location>& pts) = 0;
    virtual std::vector<Result> kNearest(double lat, double lon, int k) const = 0;
    virtual std::vector<Result> withinRadius(double lat, double lon, double r_km) const = 0;
    virtual size_t memoryBytes() const = 0;  // bytes owned by the structure itself
    virtual std::string name() const = 0;
    virtual ~SpatialIndex() = default;
};
