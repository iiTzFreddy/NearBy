#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "spatial_index.hpp"

// 2-d tree over (lat, lon). Each level splits its points at the median of
// latitude (even depth) or longitude (odd depth), down to leaves of at most
// kLeafSize points. Every node stores the tight bounding box of its points, and
// pruning uses minDistToRectKm on that box, so the sphere and the antimeridian
// are handled by geo.cpp rather than by the tree.
class KdTree : public SpatialIndex {
public:
    static constexpr uint32_t kLeafSize = 8;

    void build(const std::vector<Location>& pts) override;
    std::vector<Result> kNearest(double lat, double lon, int k) const override;
    std::vector<Result> withinRadius(double lat, double lon, double r_km) const override;
    size_t memoryBytes() const override;
    std::string name() const override;

private:
    struct Node {
        Rect box;        // tight bounds of the points under this node
        uint32_t begin;  // range [begin, end) of order_ owned by this node
        uint32_t end;
        int32_t left;    // child node indices, -1 for a leaf
        int32_t right;
    };

    int32_t buildNode(uint32_t begin, uint32_t end, int depth);

    const std::vector<Location>* points_ = nullptr;
    std::vector<uint32_t> order_;  // indices into *points_, grouped by node
    std::vector<Node> nodes_;      // nodes_[0] is the root when non-empty
};

// Rearranges idx[begin, end) so that idx[nth] holds the point that would be
// there if the range were sorted by `axis` (0 = lat, 1 = lon). Everything before
// it is <= and everything after it is >= on that axis. Expected O(end - begin).
void quickselect(std::vector<uint32_t>& idx, size_t begin, size_t end, size_t nth,
                 const std::vector<Location>& pts, int axis);
