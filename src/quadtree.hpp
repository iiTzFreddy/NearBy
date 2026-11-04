#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "spatial_index.hpp"

// Region quadtree over latitude/longitude. The root covers the whole world
// [-90, 90] x [-180, 180]; every split cuts a box into four equal quadrants.
//
// Layout: nodes live in one vector and children are always allocated as a block
// of four, so a node needs only the index of its first child. A leaf owns a slice
// of `items_`, which holds the point indices permuted so each leaf is contiguous.
// Nodes do not store their rectangle: it is recomputed on the way down.
class QuadTree : public SpatialIndex {
public:
    // A leaf splits when it holds more than bucketSize points, unless it is
    // already at maxDepth (identical coordinates would otherwise recurse forever).
    explicit QuadTree(int bucketSize = 16, int maxDepth = 20);

    void build(const std::vector<Location>& pts) override;
    std::vector<Result> kNearest(double lat, double lon, int k) const override;
    std::vector<Result> withinRadius(double lat, double lon, double r_km) const override;
    size_t memoryBytes() const override;
    std::string name() const override;

private:
    struct Node {
        uint32_t firstChild = 0;  // index of child 0; 0 means leaf (the root is never a child)
        uint32_t begin = 0;       // leaf only: start of its slice of items_
        uint32_t count = 0;       // leaf only: number of points in that slice
    };

    // Quadrant numbering, shared by build and queries:
    //   bit 0 set = east (lon >= midLon), bit 1 set = north (lat >= midLat)
    //   0 = south-west, 1 = south-east, 2 = north-west, 3 = north-east
    static int quadrantOf(const Rect& box, double lat, double lon);
    static Rect childRect(const Rect& box, int quadrant);

    static bool isLeaf(const Node& node) { return node.firstChild == 0; }

    void buildNode(uint32_t node, const Rect& box, uint32_t begin, uint32_t end, int depth);

    // Appends every point under `node` (whose rectangle is `box`) within r_km of
    // the query. Unsorted; withinRadius sorts the final list.
    void collectWithinRadius(uint32_t node, const Rect& box, double lat, double lon, double r_km,
                             std::vector<Result>& out) const;

    const std::vector<Location>* points_ = nullptr;
    std::vector<Node> nodes_;
    std::vector<uint32_t> items_;
    uint32_t bucketSize_;
    int maxDepth_;
};
