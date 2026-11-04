#include "quadtree.hpp"

#include <algorithm>
#include <utility>

#include "bounded_heap.hpp"

namespace {

// Moves every index for which `second` is true to the back of items[begin, end)
// and returns where that back part starts. Works in place with no extra array.
template <typename Pred>
uint32_t partitionRange(std::vector<uint32_t>& items, uint32_t begin, uint32_t end, Pred second) {
    uint32_t front = begin;  // items[begin, front) are known to be "first"
    uint32_t back = end;     // items[back, end) are known to be "second"
    while (front < back) {
        if (!second(items[front])) {
            ++front;
        } else {
            --back;
            std::swap(items[front], items[back]);  // the swapped-in item is unchecked
        }
    }
    return front;
}

// A binary heap whose top() is the element that comes first under `before`, i.e.
// no other element y has before(y, top()). kNearest uses it as a min-heap of nodes
// by lower bound; the k best results go in the shared BoundedMaxHeap instead.
template <typename T, typename Before>
class Heap {
public:
    explicit Heap(Before before) : before_(before) {}

    bool empty() const { return data_.empty(); }
    size_t size() const { return data_.size(); }
    const T& top() const { return data_.front(); }

    void push(const T& value) {
        data_.push_back(value);
        size_t i = data_.size() - 1;
        while (i > 0) {
            const size_t parent = (i - 1) / 2;
            if (!before_(data_[i], data_[parent])) break;
            std::swap(data_[i], data_[parent]);
            i = parent;
        }
    }

    void pop() {
        data_.front() = data_.back();
        data_.pop_back();
        size_t i = 0;
        while (true) {
            const size_t left = 2 * i + 1;
            if (left >= data_.size()) break;
            const size_t right = left + 1;
            const size_t first = (right < data_.size() && before_(data_[right], data_[left])) ? right : left;
            if (!before_(data_[first], data_[i])) break;
            std::swap(data_[first], data_[i]);
            i = first;
        }
    }

private:
    std::vector<T> data_;
    Before before_;
};

// A quadtree node waiting to be visited by kNearest. `bound` is the lower bound on
// the distance from the query to anything inside `box`.
struct PendingNode {
    double bound;
    uint32_t node;
    Rect box;
};

}  // namespace

QuadTree::QuadTree(int bucketSize, int maxDepth)
    : bucketSize_(static_cast<uint32_t>(std::max(bucketSize, 1))),
      maxDepth_(std::max(maxDepth, 0)) {}

int QuadTree::quadrantOf(const Rect& box, double lat, double lon) {
    const double midLat = (box.minLat + box.maxLat) / 2.0;
    const double midLon = (box.minLon + box.maxLon) / 2.0;
    return (lat >= midLat ? 2 : 0) | (lon >= midLon ? 1 : 0);
}

Rect QuadTree::childRect(const Rect& box, int quadrant) {
    const double midLat = (box.minLat + box.maxLat) / 2.0;
    const double midLon = (box.minLon + box.maxLon) / 2.0;
    Rect child = box;
    if (quadrant & 1) child.minLon = midLon; else child.maxLon = midLon;
    if (quadrant & 2) child.minLat = midLat; else child.maxLat = midLat;
    return child;
}

void QuadTree::build(const std::vector<Location>& pts) {
    points_ = &pts;

    items_.resize(pts.size());
    for (size_t i = 0; i < pts.size(); ++i) items_[i] = static_cast<uint32_t>(i);

    nodes_.clear();
    nodes_.emplace_back();  // the root, index 0
    buildNode(0, Rect{-90.0, 90.0, -180.0, 180.0}, 0, static_cast<uint32_t>(pts.size()), 0);
    nodes_.shrink_to_fit();
}

// Builds the subtree for items_[begin, end), which all lie inside `box`.
void QuadTree::buildNode(uint32_t node, const Rect& box, uint32_t begin, uint32_t end, int depth) {
    const uint32_t count = end - begin;
    if (count <= bucketSize_ || depth >= maxDepth_) {
        // A leaf. At maxDepth it may hold more than bucketSize_ points.
        nodes_[node].begin = begin;
        nodes_[node].count = count;
        return;
    }

    // Reorder the slice into quadrant order 0, 1, 2, 3 using two passes: split
    // south from north, then split each half west from east.
    const std::vector<Location>& pts = *points_;
    auto isNorth = [&](uint32_t i) { return (quadrantOf(box, pts[i].lat, pts[i].lon) & 2) != 0; };
    auto isEast = [&](uint32_t i) { return (quadrantOf(box, pts[i].lat, pts[i].lon) & 1) != 0; };

    const uint32_t northStart = partitionRange(items_, begin, end, isNorth);
    const uint32_t southEast = partitionRange(items_, begin, northStart, isEast);
    const uint32_t northEast = partitionRange(items_, northStart, end, isEast);
    const uint32_t cuts[5] = {begin, southEast, northStart, northEast, end};

    // Index, not reference: resize may move the vector, and the recursion below
    // grows it again.
    const uint32_t firstChild = static_cast<uint32_t>(nodes_.size());
    nodes_[node].firstChild = firstChild;
    nodes_.resize(nodes_.size() + 4);

    for (int q = 0; q < 4; ++q) {
        buildNode(firstChild + static_cast<uint32_t>(q), childRect(box, q), cuts[q], cuts[q + 1],
                  depth + 1);
    }
}

// Best-first search. Nodes wait in a priority queue ordered by their lower bound,
// so the closest unexplored region is always opened next. `best` holds the k best
// points found so far with the worst on top; once it is full, its top is the
// distance a region must beat to be worth opening.
std::vector<Result> QuadTree::kNearest(double lat, double lon, int k) const {
    if (points_ == nullptr || points_->empty() || nodes_.empty() || k <= 0) return {};

    const std::vector<Location>& pts = *points_;
    BoundedMaxHeap best(std::min(static_cast<size_t>(k), pts.size()), pts);

    auto closestFirst = [](const PendingNode& a, const PendingNode& b) { return a.bound < b.bound; };
    Heap<PendingNode, decltype(closestFirst)> pending(closestFirst);

    // Pruning is strict (">"): a region whose bound equals the k-th distance can
    // still hold a point at that same distance with a smaller id.
    auto cannotHelp = [&](double bound) { return best.full() && bound > best.top().dist_km; };

    const Rect world{-90.0, 90.0, -180.0, 180.0};
    pending.push({minDistToRectKm(lat, lon, world), 0, world});

    while (!pending.empty()) {
        const PendingNode current = pending.top();
        pending.pop();
        // Everything still queued has a bound at least this large, so we are done.
        if (cannotHelp(current.bound)) break;

        const Node& node = nodes_[current.node];
        if (isLeaf(node)) {
            for (uint32_t s = node.begin; s < node.begin + node.count; ++s) {
                const uint32_t idx = items_[s];
                best.offer({idx, haversineKm(lat, lon, pts[idx].lat, pts[idx].lon)});
            }
            continue;
        }

        for (int q = 0; q < 4; ++q) {
            const uint32_t child = node.firstChild + static_cast<uint32_t>(q);
            if (isLeaf(nodes_[child]) && nodes_[child].count == 0) continue;  // nothing in it

            const Rect childBox = childRect(current.box, q);
            const double bound = minDistToRectKm(lat, lon, childBox);
            if (!cannotHelp(bound)) pending.push({bound, child, childBox});
        }
    }

    return best.takeSorted();
}

std::vector<Result> QuadTree::withinRadius(double lat, double lon, double r_km) const {
    if (points_ == nullptr || nodes_.empty() || r_km < 0.0) return {};

    std::vector<Result> hits;
    collectWithinRadius(0, Rect{-90.0, 90.0, -180.0, 180.0}, lat, lon, r_km, hits);
    std::sort(hits.begin(), hits.end(), ResultOrder{points_});
    return hits;
}

// Depth-first descent that skips any quadrant lying entirely farther than r_km.
// The depth is at most maxDepth_, so plain recursion is safe.
void QuadTree::collectWithinRadius(uint32_t node, const Rect& box, double lat, double lon,
                                   double r_km, std::vector<Result>& out) const {
    const Node& current = nodes_[node];

    if (isLeaf(current)) {
        const std::vector<Location>& pts = *points_;
        for (uint32_t s = current.begin; s < current.begin + current.count; ++s) {
            const uint32_t idx = items_[s];
            const double dist = haversineKm(lat, lon, pts[idx].lat, pts[idx].lon);
            if (dist <= r_km) out.push_back({idx, dist});
        }
        return;
    }

    for (int q = 0; q < 4; ++q) {
        const uint32_t child = current.firstChild + static_cast<uint32_t>(q);
        if (isLeaf(nodes_[child]) && nodes_[child].count == 0) continue;  // nothing in it

        const Rect childBox = childRect(box, q);
        // Strictly greater: a quadrant whose bound equals r_km may hold a point at r_km.
        if (minDistToRectKm(lat, lon, childBox) > r_km) continue;
        collectWithinRadius(child, childBox, lat, lon, r_km, out);
    }
}

size_t QuadTree::memoryBytes() const {
    return sizeof(*this) + nodes_.capacity() * sizeof(Node) + items_.capacity() * sizeof(uint32_t);
}

std::string QuadTree::name() const {
    return "QuadTree";
}
