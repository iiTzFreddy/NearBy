#include "kdtree.hpp"

#include <algorithm>
#include <utility>

#include "bounded_heap.hpp"

namespace {

double coord(const Location& p, int axis) {
    return axis == 0 ? p.lat : p.lon;
}

}  // namespace

void quickselect(std::vector<uint32_t>& idx, size_t begin, size_t end, size_t nth,
                 const std::vector<Location>& pts, int axis) {
    auto key = [&](size_t i) { return coord(pts[idx[i]], axis); };

    while (end - begin > 1) {
        // Median of three as the pivot, so already-sorted input stays fast.
        const size_t mid = begin + (end - begin) / 2;
        double a = key(begin), b = key(mid), c = key(end - 1);
        if (a > b) std::swap(a, b);
        if (b > c) std::swap(b, c);
        if (a > b) std::swap(a, b);
        const double pivot = b;

        // Three-way partition: [begin, lt) < pivot, [lt, gt) == pivot, [gt, end) > pivot.
        // Grouping equal keys keeps many identical coordinates from going quadratic.
        size_t lt = begin, i = begin, gt = end;
        while (i < gt) {
            const double k = key(i);
            if (k < pivot) {
                std::swap(idx[lt++], idx[i++]);
            } else if (k > pivot) {
                std::swap(idx[i], idx[--gt]);
            } else {
                ++i;
            }
        }

        if (nth < lt) {
            end = lt;
        } else if (nth >= gt) {
            begin = gt;
        } else {
            return;  // nth landed among the keys equal to the pivot
        }
    }
}

void KdTree::build(const std::vector<Location>& pts) {
    points_ = &pts;
    order_.resize(pts.size());
    for (size_t i = 0; i < pts.size(); ++i) order_[i] = static_cast<uint32_t>(i);

    nodes_.clear();
    nodes_.reserve(2 * (pts.size() / kLeafSize + 1));
    if (!pts.empty()) buildNode(0, static_cast<uint32_t>(pts.size()), 0);

    nodes_.shrink_to_fit();
    order_.shrink_to_fit();
}

int32_t KdTree::buildNode(uint32_t begin, uint32_t end, int depth) {
    const std::vector<Location>& pts = *points_;

    Rect box = {90.0, -90.0, 180.0, -180.0};
    for (uint32_t i = begin; i < end; ++i) {
        const Location& p = pts[order_[i]];
        box.minLat = std::min(box.minLat, p.lat);
        box.maxLat = std::max(box.maxLat, p.lat);
        box.minLon = std::min(box.minLon, p.lon);
        box.maxLon = std::max(box.maxLon, p.lon);
    }

    const int32_t self = static_cast<int32_t>(nodes_.size());
    nodes_.push_back({box, begin, end, -1, -1});
    if (end - begin <= kLeafSize) return self;

    const uint32_t mid = begin + (end - begin) / 2;
    quickselect(order_, begin, end, mid, pts, depth % 2);

    // nodes_ can reallocate while the children are built, so store indices
    // afterwards instead of holding a reference to this node.
    const int32_t left = buildNode(begin, mid, depth + 1);
    const int32_t right = buildNode(mid, end, depth + 1);
    nodes_[self].left = left;
    nodes_[self].right = right;
    return self;
}

std::vector<Result> KdTree::kNearest(double lat, double lon, int k) const {
    if (points_ == nullptr || nodes_.empty() || k <= 0) return {};

    const std::vector<Location>& pts = *points_;
    BoundedMaxHeap heap(std::min(static_cast<size_t>(k), pts.size()), pts);

    // Depth-first with an explicit stack. Each entry carries its node's bound so
    // it can be checked again when popped, after the heap has improved.
    std::vector<std::pair<int32_t, double>> stack;
    stack.push_back({0, minDistToRectKm(lat, lon, nodes_[0].box)});

    while (!stack.empty()) {
        const int32_t nodeIdx = stack.back().first;
        const double bound = stack.back().second;
        stack.pop_back();

        // Strictly greater: a subtree whose bound equals the k-th distance can
        // still hold a tie with a smaller id (see geo.hpp).
        if (heap.full() && bound > heap.top().dist_km) continue;

        const Node& node = nodes_[nodeIdx];
        if (node.left < 0) {
            for (uint32_t i = node.begin; i < node.end; ++i) {
                const Location& p = pts[order_[i]];
                heap.offer({order_[i], haversineKm(lat, lon, p.lat, p.lon)});
            }
            continue;
        }

        // Push the far child first so the near child is searched first. That
        // fills the heap with close results early and lets the far side be pruned.
        const double leftBound = minDistToRectKm(lat, lon, nodes_[node.left].box);
        const double rightBound = minDistToRectKm(lat, lon, nodes_[node.right].box);
        if (leftBound <= rightBound) {
            stack.push_back({node.right, rightBound});
            stack.push_back({node.left, leftBound});
        } else {
            stack.push_back({node.left, leftBound});
            stack.push_back({node.right, rightBound});
        }
    }

    return heap.takeSorted();
}

std::vector<Result> KdTree::withinRadius(double lat, double lon, double r_km) const {
    if (points_ == nullptr || nodes_.empty() || r_km < 0.0) return {};

    const std::vector<Location>& pts = *points_;
    std::vector<Result> hits;
    std::vector<int32_t> stack = {0};

    while (!stack.empty()) {
        const Node& node = nodes_[stack.back()];
        stack.pop_back();

        if (minDistToRectKm(lat, lon, node.box) > r_km) continue;

        if (node.left < 0) {
            for (uint32_t i = node.begin; i < node.end; ++i) {
                const Location& p = pts[order_[i]];
                const double dist = haversineKm(lat, lon, p.lat, p.lon);
                if (dist <= r_km) hits.push_back({order_[i], dist});
            }
        } else {
            stack.push_back(node.left);
            stack.push_back(node.right);
        }
    }

    std::sort(hits.begin(), hits.end(), ResultOrder{points_});
    return hits;
}

size_t KdTree::memoryBytes() const {
    return sizeof(*this) + nodes_.capacity() * sizeof(Node) +
           order_.capacity() * sizeof(uint32_t);
}

std::string KdTree::name() const {
    return "KdTree";
}
