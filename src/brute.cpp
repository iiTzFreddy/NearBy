#include "brute.hpp"

#include <algorithm>

void BruteForce::build(const std::vector<Location>& pts) {
    points_ = &pts;
}

std::vector<Result> BruteForce::kNearest(double lat, double lon, int k) const {
    if (points_ == nullptr || k <= 0) return {};

    const std::vector<Location>& pts = *points_;
    std::vector<Result> all;
    all.reserve(pts.size());
    for (size_t i = 0; i < pts.size(); ++i) {
        all.push_back({static_cast<uint32_t>(i), haversineKm(lat, lon, pts[i].lat, pts[i].lon)});
    }

    const size_t keep = std::min(static_cast<size_t>(k), all.size());
    std::partial_sort(all.begin(), all.begin() + static_cast<std::ptrdiff_t>(keep), all.end(),
                      ResultOrder{points_});
    all.resize(keep);
    return all;
}

std::vector<Result> BruteForce::withinRadius(double lat, double lon, double r_km) const {
    if (points_ == nullptr || r_km < 0.0) return {};

    const std::vector<Location>& pts = *points_;
    std::vector<Result> hits;
    for (size_t i = 0; i < pts.size(); ++i) {
        const double dist = haversineKm(lat, lon, pts[i].lat, pts[i].lon);
        if (dist <= r_km) hits.push_back({static_cast<uint32_t>(i), dist});
    }

    std::sort(hits.begin(), hits.end(), ResultOrder{points_});
    return hits;
}

size_t BruteForce::memoryBytes() const {
    return sizeof(*this);
}

std::string BruteForce::name() const {
    return "BruteForce";
}
