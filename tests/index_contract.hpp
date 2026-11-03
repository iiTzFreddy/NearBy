#pragma once

// Checks any SpatialIndex against BruteForce. KdTree and QuadTree get full
// coverage by calling runIndexContract(tree) from test_main.cpp.

#include <cstdint>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "brute.hpp"
#include "loader.hpp"

// Empty if the two lists match exactly, otherwise a description of the first
// difference. Distances are compared with ==: both must come from haversineKm.
inline std::string firstMismatch(const std::vector<Result>& expected,
                                 const std::vector<Result>& actual) {
    std::ostringstream out;
    out.precision(17);
    if (expected.size() != actual.size()) {
        out << "expected " << expected.size() << " results, got " << actual.size();
        return out.str();
    }
    for (size_t i = 0; i < expected.size(); ++i) {
        if (expected[i].idx != actual[i].idx || expected[i].dist_km != actual[i].dist_km) {
            out << "position " << i << ": expected idx " << expected[i].idx << " dist "
                << expected[i].dist_km << ", got idx " << actual[i].idx << " dist "
                << actual[i].dist_km;
            return out.str();
        }
    }
    return {};
}

namespace contract_detail {

struct Query {
    double lat;
    double lon;
};

inline double randomIn(std::mt19937_64& rng, double low, double high) {
    return low + (high - low) * (static_cast<double>(rng() >> 11) / 9007199254740992.0);
}

// Half the queries sit on data points; the rest are spread out, with fixed cases
// near the poles, along the antimeridian, and far from any data.
inline std::vector<Query> makeQueries(const std::vector<Location>& pts, size_t count,
                                      uint64_t seed) {
    std::vector<Query> queries = {{89.9, 10.0}, {-89.9, -170.0}, {90.0, 0.0},    {-90.0, 0.0},
                                  {0.0, 180.0}, {0.0, -180.0},   {10.0, 179.9},  {-10.0, -179.9},
                                  {-48.9, -123.4}};
    std::mt19937_64 rng(seed);
    while (queries.size() < count) {
        if (!pts.empty() && queries.size() % 2 == 0) {
            const Location& p = pts[rng() % pts.size()];
            queries.push_back({p.lat, p.lon});
        } else {
            queries.push_back({randomIn(rng, -90.0, 90.0), randomIn(rng, -180.0, 180.0)});
        }
    }
    return queries;
}

// Returns the number of failing queries.
inline int checkDataset(SpatialIndex& index, const std::string& label,
                        const std::vector<Location>& pts, size_t queryCount, uint64_t seed) {
    BruteForce brute;
    brute.build(pts);
    index.build(pts);

    const int n = static_cast<int>(pts.size());
    const std::vector<int> ks = {0, 1, 5, 10, 100, n, n + 1};
    const std::vector<double> radii = {0.0, 0.5, 5.0, 50.0, 500.0, 25000.0};

    int failures = 0;
    auto report = [&](const std::string& mismatch, const Query& q, const std::string& what) {
        if (mismatch.empty()) return;
        if (++failures <= 5) {
            std::cerr.precision(17);
            std::cerr << "MISMATCH " << index.name() << " on " << label << " (n=" << n
                      << ", seed=" << seed << ") " << what << " at (" << q.lat << ", " << q.lon
                      << "): " << mismatch << '\n';
        }
    };

    for (const Query& q : makeQueries(pts, queryCount, seed)) {
        for (int k : ks) {
            report(firstMismatch(brute.kNearest(q.lat, q.lon, k), index.kNearest(q.lat, q.lon, k)),
                   q, "kNearest k=" + std::to_string(k));
        }
        for (double r : radii) {
            report(firstMismatch(brute.withinRadius(q.lat, q.lon, r),
                                 index.withinRadius(q.lat, q.lon, r)),
                   q, "withinRadius r=" + std::to_string(r));
        }
    }
    return failures;
}

}  // namespace contract_detail

// Runs the plan's randomized and edge-case tests. Returns the failure count.
inline int runIndexContract(SpatialIndex& index) {
    using contract_detail::checkDataset;
    constexpr uint64_t kSeed = 42;
    constexpr size_t kQueries = 1000;

    const std::vector<Location> uniform = generateSynthetic(SyntheticKind::Uniform, 1000, kSeed);
    const std::vector<Location> clustered =
        generateSynthetic(SyntheticKind::Clustered, 1000, kSeed);

    const std::vector<Location> empty;
    const std::vector<Location> onePoint = {{7, 29.65, -82.32}};

    // Identical coordinates exercise the tie-break and the quadtree depth limit.
    // The ids run backwards so that array order and id order disagree.
    std::vector<Location> identical;
    for (int i = 0; i < 200; ++i) identical.push_back({1000 - i, 29.65, -82.32});

    // Duplicates mixed into real structure, with queries landing on them.
    std::vector<Location> mixed = clustered;
    for (int i = 0; i < 300; ++i) {
        const Location& source = clustered[static_cast<size_t>(i % 10)];
        mixed.push_back({5000 - i, source.lat, source.lon});
    }

    // Data on both sides of the antimeridian.
    std::vector<Location> antimeridian;
    for (int i = 0; i < 200; ++i) {
        const double offset = 0.01 * (i + 1);
        antimeridian.push_back({2 * i, 5.0, 180.0 - offset});
        antimeridian.push_back({2 * i + 1, 5.0, -180.0 + offset});
    }

    int failures = 0;
    failures += checkDataset(index, "uniform", uniform, kQueries, kSeed);
    failures += checkDataset(index, "clustered", clustered, kQueries, kSeed);
    failures += checkDataset(index, "empty", empty, 50, kSeed);
    failures += checkDataset(index, "one point", onePoint, 50, kSeed);
    failures += checkDataset(index, "identical", identical, 50, kSeed);
    failures += checkDataset(index, "mixed duplicates", mixed, 200, kSeed);
    failures += checkDataset(index, "antimeridian", antimeridian, 200, kSeed);
    return failures;
}
