// Edge cases the shared contract doesn't reach, run against every tree: data
// exactly on the poles and on the 180th meridian, exact ties on opposite sides of a
// split, a radius exactly equal to a point's distance, and extreme k and radius.
// Besides comparing with BruteForce, some checks assert the expected answer
// directly, so a bug that brute force shares would still be caught.

#include <algorithm>
#include <climits>
#include <iostream>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "index_contract.hpp"
#include "kdtree.hpp"
#include "quadtree.hpp"

namespace {

int failures = 0;

#define CHECK(condition)                                                                  \
    do {                                                                                  \
        if (!(condition)) {                                                               \
            ++failures;                                                                   \
            std::cerr << "FAILED " << __FILE__ << ":" << __LINE__ << ": " #condition "\n"; \
        }                                                                                 \
    } while (false)

struct Probe {
    double lat;
    double lon;
};

std::vector<std::unique_ptr<SpatialIndex>> trees() {
    std::vector<std::unique_ptr<SpatialIndex>> all;
    all.push_back(std::make_unique<KdTree>());
    all.push_back(std::make_unique<QuadTree>());
    all.push_back(std::make_unique<QuadTree>(1, 24));  // one point per leaf where possible
    return all;
}

// Results must be in strict (dist_km, id) order, with distances from haversineKm.
bool wellOrdered(const std::vector<Result>& results, const std::vector<Location>& pts, double lat,
                 double lon) {
    const ResultOrder order{&pts};
    for (size_t i = 0; i < results.size(); ++i) {
        const Location& p = pts[results[i].idx];
        if (results[i].dist_km != haversineKm(lat, lon, p.lat, p.lon)) return false;
        if (i > 0 && !order(results[i - 1], results[i])) return false;
    }
    return true;
}

// Compares every tree with brute force on the given probes, for a spread of k and
// radius values, and checks the ordering of each answer independently.
void compareAll(const std::string& label, const std::vector<Location>& pts,
                const std::vector<Probe>& probes) {
    BruteForce brute;
    brute.build(pts);
    const int n = static_cast<int>(pts.size());
    const std::vector<int> ks = {1, 2, 3, 5, 10, n, n + 1, INT_MAX};
    const std::vector<double> radii = {0.0, 1e-9, 0.001, 1.0, 100.0, 5000.0, 20015.1, 1e9,
                                       std::numeric_limits<double>::infinity()};

    for (std::unique_ptr<SpatialIndex>& tree : trees()) {
        tree->build(pts);
        int bad = 0;
        for (const Probe& q : probes) {
            for (int k : ks) {
                const std::vector<Result> expected = brute.kNearest(q.lat, q.lon, k);
                const std::vector<Result> actual = tree->kNearest(q.lat, q.lon, k);
                const std::string mismatch = firstMismatch(expected, actual);
                const bool ordered = wellOrdered(actual, pts, q.lat, q.lon);
                const bool sized = k <= 0 || actual.size() == std::min<size_t>(static_cast<size_t>(k), pts.size());
                if (!mismatch.empty() || !ordered || !sized) {
                    if (++bad <= 3) {
                        std::cerr << "MISMATCH " << tree->name() << " on " << label << " kNearest k=" << k
                                  << " at (" << q.lat << ", " << q.lon << "): "
                                  << (mismatch.empty() ? (ordered ? "wrong size" : "bad order") : mismatch) << '\n';
                    }
                }
            }
            for (double r : radii) {
                const std::vector<Result> expected = brute.withinRadius(q.lat, q.lon, r);
                const std::vector<Result> actual = tree->withinRadius(q.lat, q.lon, r);
                const std::string mismatch = firstMismatch(expected, actual);
                const bool ordered = wellOrdered(actual, pts, q.lat, q.lon);
                if (!mismatch.empty() || !ordered) {
                    if (++bad <= 3) {
                        std::cerr << "MISMATCH " << tree->name() << " on " << label << " withinRadius r=" << r
                                  << " at (" << q.lat << ", " << q.lon << "): "
                                  << (mismatch.empty() ? "bad order" : mismatch) << '\n';
                    }
                }
            }
        }
        failures += bad;
    }
}

// Many points at each pole, at different longitudes. Each pile is one physical spot,
// so distances between them are near zero and ties are everywhere.
void testDataOnThePoles() {
    std::vector<Location> pts;
    int64_t id = 1000;
    for (int i = 0; i <= 72; ++i) {
        const double lon = -180.0 + 5.0 * i;
        pts.push_back({id--, 90.0, lon});
        pts.push_back({id--, -90.0, lon});
        pts.push_back({id--, 89.99, lon});
        pts.push_back({id--, -89.95, lon});
    }
    const std::vector<Location> extra = generateSynthetic(SyntheticKind::Uniform, 300, 31);
    for (const Location& p : extra) pts.push_back({5000 + p.id, p.lat, p.lon});

    const std::vector<Probe> probes = {{90.0, 0.0},   {90.0, 180.0},  {90.0, -180.0}, {-90.0, 37.5},
                                       {89.999, 0.0}, {-89.97, 90.0}, {85.0, -45.0},  {0.0, 0.0}};
    compareAll("poles", pts, probes);
}

// Points exactly on +180 and -180, the same meridian written two ways.
void testDataOnTheAntimeridian() {
    std::vector<Location> pts;
    for (int i = 0; i < 60; ++i) {
        const double lat = -60.0 + 2.0 * i;
        pts.push_back({2 * i, lat, 180.0});
        pts.push_back({2 * i + 1, lat, -180.0});
        pts.push_back({1000 + i, lat, 179.999});
        pts.push_back({2000 + i, lat, -179.999});
    }
    const std::vector<Probe> probes = {{0.0, 180.0},  {0.0, -180.0},   {10.0, 179.9999}, {10.0, -179.9999},
                                       {-61.0, 180.0}, {90.0, 180.0}, {0.0, 0.0},        {30.0, 90.0}};
    compareAll("antimeridian", pts, probes);
}

// Pairs at exactly the same distance from the query on opposite sides (east/west and
// north/south give bit-identical haversine distances). The split puts each pair in
// different subtrees, so the tie-break has to work across them.
void testExactTiesAcrossSplits() {
    std::vector<Location> pts;
    for (int i = 1; i <= 40; ++i) {
        const double d = 0.01 * i;
        // Larger id on the side visited first, so a wrong tie-break shows up.
        pts.push_back({1000 + i, 0.0, d});
        pts.push_back({i, 0.0, -d});
        pts.push_back({3000 + i, d, 0.0});
        pts.push_back({2000 + i, -d, 0.0});
    }
    const std::vector<Probe> probes = {{0.0, 0.0}};
    compareAll("exact ties", pts, probes);

    // And directly: the nearest east/west pair is ids 1 (west) and 1001 (east),
    // at the same distance, so id 1 must come first.
    for (std::unique_ptr<SpatialIndex>& tree : trees()) {
        tree->build(pts);
        const std::vector<Result> two = tree->kNearest(0.0, 0.0, 4);
        CHECK(two.size() == 4);
        if (two.size() == 4) {
            CHECK(two[0].dist_km == two[1].dist_km);
            CHECK(pts[two[0].idx].id < pts[two[1].idx].id);
        }
    }
}

// A radius exactly equal to a point's distance must include that point.
void testRadiusExactlyAtAPoint() {
    const std::vector<Location> pts = generateSynthetic(SyntheticKind::Clustered, 2000, 41);
    const double qLat = 29.65;
    const double qLon = -82.32;

    for (std::unique_ptr<SpatialIndex>& tree : trees()) {
        tree->build(pts);
        for (size_t i = 0; i < pts.size(); i += 97) {
            const double r = haversineKm(qLat, qLon, pts[i].lat, pts[i].lon);
            const std::vector<Result> hits = tree->withinRadius(qLat, qLon, r);
            const bool found = std::any_of(hits.begin(), hits.end(),
                                           [&](const Result& h) { return h.idx == i; });
            CHECK(found);
        }
    }
}

// Extreme arguments on real-sized and on empty data.
void testExtremeArguments() {
    const std::vector<Location> pts = generateSynthetic(SyntheticKind::Uniform, 500, 51);
    const std::vector<Location> none;
    const double inf = std::numeric_limits<double>::infinity();

    for (std::unique_ptr<SpatialIndex>& tree : trees()) {
        tree->build(pts);
        CHECK(tree->kNearest(10.0, 10.0, INT_MAX).size() == pts.size());
        CHECK(tree->withinRadius(10.0, 10.0, inf).size() == pts.size());
        CHECK(tree->withinRadius(10.0, 10.0, 1e300).size() == pts.size());
        CHECK(tree->kNearest(10.0, 10.0, INT_MIN).empty());
        CHECK(tree->withinRadius(10.0, 10.0, -inf).empty());

        tree->build(none);
        CHECK(tree->kNearest(10.0, 10.0, INT_MAX).empty());
        CHECK(tree->withinRadius(10.0, 10.0, inf).empty());
    }
}

}  // namespace

int main() {
    testDataOnThePoles();
    testDataOnTheAntimeridian();
    testExactTiesAcrossSplits();
    testRadiusExactlyAtAPoint();
    testExtremeArguments();

    if (failures != 0) {
        std::cerr << failures << " edge-case check(s) failed\n";
        return 1;
    }
    std::cout << "all edge-case tests passed\n";
    return 0;
}
