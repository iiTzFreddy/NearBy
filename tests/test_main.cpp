#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "bench.hpp"
#include "bounded_heap.hpp"
#include "brute.hpp"
#include "geo.hpp"
#include "index_contract.hpp"
#include "kdtree.hpp"
#include "loader.hpp"

namespace {

int failures = 0;

#define CHECK(condition)                                                              \
    do {                                                                              \
        if (!(condition)) {                                                           \
            ++failures;                                                               \
            std::cerr << "FAILED " << __FILE__ << ":" << __LINE__ << ": " #condition "\n"; \
        }                                                                             \
    } while (false)

bool near(double a, double b, double tolerance) {
    return std::fabs(a - b) <= tolerance;
}

std::string tempPath(const std::string& name) {
    return (std::filesystem::temp_directory_path() / ("nearby_test_" + name)).string();
}

void testHaversine() {
    constexpr double kPi = 3.14159265358979323846;

    CHECK(haversineKm(29.65, -82.32, 29.65, -82.32) == 0.0);
    CHECK(near(haversineKm(0, 0, 90, 0), kPi / 2 * kEarthRadiusKm, 1e-6));
    CHECK(near(haversineKm(0, 0, 0, 180), kPi * kEarthRadiusKm, 1e-3));
    CHECK(near(haversineKm(51.5074, -0.1278, 48.8566, 2.3522), 343.6, 1.0));  // London-Paris

    // One degree of longitude at the equator, measured across the antimeridian.
    CHECK(near(haversineKm(0, 179.5, 0, -179.5), kPi / 180 * kEarthRadiusKm, 1e-6));

    // Argument order must not change a single bit.
    CHECK(haversineKm(10.5, 20.25, -33.1, 151.2) == haversineKm(-33.1, 151.2, 10.5, 20.25));
}

void testRectBoundCases() {
    const Rect world = {-90, 90, -180, 180};
    CHECK(minDistToRectKm(29.65, -82.32, world) == 0.0);

    const Rect box = {10, 20, 30, 40};
    CHECK(minDistToRectKm(15, 35, box) == 0.0);
    CHECK(near(minDistToRectKm(25, 35, box), haversineKm(25, 35, 20, 35), 0.01));

    // The case a plain splitting-line bound gets wrong: the box is 2 degrees away
    // across the antimeridian, not 179 degrees away through lon 0.
    const Rect eastHalf = {-90, 90, 0, 180};
    CHECK(minDistToRectKm(0, -179, eastHalf) < 112.0);
    const Rect nearLine = {-10, 10, 170, 179};
    CHECK(near(minDistToRectKm(0, -179, nearLine), haversineKm(0, -179, 0, 179), 0.01));

    // At high latitude the nearest point of a meridian edge is poleward of the
    // query, so a corner or same-latitude shortcut would overestimate.
    const Rect tall = {0, 89, 40, 50};
    const double bound = minDistToRectKm(70, 0, tall);
    CHECK(bound <= haversineKm(70, 0, 76.0, 40));
    CHECK(bound < haversineKm(70, 0, 70, 40));

    // A box touching the pole is reachable over the top.
    const Rect polar = {80, 90, 170, 180};
    CHECK(minDistToRectKm(85, 0, polar) <= haversineKm(85, 0, 90, 0));
}

Rect randomRect(std::mt19937_64& rng) {
    using contract_detail::randomIn;
    const double latA = randomIn(rng, -90, 90);
    const double latB = randomIn(rng, -90, 90);
    const double lonA = randomIn(rng, -180, 180);
    const double lonB = randomIn(rng, -180, 180);
    return {std::min(latA, latB), std::max(latA, latB), std::min(lonA, lonB), std::max(lonA, lonB)};
}

// A bound that overestimates makes trees silently lose neighbours, so this checks
// it from both sides: never above any point in the box, and not uselessly below.
void testRectBoundRandomized() {
    using contract_detail::randomIn;
    std::mt19937_64 rng(7);
    constexpr int kGrid = 60;

    for (int trial = 0; trial < 20000; ++trial) {
        const Rect box = randomRect(rng);
        const double lat = randomIn(rng, -90, 90);
        const double lon = randomIn(rng, -180, 180);
        const double bound = minDistToRectKm(lat, lon, box);

        for (int i = 0; i < 40; ++i) {
            const double pLat = randomIn(rng, box.minLat, box.maxLat);
            const double pLon = randomIn(rng, box.minLon, box.maxLon);
            CHECK(bound <= haversineKm(lat, lon, pLat, pLon));
        }

        if (trial < 2000) {
            double sampledMin = 1e18;
            for (int i = 0; i <= kGrid; ++i) {
                for (int j = 0; j <= kGrid; ++j) {
                    const double pLat = box.minLat + (box.maxLat - box.minLat) * i / kGrid;
                    const double pLon = box.minLon + (box.maxLon - box.minLon) * j / kGrid;
                    sampledMin = std::min(sampledMin, haversineKm(lat, lon, pLat, pLon));
                }
            }
            CHECK(bound <= sampledMin);
            // Grid cells are at most 3 x 6 degrees, so the nearest grid point is
            // within half a cell diagonal (about 375 km) of the true minimum.
            CHECK(sampledMin - bound < 500.0);
        }
    }

    // Points on the box's own edges and corners are the tightest case for rounding.
    for (int trial = 0; trial < 200000; ++trial) {
        const Rect box = randomRect(rng);
        const double lat = randomIn(rng, -90, 90);
        const double lon = randomIn(rng, -180, 180);
        const double bound = minDistToRectKm(lat, lon, box);
        CHECK(bound <= haversineKm(lat, lon, box.minLat, box.minLon));
        CHECK(bound <= haversineKm(lat, lon, box.maxLat, box.maxLon));
        CHECK(bound <= haversineKm(lat, lon, box.minLat, box.maxLon));
        CHECK(bound <= haversineKm(lat, lon, box.maxLat, box.minLon));
    }

    // Nearly antipodal pairs, where haversine rounding is at its worst.
    for (int trial = 0; trial < 200000; ++trial) {
        const double pLat = randomIn(rng, -90, 90);
        const double pLon = randomIn(rng, -180, 180);
        const double lat = -pLat + randomIn(rng, -1e-5, 1e-5);
        const double lon = std::clamp(pLon + (pLon > 0 ? -180.0 : 180.0) + randomIn(rng, -1e-5, 1e-5),
                                      -180.0, 180.0);
        const Rect box = {pLat, pLat, pLon, pLon};
        CHECK(minDistToRectKm(lat, lon, box) <= haversineKm(lat, lon, pLat, pLon));
    }
}

void testResultOrder() {
    const std::vector<Location> pts = {{30, 0, 0}, {10, 0, 0}, {20, 0, 0}};
    std::vector<Result> results = {{0, 1.0}, {1, 1.0}, {2, 0.5}};
    std::sort(results.begin(), results.end(), ResultOrder{&pts});
    CHECK(results[0].idx == 2);  // smallest distance first
    CHECK(results[1].idx == 1);  // equal distance: id 10 before id 30
    CHECK(results[2].idx == 0);
}

void testParseGowalla() {
    const std::string path = tempPath("raw.txt");
    {
        std::ofstream out(path, std::ios::binary);
        out << "0\t2010-10-19T23:55:27Z\t30.2359\t-97.7951\t22847\n"    // kept
               "0\t2010-10-18T22:17:43Z\t30.2691\t-97.7494\t420315\r\n" // kept, CRLF ending
               "1\t2010-10-17T23:42:03Z\t30.2359\t-97.7951\t22847\n"    // duplicate id
               "1\t2010-10-17T19:26:05Z\t91.5\t-97.7\t5\n"              // lat out of range
               "1\t2010-10-17T19:26:05Z\t30.0\t-180.5\t6\n"             // lon out of range
               "2\t2010-10-17T19:26:05Z\tnan\t10.0\t8\n"                // lat is not a number
               "2\t2010-10-16T18:50:42Z\t0.0\t0.0\t7\n"                 // placeholder (0, 0)
               "2\t2010-10-16T18:50:42Z\t0.0\t12.5\t9\n"                // on the equator: kept
               "3\t2010-10-12T23:58:03Z\tabc\t-97.7\t8\n"               // malformed lat
               "3\t2010-10-12T23:58:03Z\t30.2\n"                        // too few fields
               "\n"
               "3\t2010-10-12T23:58:03Z\t30.2\t-97.7\t\n";              // empty id
    }

    CleanStats stats;
    const std::vector<Location> pts = parseGowalla(path, stats);
    CHECK(stats.rowsRead == 11);
    CHECK(stats.malformed == 3);
    CHECK(stats.outOfRange == 3);
    CHECK(stats.zeroZero == 1);
    CHECK(stats.duplicates == 1);
    CHECK(stats.kept == 3);
    CHECK(stats.rowsRead == stats.malformed + stats.outOfRange + stats.zeroZero +
                                stats.duplicates + stats.kept);

    CHECK(pts.size() == 3);
    if (pts.size() == 3) {
        CHECK(pts[0].id == 9);  // sorted by id
        CHECK(pts[1].id == 22847);
        CHECK(pts[1].lat == 30.2359);
        CHECK(pts[1].lon == -97.7951);
        CHECK(pts[2].id == 420315);
    }
    std::remove(path.c_str());

    bool threw = false;
    try {
        parseGowalla(tempPath("does_not_exist.txt"), stats);
    } catch (const std::runtime_error&) {
        threw = true;
    }
    CHECK(threw);
}

void testCacheFile() {
    const std::string path = tempPath("cache.bin");

    const std::vector<Location> pts = generateSynthetic(SyntheticKind::Clustered, 1000, 1);
    saveLocations(path, pts);
    const std::vector<Location> loaded = loadLocations(path);
    CHECK(loaded.size() == pts.size());
    for (size_t i = 0; i < std::min(pts.size(), loaded.size()); ++i) {
        CHECK(loaded[i].id == pts[i].id);
        CHECK(loaded[i].lat == pts[i].lat);
        CHECK(loaded[i].lon == pts[i].lon);
    }

    saveLocations(path, {});
    CHECK(loadLocations(path).empty());

    auto rejects = [&](const std::string& contents) {
        std::ofstream(path, std::ios::binary) << contents;
        try {
            loadLocations(path);
        } catch (const std::runtime_error&) {
            return true;
        }
        return false;
    };
    CHECK(rejects(""));
    CHECK(rejects("not a cache file at all, just some text"));

    // Valid header claiming two records, followed by only one.
    std::string truncated = "NBY1";
    truncated.push_back('\x02');
    truncated.append(7, '\0');
    truncated.append(sizeof(Location), 'x');
    CHECK(rejects(truncated));

    std::remove(path.c_str());
}

void testSynthetic() {
    for (SyntheticKind kind : {SyntheticKind::Uniform, SyntheticKind::Clustered}) {
        const std::vector<Location> a = generateSynthetic(kind, 1000, 42);
        const std::vector<Location> b = generateSynthetic(kind, 1000, 42);
        const std::vector<Location> c = generateSynthetic(kind, 1000, 43);

        CHECK(a.size() == 1000);
        bool sameAsB = true;
        bool sameAsC = true;
        for (size_t i = 0; i < a.size(); ++i) {
            sameAsB = sameAsB && a[i].lat == b[i].lat && a[i].lon == b[i].lon;
            sameAsC = sameAsC && a[i].lat == c[i].lat && a[i].lon == c[i].lon;
            CHECK(a[i].id == static_cast<int64_t>(i));
            CHECK(a[i].lat >= -90 && a[i].lat <= 90);
            CHECK(a[i].lon >= -180 && a[i].lon < 180);
        }
        CHECK(sameAsB);
        CHECK(!sameAsC);
    }

    // Clustered data should crowd together far more than uniform data does.
    auto meanNearestKm = [](const std::vector<Location>& pts) {
        BruteForce brute;
        brute.build(pts);
        double total = 0;
        for (const Location& p : pts) total += brute.kNearest(p.lat, p.lon, 2)[1].dist_km;
        return total / static_cast<double>(pts.size());
    };
    CHECK(meanNearestKm(generateSynthetic(SyntheticKind::Clustered, 1000, 42)) <
          meanNearestKm(generateSynthetic(SyntheticKind::Uniform, 1000, 42)) / 3);
}

// BruteForce is the reference for everything else, so it is checked against an
// independent full sort rather than against itself.
void testBruteForce() {
    std::vector<Location> pts = generateSynthetic(SyntheticKind::Clustered, 1000, 3);
    for (int i = 0; i < 50; ++i) pts.push_back({9000 - i, pts[0].lat, pts[0].lon});

    BruteForce brute;
    brute.build(pts);

    auto fullSort = [&](double lat, double lon) {
        std::vector<std::pair<double, int64_t>> all;
        for (const Location& p : pts) all.push_back({haversineKm(lat, lon, p.lat, p.lon), p.id});
        std::sort(all.begin(), all.end());
        return all;
    };

    std::mt19937_64 rng(11);
    for (int trial = 0; trial < 200; ++trial) {
        const Location& anchor = pts[rng() % pts.size()];
        const double lat = trial % 2 == 0 ? anchor.lat : contract_detail::randomIn(rng, -90, 90);
        const double lon = trial % 2 == 0 ? anchor.lon : contract_detail::randomIn(rng, -180, 180);
        const auto expected = fullSort(lat, lon);

        const std::vector<Result> nearest = brute.kNearest(lat, lon, 100);
        CHECK(nearest.size() == 100);
        for (size_t i = 0; i < nearest.size(); ++i) {
            CHECK(nearest[i].dist_km == expected[i].first);
            CHECK(pts[nearest[i].idx].id == expected[i].second);
        }

        const std::vector<Result> inRange = brute.withinRadius(lat, lon, 300.0);
        size_t expectedCount = 0;
        while (expectedCount < expected.size() && expected[expectedCount].first <= 300.0) {
            ++expectedCount;
        }
        CHECK(inRange.size() == expectedCount);
        for (size_t i = 0; i < std::min(inRange.size(), expectedCount); ++i) {
            CHECK(pts[inRange[i].idx].id == expected[i].second);
        }
    }

    CHECK(brute.kNearest(0, 0, 0).empty());
    CHECK(brute.kNearest(0, 0, -1).empty());
    CHECK(brute.kNearest(0, 0, 100000).size() == pts.size());
    CHECK(brute.withinRadius(0, 0, -1.0).empty());
    CHECK(brute.withinRadius(0, 0, 25000.0).size() == pts.size());
    CHECK(brute.withinRadius(pts[0].lat, pts[0].lon, 0.0).size() == 51);  // itself + 50 copies

    const std::vector<Location> none;
    BruteForce emptyIndex;
    CHECK(emptyIndex.kNearest(0, 0, 5).empty());  // never built
    emptyIndex.build(none);
    CHECK(emptyIndex.kNearest(0, 0, 5).empty());
    CHECK(emptyIndex.withinRadius(0, 0, 100.0).empty());
    CHECK(emptyIndex.name() == "BruteForce");
    CHECK(emptyIndex.memoryBytes() > 0);
}

void testCleanStatsFile() {
    const std::string cache = tempPath("stats_cache.bin");

    CleanStats stats;
    stats.rowsRead = 100;
    stats.malformed = 1;
    stats.outOfRange = 2;
    stats.zeroZero = 3;
    stats.duplicates = 4;
    stats.kept = 90;
    saveCleanStats(cache, stats);

    CleanStats loaded;
    CHECK(loadCleanStats(cache, loaded));
    CHECK(loaded.rowsRead == 100);
    CHECK(loaded.malformed == 1);
    CHECK(loaded.outOfRange == 2);
    CHECK(loaded.zeroZero == 3);
    CHECK(loaded.duplicates == 4);
    CHECK(loaded.kept == 90);
    std::remove((cache + ".stats").c_str());

    CHECK(!loadCleanStats(cache, loaded));
}

void testPercentile() {
    std::vector<double> values;
    CHECK(percentile(values, 95) == 0.0);

    values = {5.0};
    CHECK(percentile(values, 95) == 5.0);

    for (int i = 100; i >= 2; --i) values.push_back(i);  // 2..100 plus the 5 above
    values[0] = 1.0;                                     // now exactly 1..100, unsorted
    CHECK(percentile(values, 95) == 95.0);
    CHECK(percentile(values, 50) == 50.0);
    CHECK(percentile(values, 100) == 100.0);
}

std::vector<std::string> readLines(const std::string& path) {
    std::ifstream in(path);
    std::vector<std::string> lines;
    std::string line;
    while (std::getline(in, line)) lines.push_back(line);
    return lines;
}

BenchConfig tinyBenchConfig() {
    BenchConfig config = quickBenchConfig();
    config.sizes = {100, 300};
    config.ks = {1, 10};
    config.radiiKm = {10.0};
    config.queryCount = 30;
    config.minQueries = 5;
    config.buildRepeats = 1;
    return config;
}

void testBenchmarkCsv() {
    const std::string path = tempPath("benchmark.csv");
    const std::vector<Location> gowalla = generateSynthetic(SyntheticKind::Clustered, 500, 5);
    BenchConfig config = tinyBenchConfig();

    const size_t structures = makeAllIndexes().size();
    const size_t perSize = structures * (config.ks.size() + config.radiiKm.size());

    // 3 datasets at 2 sizes.
    std::streambuf* quiet = std::cout.rdbuf(nullptr);
    runBenchmark(gowalla, config, path);
    std::vector<std::string> lines = readLines(path);
    CHECK(lines.size() == 1 + 3 * 2 * perSize);
    if (!lines.empty()) {
        CHECK(lines[0] ==
              "structure,dataset,n,query_type,param,queries,build_ms,mean_us,p95_us,"
              "mean_results,memory_bytes");
    }
    CHECK(lines.size() > 1 && lines[1].rfind("BruteForce,gowalla,100,knn,1,30,", 0) == 0);

    // With no real data only the 2 synthetic datasets run. A size equal to the
    // dataset is kept, and the full-size pass doesn't repeat it.
    config.sizes = {100};
    runBenchmark({}, config, path);
    CHECK(readLines(path).size() == 1 + 2 * perSize);
    config.includeFullSize = true;
    runBenchmark({}, config, path);
    CHECK(readLines(path).size() == 1 + 2 * perSize);

    // The full-size pass adds the dataset's own size when it isn't listed.
    config.sizes = {100, 300};
    runBenchmark(gowalla, config, path);
    CHECK(readLines(path).size() == 1 + 3 * 3 * perSize);

    // The time budget stops a measurement early but keeps the minimum query count.
    config.budgetSeconds = 0.0;
    config.includeFullSize = false;
    config.sizes = {100};
    runBenchmark({}, config, path);
    lines = readLines(path);
    CHECK(lines.size() > 1 && lines[1].rfind("BruteForce,uniform,100,knn,1,5,", 0) == 0);
    std::cout.rdbuf(quiet);

    std::remove(path.c_str());
}

void testTuningCsv() {
    const std::string path = tempPath("tuning.csv");
    const std::vector<Location> pts = generateSynthetic(SyntheticKind::Uniform, 200, 5);

    std::vector<std::pair<int, int>> seen;
    const TunableFactory factory = [&](int bucket, int depth) {
        seen.push_back({bucket, depth});
        return std::make_unique<BruteForce>();
    };

    std::streambuf* quiet = std::cout.rdbuf(nullptr);
    runTuning(pts, factory, tinyBenchConfig(), path);
    std::cout.rdbuf(quiet);

    CHECK(seen.size() == 20);  // 5 bucket sizes x 4 depth limits
    CHECK(seen.front() == std::make_pair(4, 12));
    CHECK(seen.back() == std::make_pair(64, 24));
    CHECK(readLines(path).size() == 21);
    std::remove(path.c_str());
}

void testMismatchDetection() {
    const std::vector<Result> base = {{1, 1.0}, {2, 2.0}};
    CHECK(firstMismatch(base, base).empty());
    CHECK(!firstMismatch(base, {{1, 1.0}}).empty());
    CHECK(!firstMismatch(base, {{1, 1.0}, {3, 2.0}}).empty());
    CHECK(!firstMismatch(base, {{1, 1.0}, {2, 2.0000000001}}).empty());
}

// Checks the quickselect postcondition: idx[nth] holds the sorted value, with
// nothing larger before it and nothing smaller after it.
bool selectedCorrectly(const std::vector<uint32_t>& idx, size_t begin, size_t end, size_t nth,
                       const std::vector<Location>& pts, int axis) {
    auto key = [&](size_t i) { return axis == 0 ? pts[idx[i]].lat : pts[idx[i]].lon; };

    std::vector<double> sorted;
    for (size_t i = begin; i < end; ++i) sorted.push_back(key(i));
    std::sort(sorted.begin(), sorted.end());
    if (key(nth) != sorted[nth - begin]) return false;

    for (size_t i = begin; i < nth; ++i) {
        if (key(i) > key(nth)) return false;
    }
    for (size_t i = nth + 1; i < end; ++i) {
        if (key(i) < key(nth)) return false;
    }
    return true;
}

void testQuickselect() {
    std::mt19937_64 rng(7);
    const std::vector<Location> uniform = generateSynthetic(SyntheticKind::Uniform, 500, 7);

    for (int trial = 0; trial < 200; ++trial) {
        std::vector<uint32_t> idx(uniform.size());
        for (size_t i = 0; i < idx.size(); ++i) idx[i] = static_cast<uint32_t>(i);
        const size_t begin = rng() % 100;
        const size_t end = begin + 1 + rng() % (idx.size() - begin);
        const size_t nth = begin + rng() % (end - begin);
        const int axis = trial % 2;
        quickselect(idx, begin, end, nth, uniform, axis);
        CHECK(selectedCorrectly(idx, begin, end, nth, uniform, axis));

        // The range is permuted, and nothing outside it moves.
        std::vector<uint32_t> sortedIdx = idx;
        std::sort(sortedIdx.begin(), sortedIdx.end());
        for (size_t i = 0; i < sortedIdx.size(); ++i) CHECK(sortedIdx[i] == i);
        for (size_t i = 0; i < begin; ++i) CHECK(idx[i] == i);
        for (size_t i = end; i < idx.size(); ++i) CHECK(idx[i] == i);
    }

    // All keys equal, and already-sorted keys: the inputs that break naive pivots.
    std::vector<Location> same, ascending;
    for (int i = 0; i < 1000; ++i) {
        same.push_back({i, 29.65, -82.32});
        ascending.push_back({i, -90.0 + 0.18 * i, -180.0 + 0.36 * i});
    }
    for (const std::vector<Location>* pts : {&same, &ascending}) {
        for (int axis = 0; axis < 2; ++axis) {
            std::vector<uint32_t> idx(pts->size());
            for (size_t i = 0; i < idx.size(); ++i) idx[i] = static_cast<uint32_t>(i);
            quickselect(idx, 0, idx.size(), idx.size() / 2, *pts, axis);
            CHECK(selectedCorrectly(idx, 0, idx.size(), idx.size() / 2, *pts, axis));
        }
    }
}

void testBoundedHeap() {
    // Ids run backwards from the array order, so ties are broken by id, not idx.
    std::vector<Location> pts;
    for (int i = 0; i < 50; ++i) pts.push_back({100 - i, 0.0, 0.0});

    std::mt19937_64 rng(11);
    for (size_t k : {size_t{0}, size_t{1}, size_t{5}, size_t{50}, size_t{80}}) {
        std::vector<Result> all;
        BoundedMaxHeap heap(k, pts);
        for (uint32_t i = 0; i < pts.size(); ++i) {
            const Result r = {i, static_cast<double>(rng() % 10)};  // plenty of ties
            all.push_back(r);
            heap.offer(r);
            CHECK(heap.size() == std::min(k, static_cast<size_t>(i) + 1));
        }

        std::sort(all.begin(), all.end(), ResultOrder{&pts});
        all.resize(std::min(k, all.size()));
        CHECK(firstMismatch(all, heap.takeSorted()).empty());
        CHECK(heap.size() == 0);
    }
}

void testKdTreeBasics() {
    KdTree tree;
    CHECK(tree.name() == "KdTree");
    CHECK(tree.kNearest(0, 0, 5).empty());  // queried before build

    const std::vector<Location> pts = generateSynthetic(SyntheticKind::Clustered, 10000, 3);
    tree.build(pts);
    CHECK(tree.memoryBytes() > pts.size() * sizeof(uint32_t));

    // Rebuilding on different data must not keep anything from the first build.
    const std::vector<Location> small = {{1, 10.0, 10.0}, {2, 10.0, 10.5}};
    tree.build(small);
    const std::vector<Result> nearest = tree.kNearest(10.0, 10.1, 5);
    CHECK(nearest.size() == 2);
    CHECK(!nearest.empty() && nearest[0].idx == 0);
}

}  // namespace

int main() {
    testHaversine();
    testRectBoundCases();
    testRectBoundRandomized();
    testResultOrder();
    testParseGowalla();
    testCacheFile();
    testSynthetic();
    testBruteForce();
    testMismatchDetection();
    testCleanStatsFile();
    testPercentile();
    testBenchmarkCsv();
    testTuningCsv();
    testQuickselect();
    testBoundedHeap();
    testKdTreeBasics();

    BruteForce brute;
    failures += runIndexContract(brute);

    KdTree kdTree;
    failures += runIndexContract(kdTree);

    // When the quadtree lands, add it here:
    //   QuadTree quadTree; failures += runIndexContract(quadTree);

    if (failures != 0) {
        std::cerr << failures << " check(s) failed\n";
        return 1;
    }
    std::cout << "all tests passed\n";
    return 0;
}
