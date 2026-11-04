#include "bench.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <utility>

#include "brute.hpp"
#include "kdtree.hpp"
#include "loader.hpp"
#include "quadtree.hpp"

namespace {

using Clock = std::chrono::steady_clock;

struct Query {
    double lat;
    double lon;
};

struct Timing {
    double meanUs = 0.0;
    double p95Us = 0.0;
    double meanResults = 0.0;
    size_t queries = 0;
};

double elapsedSeconds(Clock::time_point start) {
    return std::chrono::duration<double>(Clock::now() - start).count();
}

double randomUnit(std::mt19937_64& rng) {
    return static_cast<double>(rng() >> 11) / 9007199254740992.0;  // 2^53
}

// Alternates real data points ("near me") with points spread over the data's
// bounding box (harder cases), so any prefix of the list keeps the same mix.
std::vector<Query> makeQueries(const std::vector<Location>& pts, size_t count, uint64_t seed) {
    std::vector<Query> queries;
    if (pts.empty()) return queries;

    double minLat = 90.0;
    double maxLat = -90.0;
    double minLon = 180.0;
    double maxLon = -180.0;
    for (const Location& p : pts) {
        minLat = std::min(minLat, p.lat);
        maxLat = std::max(maxLat, p.lat);
        minLon = std::min(minLon, p.lon);
        maxLon = std::max(maxLon, p.lon);
    }

    std::mt19937_64 rng(seed);
    queries.reserve(count);
    for (size_t i = 0; i < count; ++i) {
        if (i % 2 == 0) {
            const Location& p = pts[rng() % pts.size()];
            queries.push_back({p.lat, p.lon});
        } else {
            const double lat = minLat + (maxLat - minLat) * randomUnit(rng);
            const double lon = minLon + (maxLon - minLon) * randomUnit(rng);
            queries.push_back({lat, lon});
        }
    }
    return queries;
}

// `search` runs one query and returns how many results it found. A short warm-up
// comes first, using few queries so that slow structures don't pay for a full pass.
template <typename Search>
Timing timeQueries(const std::vector<Query>& queries, const BenchConfig& config, Search search) {
    constexpr size_t kWarmupQueries = 10;
    for (size_t i = 0; i < std::min(kWarmupQueries, queries.size()); ++i) search(queries[i]);

    std::vector<double> times;
    times.reserve(queries.size());
    double totalResults = 0.0;

    const Clock::time_point start = Clock::now();
    for (const Query& q : queries) {
        const Clock::time_point queryStart = Clock::now();
        const size_t found = search(q);
        times.push_back(elapsedSeconds(queryStart) * 1e6);
        totalResults += static_cast<double>(found);

        if (times.size() >= config.minQueries && elapsedSeconds(start) > config.budgetSeconds) {
            break;
        }
    }

    Timing timing;
    timing.queries = times.size();
    if (times.empty()) return timing;

    double total = 0.0;
    for (double t : times) total += t;
    timing.meanUs = total / static_cast<double>(times.size());
    timing.meanResults = totalResults / static_cast<double>(times.size());
    timing.p95Us = percentile(times, 95.0);
    return timing;
}

double medianBuildMs(SpatialIndex& index, const std::vector<Location>& pts, int repeats) {
    std::vector<double> times;
    for (int i = 0; i < repeats; ++i) {
        const Clock::time_point start = Clock::now();
        index.build(pts);
        times.push_back(elapsedSeconds(start) * 1e3);
    }
    return percentile(times, 50.0);
}

std::ofstream openCsv(const std::string& path, const std::string& header) {
    const std::filesystem::path parent = std::filesystem::path(path).parent_path();
    if (!parent.empty()) std::filesystem::create_directories(parent);

    std::ofstream out(path);
    if (!out) throw std::runtime_error("cannot write " + path);
    out.precision(10);
    out << header << '\n';
    return out;
}

void benchmarkDataset(const std::string& dataset, const std::vector<Location>& source,
                      const BenchConfig& config, std::ofstream& csv) {
    std::vector<size_t> sizes;
    for (size_t n : config.sizes) {
        if (n <= source.size()) sizes.push_back(n);
    }
    if (config.includeFullSize && (sizes.empty() || sizes.back() != source.size())) {
        sizes.push_back(source.size());
    }

    for (size_t n : sizes) {
        // `source` is pre-shuffled, so a prefix is a random sample.
        const std::vector<Location> pts(source.begin(),
                                        source.begin() + static_cast<std::ptrdiff_t>(n));
        const std::vector<Query> queries = makeQueries(pts, config.queryCount, config.seed);

        for (const std::unique_ptr<SpatialIndex>& index : makeAllIndexes()) {
            std::cout << "  " << dataset << " n=" << n << " " << index->name() << std::endl;

            const double buildMs = medianBuildMs(*index, pts, config.buildRepeats);
            const size_t memory = index->memoryBytes();

            auto writeRow = [&](const char* type, double param, const Timing& t) {
                csv << index->name() << ',' << dataset << ',' << n << ',' << type << ',' << param
                    << ',' << t.queries << ',' << buildMs << ',' << t.meanUs << ',' << t.p95Us
                    << ',' << t.meanResults << ',' << memory << '\n';
            };

            for (int k : config.ks) {
                const Timing timing = timeQueries(queries, config, [&](const Query& q) {
                    return index->kNearest(q.lat, q.lon, k).size();
                });
                writeRow("knn", k, timing);
            }
            for (double r : config.radiiKm) {
                const Timing timing = timeQueries(queries, config, [&](const Query& q) {
                    return index->withinRadius(q.lat, q.lon, r).size();
                });
                writeRow("radius", r, timing);
            }
            csv.flush();  // an interrupted run keeps what it finished
        }
    }
}

std::vector<Location> shuffled(std::vector<Location> pts, uint64_t seed) {
    std::mt19937_64 rng(seed);
    for (size_t i = pts.size(); i > 1; --i) {
        std::swap(pts[i - 1], pts[rng() % i]);
    }
    return pts;
}

}  // namespace

std::vector<std::unique_ptr<SpatialIndex>> makeAllIndexes() {
    std::vector<std::unique_ptr<SpatialIndex>> indexes;
    indexes.push_back(std::make_unique<BruteForce>());
    indexes.push_back(std::make_unique<QuadTree>());
    indexes.push_back(std::make_unique<KdTree>());
    return indexes;
}

BenchConfig fullBenchConfig() {
    BenchConfig config;
    config.sizes = {10000, 50000, 100000, 250000, 500000, 1000000};
    config.includeFullSize = true;
    config.ks = {1, 10, 100};
    config.radiiKm = {1.0, 10.0, 50.0};
    config.queryCount = 1000;
    config.minQueries = 20;
    config.budgetSeconds = 1.0;
    config.buildRepeats = 5;
    config.seed = 42;
    return config;
}

BenchConfig quickBenchConfig() {
    BenchConfig config = fullBenchConfig();
    config.sizes = {10000, 50000, 100000};
    config.includeFullSize = false;
    config.queryCount = 200;
    config.budgetSeconds = 0.2;
    config.buildRepeats = 3;
    return config;
}

double percentile(std::vector<double>& values, double pct) {
    if (values.empty()) return 0.0;
    std::sort(values.begin(), values.end());
    const double rank = std::ceil(pct / 100.0 * static_cast<double>(values.size()));
    const size_t index = rank < 1.0 ? 0 : static_cast<size_t>(rank) - 1;
    return values[std::min(index, values.size() - 1)];
}

void runBenchmark(const std::vector<Location>& gowalla, const BenchConfig& config,
                  const std::string& csvPath) {
    std::ofstream csv = openCsv(csvPath,
                                "structure,dataset,n,query_type,param,queries,build_ms,"
                                "mean_us,p95_us,mean_results,memory_bytes");

    // Synthetic data matches the real dataset's size so the curves line up.
    size_t syntheticSize = gowalla.size();
    for (size_t n : config.sizes) syntheticSize = std::max(syntheticSize, n);

    if (!gowalla.empty()) {
        benchmarkDataset("gowalla", shuffled(gowalla, config.seed), config, csv);
    }
    benchmarkDataset("uniform", generateSynthetic(SyntheticKind::Uniform, syntheticSize, config.seed),
                     config, csv);
    benchmarkDataset("clustered",
                     shuffled(generateSynthetic(SyntheticKind::Clustered, syntheticSize, config.seed),
                              config.seed),
                     config, csv);

    if (!csv) throw std::runtime_error("failed while writing " + csvPath);
}

void runTuning(const std::vector<Location>& pts, const TunableFactory& factory,
               const BenchConfig& config, const std::string& csvPath) {
    constexpr int kTuningK = 10;
    constexpr double kTuningRadiusKm = 10.0;

    std::ofstream csv =
        openCsv(csvPath, "bucket,depth,build_ms,knn_mean_us,radius_mean_us,memory_bytes");
    const std::vector<Query> queries = makeQueries(pts, config.queryCount, config.seed);

    for (int bucket : {4, 8, 16, 32, 64}) {
        for (int depth : {12, 16, 20, 24}) {
            std::cout << "  tuning bucket=" << bucket << " depth=" << depth << std::endl;

            const std::unique_ptr<SpatialIndex> index = factory(bucket, depth);
            const double buildMs = medianBuildMs(*index, pts, config.buildRepeats);
            const Timing knn = timeQueries(queries, config, [&](const Query& q) {
                return index->kNearest(q.lat, q.lon, kTuningK).size();
            });
            const Timing radius = timeQueries(queries, config, [&](const Query& q) {
                return index->withinRadius(q.lat, q.lon, kTuningRadiusKm).size();
            });

            csv << bucket << ',' << depth << ',' << buildMs << ',' << knn.meanUs << ','
                << radius.meanUs << ',' << index->memoryBytes() << '\n';
        }
    }
    if (!csv) throw std::runtime_error("failed while writing " + csvPath);
}

TunableFactory quadTreeFactory() {
    return [](int bucket, int depth) { return std::make_unique<QuadTree>(bucket, depth); };
}
