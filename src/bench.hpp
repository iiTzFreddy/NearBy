#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "spatial_index.hpp"

// Every structure the CLI and benchmark know about. New structures are added in
// this one function.
std::vector<std::unique_ptr<SpatialIndex>> makeAllIndexes();

struct BenchConfig {
    std::vector<size_t> sizes;  // dataset sizes; any larger than the dataset are skipped
    bool includeFullSize;       // also run at the dataset's full size
    std::vector<int> ks;
    std::vector<double> radiiKm;
    size_t queryCount;          // queries per measurement, before the time budget
    size_t minQueries;          // always run at least this many
    double budgetSeconds;       // stop a measurement early after this long
    int buildRepeats;           // build time is the median of this many builds
    uint64_t seed;
};

BenchConfig fullBenchConfig();   // the plan's sizes and parameters
BenchConfig quickBenchConfig();  // smaller sizes, for checking things work

// Value at the given percentile (0-100) by the nearest-rank method. Sorts `values`.
double percentile(std::vector<double>& values, double pct);

// Benchmarks every structure from makeAllIndexes on uniform and clustered data,
// and on random samples of `gowalla` unless it is empty. Writes one CSV row per
// (structure, dataset, n, query type, parameter). A slow structure can stop short
// of queryCount; the `queries` column records how many actually ran.
void runBenchmark(const std::vector<Location>& gowalla, const BenchConfig& config,
                  const std::string& csvPath);

// Makes a quadtree with the given bucket size and depth limit.
using TunableFactory = std::function<std::unique_ptr<SpatialIndex>(int bucket, int depth)>;

// Empty until QuadTree exists. Once it returns a factory, the benchmark runs the
// tuning sweep automatically.
TunableFactory quadTreeFactory();

// Sweeps the plan's bucket sizes and depth limits on `pts` and writes one CSV row
// per combination.
void runTuning(const std::vector<Location>& pts, const TunableFactory& factory,
               const BenchConfig& config, const std::string& csvPath);
