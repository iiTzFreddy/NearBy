// NearBy: run with no arguments for the interactive menu, or with a subcommand to
// prepare datasets and run the benchmark from a script.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <functional>
#include <iomanip>
#include <iostream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "bench.hpp"
#include "loader.hpp"

namespace {

using Clock = std::chrono::steady_clock;

const char* const kDefaultDataset = "data/gowalla_clean.bin";
const char* const kBenchmarkCsv = "results/benchmark.csv";
const char* const kTuningCsv = "results/tuning.csv";
constexpr size_t kTuningSampleSize = 100000;
constexpr size_t kMaxRowsShown = 10;

double millisecondsSince(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

bool endsWith(const std::string& text, const std::string& suffix) {
    return text.size() >= suffix.size() &&
           text.compare(text.size() - suffix.size(), suffix.size(), suffix) == 0;
}

void printCleanStats(const CleanStats& stats) {
    std::cout << "rows read:    " << stats.rowsRead << '\n'
              << "malformed:    " << stats.malformed << '\n'
              << "out of range: " << stats.outOfRange << '\n'
              << "at (0, 0):    " << stats.zeroZero << '\n'
              << "duplicates:   " << stats.duplicates << '\n'
              << "kept:         " << stats.kept << '\n';
}

void benchmark(const std::vector<Location>& gowalla, const BenchConfig& config) {
    std::cout << "Running benchmark...\n";
    runBenchmark(gowalla, config, kBenchmarkCsv);
    std::cout << "Wrote " << kBenchmarkCsv << '\n';

    if (const TunableFactory factory = quadTreeFactory()) {
        const std::vector<Location> sample =
            gowalla.empty()
                ? generateSynthetic(SyntheticKind::Clustered, kTuningSampleSize, config.seed)
                : gowalla;
        runTuning(sample, factory, config, kTuningCsv);
        std::cout << "Wrote " << kTuningCsv << '\n';
    }
    std::cout << "Make the graphs with: python scripts/plot.py\n";
}

// ---- Interactive menu ----

struct App {
    std::vector<Location> points;
    std::vector<std::unique_ptr<SpatialIndex>> indexes;

    bool loaded() const { return !indexes.empty(); }
};

// Every prompt returns nullopt at end of input, which exits the menu.
std::optional<std::string> readLine(const std::string& prompt) {
    std::cout << prompt << std::flush;
    std::string line;
    if (!std::getline(std::cin, line)) return std::nullopt;
    if (!line.empty() && line.back() == '\r') line.pop_back();
    return line;
}

std::optional<double> promptNumber(const std::string& prompt, double low, double high) {
    while (true) {
        const std::optional<std::string> line = readLine(prompt);
        if (!line) return std::nullopt;

        char* end = nullptr;
        const double value = std::strtod(line->c_str(), &end);
        const bool parsed = end != line->c_str() && *end == '\0' && std::isfinite(value);
        if (parsed && value >= low && value <= high) return value;
        std::cout << std::setprecision(10) << "Enter a number from " << low << " to " << high
                  << ".\n";
    }
}

std::optional<int> promptInt(const std::string& prompt, int low, int high) {
    while (true) {
        const std::optional<double> value = promptNumber(prompt, low, high);
        if (!value) return std::nullopt;
        if (*value == std::floor(*value)) return static_cast<int>(*value);
        std::cout << "Enter a whole number.\n";
    }
}

void loadDataset(App& app) {
    const std::optional<std::string> entered =
        readLine("Dataset path [" + std::string(kDefaultDataset) + "]: ");
    if (!entered) return;
    const std::string path = entered->empty() ? kDefaultDataset : *entered;

    std::vector<Location> points;
    CleanStats stats;
    bool haveStats = false;
    try {
        const Clock::time_point start = Clock::now();
        if (endsWith(path, ".bin")) {
            points = loadLocations(path);
            haveStats = loadCleanStats(path, stats);
        } else {
            points = parseGowalla(path, stats);
            haveStats = true;
        }
        std::cout << "Loaded " << points.size() << " locations in " << std::fixed
                  << std::setprecision(1) << millisecondsSince(start) << " ms\n"
                  << std::defaultfloat;
    } catch (const std::exception& error) {
        std::cout << "Could not load the dataset: " << error.what() << '\n';
        return;
    }
    if (haveStats) printCleanStats(stats);

    app.points = std::move(points);
    app.indexes = makeAllIndexes();
    for (const std::unique_ptr<SpatialIndex>& index : app.indexes) {
        const Clock::time_point start = Clock::now();
        index->build(app.points);
        std::cout << "Built " << std::left << std::setw(12) << index->name() << std::right
                  << std::fixed << std::setprecision(1) << std::setw(10) << millisecondsSince(start)
                  << " ms, " << std::setprecision(2)
                  << static_cast<double>(index->memoryBytes()) / (1024.0 * 1024.0) << " MB\n"
                  << std::defaultfloat;
    }
}

// Returns the chosen structures, or nothing at end of input.
std::vector<const SpatialIndex*> chooseStructures(const App& app) {
    std::vector<const SpatialIndex*> all;
    for (const std::unique_ptr<SpatialIndex>& index : app.indexes) all.push_back(index.get());
    if (all.size() == 1) return all;

    for (size_t i = 0; i < all.size(); ++i) {
        std::cout << "  " << i + 1 << ". " << all[i]->name() << '\n';
    }
    const int allChoice = static_cast<int>(all.size()) + 1;
    std::cout << "  " << allChoice << ". All of them\n";

    const std::optional<int> choice = promptInt("Structure: ", 1, allChoice);
    if (!choice) return {};
    if (*choice == allChoice) return all;
    return {all[static_cast<size_t>(*choice) - 1]};
}

bool sameResults(const std::vector<Result>& a, const std::vector<Result>& b) {
    return std::equal(a.begin(), a.end(), b.begin(), b.end(), [](const Result& x, const Result& y) {
        return x.idx == y.idx && x.dist_km == y.dist_km;
    });
}

void printResults(const App& app, const std::vector<Result>& results) {
    if (results.empty()) {
        std::cout << "No places found.\n";
        return;
    }

    std::cout << std::fixed << '\n'
              << std::setw(5) << "#" << std::setw(12) << "id" << std::setw(12) << "lat"
              << std::setw(13) << "lon" << std::setw(14) << "distance km" << '\n';
    const size_t shown = std::min(results.size(), kMaxRowsShown);
    for (size_t i = 0; i < shown; ++i) {
        const Location& place = app.points[results[i].idx];
        std::cout << std::setw(5) << i + 1 << std::setw(12) << place.id << std::setprecision(5)
                  << std::setw(12) << place.lat << std::setw(13) << place.lon
                  << std::setprecision(3) << std::setw(14) << results[i].dist_km << '\n';
    }
    if (results.size() > shown) {
        std::cout << "  ... and " << results.size() - shown << " more\n";
    }
    std::cout << std::defaultfloat;
}

using Search = std::function<std::vector<Result>(const SpatialIndex&)>;

void runSearch(const App& app, const std::vector<const SpatialIndex*>& structures,
               const Search& search) {
    std::vector<std::vector<Result>> outputs;
    std::vector<double> times;
    for (const SpatialIndex* index : structures) {
        const Clock::time_point start = Clock::now();
        outputs.push_back(search(*index));
        times.push_back(millisecondsSince(start));
    }

    printResults(app, outputs.front());

    std::cout << '\n' << outputs.front().size() << " result(s)\n";
    for (size_t i = 0; i < structures.size(); ++i) {
        std::cout << std::left << std::setw(12) << structures[i]->name() << std::right << std::fixed
                  << std::setprecision(3) << std::setw(12) << times[i] << " ms\n"
                  << std::defaultfloat;
    }
    if (structures.size() > 1) {
        bool identical = true;
        for (const std::vector<Result>& output : outputs) {
            identical = identical && sameResults(outputs.front(), output);
        }
        std::cout << (identical ? "All structures returned identical results.\n"
                                : "WARNING: the structures returned different results.\n");
    }
}

bool promptLocation(double& lat, double& lon) {
    const std::optional<double> enteredLat = promptNumber("Latitude (-90 to 90): ", -90.0, 90.0);
    if (!enteredLat) return false;
    const std::optional<double> enteredLon =
        promptNumber("Longitude (-180 to 180): ", -180.0, 180.0);
    if (!enteredLon) return false;
    lat = *enteredLat;
    lon = *enteredLon;
    return true;
}

void findNearest(const App& app) {
    double lat = 0.0;
    double lon = 0.0;
    if (!promptLocation(lat, lon)) return;

    const int maxK = static_cast<int>(std::min<size_t>(app.points.size(), 1000000));
    if (maxK == 0) {
        std::cout << "The dataset is empty.\n";
        return;
    }
    const std::optional<int> k =
        promptInt("How many places (k, 1 to " + std::to_string(maxK) + "): ", 1, maxK);
    if (!k) return;

    const std::vector<const SpatialIndex*> structures = chooseStructures(app);
    if (structures.empty()) return;
    runSearch(app, structures,
              [&](const SpatialIndex& index) { return index.kNearest(lat, lon, *k); });
}

void findWithinRadius(const App& app) {
    double lat = 0.0;
    double lon = 0.0;
    if (!promptLocation(lat, lon)) return;

    const std::optional<double> radius = promptNumber("Radius in km: ", 0.0, 21000.0);
    if (!radius) return;

    const std::vector<const SpatialIndex*> structures = chooseStructures(app);
    if (structures.empty()) return;
    runSearch(app, structures,
              [&](const SpatialIndex& index) { return index.withinRadius(lat, lon, *radius); });
}

void runBenchmarkFromMenu(const App& app) {
    std::cout << "  1. Quick check (up to 100k points, about a minute)\n"
                 "  2. Full run (every size, several minutes or more)\n";
    const std::optional<int> choice = promptInt("Benchmark: ", 1, 2);
    if (!choice) return;
    if (app.points.empty()) {
        std::cout << "No dataset loaded, so this uses synthetic data only.\n";
    }
    benchmark(app.points, *choice == 1 ? quickBenchConfig() : fullBenchConfig());
}

int runMenu() {
    App app;
    while (true) {
        std::cout << "\nNearBy\n"
                     "  1. Load dataset\n"
                     "  2. Find the k nearest places\n"
                     "  3. Find places within a radius\n"
                     "  4. Run benchmark\n"
                     "  5. Exit\n";
        const std::optional<int> choice = promptInt("Choice: ", 1, 5);
        if (!choice || *choice == 5) return 0;

        if (*choice == 1) {
            loadDataset(app);
        } else if (*choice == 4) {
            runBenchmarkFromMenu(app);
        } else if (!app.loaded()) {
            std::cout << "Load a dataset first (option 1).\n";
        } else if (*choice == 2) {
            findNearest(app);
        } else {
            findWithinRadius(app);
        }

        if (!std::cin) return 0;
    }
}

// ---- Subcommands ----

int usage() {
    std::cerr << "Usage:\n"
                 "  nearby                                  interactive menu\n"
                 "  nearby clean <raw_gowalla.txt> <out.bin>\n"
                 "  nearby generate <uniform|clustered> <n> <seed> <out.bin>\n"
                 "  nearby bench [--quick] [data.bin]\n";
    return 2;
}

int runClean(const std::string& rawPath, const std::string& outPath) {
    CleanStats stats;
    const std::vector<Location> pts = parseGowalla(rawPath, stats);
    saveLocations(outPath, pts);
    saveCleanStats(outPath, stats);

    printCleanStats(stats);
    std::cout << "wrote " << outPath << '\n';
    return 0;
}

int runGenerate(const std::string& kindName, const std::string& n, const std::string& seed,
                const std::string& outPath) {
    SyntheticKind kind;
    if (kindName == "uniform") {
        kind = SyntheticKind::Uniform;
    } else if (kindName == "clustered") {
        kind = SyntheticKind::Clustered;
    } else {
        return usage();
    }

    const std::vector<Location> pts =
        generateSynthetic(kind, std::stoull(n), std::stoull(seed));
    saveLocations(outPath, pts);
    std::cout << "wrote " << pts.size() << " " << kindName << " points to " << outPath << '\n';
    return 0;
}

int runBench(const std::vector<std::string>& args) {
    bool quick = false;
    std::string dataPath = kDefaultDataset;
    for (const std::string& arg : args) {
        if (arg == "--quick") {
            quick = true;
        } else {
            dataPath = arg;
        }
    }

    benchmark(loadLocations(dataPath), quick ? quickBenchConfig() : fullBenchConfig());
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    const std::vector<std::string> args(argv + 1, argv + argc);

    try {
        if (args.empty()) return runMenu();
        if (args.size() == 3 && args[0] == "clean") return runClean(args[1], args[2]);
        if (args.size() == 5 && args[0] == "generate") {
            return runGenerate(args[1], args[2], args[3], args[4]);
        }
        if (args[0] == "bench" && args.size() <= 3) {
            return runBench(std::vector<std::string>(args.begin() + 1, args.end()));
        }
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
    return usage();
}
