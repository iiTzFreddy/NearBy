// Tests for QuadTree: the shared contract against BruteForce, plus the cases that
// are specific to a quadtree (bucket size, depth limit, rebuilding).

#include <iostream>
#include <string>
#include <vector>

#include "index_contract.hpp"
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

// The shared randomized and edge-case tests, with the default settings.
void testDefaultContract() {
    QuadTree tree;
    failures += runIndexContract(tree);
}

// Every setting the tuning sweep will try must still give exact answers, and so
// must the extremes: one point per leaf, and a depth limit that forbids splitting.
void testContractAcrossSettings() {
    using contract_detail::checkDataset;

    // The full contract is slow, so it runs only on the extremes.
    const std::vector<std::pair<int, int>> extremes = {
        {4, 12},   // smallest sweep bucket, shallowest sweep depth
        {64, 24},  // largest bucket, deepest
        {1, 24},   // a leaf holds one point unless the depth limit stops it
        {1, 3},    // shallow: most leaves overflow
        {16, 0},   // no splitting at all: the root is the only leaf
        {16, 1},
    };
    for (const auto& [bucket, depth] : extremes) {
        QuadTree tree(bucket, depth);
        const int bad = runIndexContract(tree);
        if (bad != 0) {
            std::cerr << "contract failed with bucket=" << bucket << " depth=" << depth << '\n';
        }
        failures += bad;
    }

    // Every other combination the tuning sweep will run gets a lighter check.
    const std::vector<Location> clustered = generateSynthetic(SyntheticKind::Clustered, 1000, 21);
    for (int bucket : {4, 8, 16, 32, 64}) {
        for (int depth : {12, 16, 20, 24}) {
            QuadTree tree(bucket, depth);
            failures += checkDataset(tree,
                                     "sweep bucket=" + std::to_string(bucket) +
                                         " depth=" + std::to_string(depth),
                                     clustered, 150, 21);
        }
    }
}

// Bigger data and several seeds, so that a pruning mistake that only shows up with
// deeper trees or unlucky ties has a chance to appear.
void testLargerRandomData() {
    using contract_detail::checkDataset;

    for (uint64_t seed : {1, 2, 3}) {
        const std::vector<Location> clustered = generateSynthetic(SyntheticKind::Clustered, 4000, seed);
        const std::vector<Location> uniform = generateSynthetic(SyntheticKind::Uniform, 4000, seed);
        for (int depth : {8, 20}) {
            QuadTree tree(8, depth);
            failures += checkDataset(tree, "clustered 4k depth " + std::to_string(depth), clustered, 100, seed);
            failures += checkDataset(tree, "uniform 4k depth " + std::to_string(depth), uniform, 100, seed);
        }
    }

    // Tiny datasets, where the root is a leaf or barely splits.
    for (size_t n : {size_t{1}, size_t{2}, size_t{3}, size_t{17}, size_t{18}}) {
        const std::vector<Location> pts = generateSynthetic(SyntheticKind::Clustered, n, 9);
        QuadTree tree;
        failures += checkDataset(tree, "tiny n=" + std::to_string(n), pts, 100, 9);
    }
}

// Thousands of identical coordinates can never be separated by splitting. The depth
// limit must stop the recursion, and the oversized leaf must still answer exactly.
void testDepthLimitStopsIdenticalPoints() {
    std::vector<Location> pts;
    for (int i = 0; i < 5000; ++i) pts.push_back({100000 - i, 29.65, -82.32});

    QuadTree tree(4, 20);
    tree.build(pts);

    // One chain of splits: at most 4 new nodes per level for 20 levels, plus the
    // root. Anything bigger means the tree kept splitting.
    CHECK(tree.memoryBytes() < sizeof(QuadTree) + 100 * 16 + pts.size() * sizeof(uint32_t) + 4096);

    BruteForce brute;
    brute.build(pts);
    CHECK(firstMismatch(brute.kNearest(29.65, -82.32, 50), tree.kNearest(29.65, -82.32, 50)).empty());
    CHECK(firstMismatch(brute.withinRadius(29.65, -82.32, 0.0), tree.withinRadius(29.65, -82.32, 0.0)).empty());
    CHECK(tree.withinRadius(29.65, -82.32, 0.0).size() == pts.size());
}

// The tie-break must pick the smaller id even when the tied points sit in a leaf
// the search would otherwise be tempted to skip: query exactly on a duplicate pile.
void testTiesPreferSmallerId() {
    std::vector<Location> pts;
    for (int i = 0; i < 40; ++i) pts.push_back({500 - i, 10.0, 20.0});  // ids 461..500, same spot
    for (int i = 0; i < 40; ++i) pts.push_back({i, 10.0 + 0.001 * (i + 1), 20.0});

    QuadTree tree(4, 24);
    tree.build(pts);

    const std::vector<Result> nearest = tree.kNearest(10.0, 20.0, 5);
    CHECK(nearest.size() == 5);
    for (size_t i = 0; i < nearest.size(); ++i) {
        CHECK(pts[nearest[i].idx].id == 461 + static_cast<int64_t>(i));
    }
}

void testBuildTwiceReplacesTheTree() {
    const std::vector<Location> first = generateSynthetic(SyntheticKind::Uniform, 500, 1);
    const std::vector<Location> second = generateSynthetic(SyntheticKind::Clustered, 800, 2);

    QuadTree tree;
    BruteForce brute;

    tree.build(first);
    tree.build(second);  // must forget `first` completely
    brute.build(second);

    CHECK(firstMismatch(brute.kNearest(30.0, -90.0, 25), tree.kNearest(30.0, -90.0, 25)).empty());
    CHECK(firstMismatch(brute.withinRadius(30.0, -90.0, 800.0), tree.withinRadius(30.0, -90.0, 800.0)).empty());

    const std::vector<Location> none;
    tree.build(none);
    CHECK(tree.kNearest(30.0, -90.0, 5).empty());
    CHECK(tree.withinRadius(30.0, -90.0, 1e6).empty());
}

void testArgumentsAndBasics() {
    QuadTree neverBuilt;
    CHECK(neverBuilt.kNearest(0, 0, 5).empty());
    CHECK(neverBuilt.withinRadius(0, 0, 100.0).empty());

    const std::vector<Location> pts = generateSynthetic(SyntheticKind::Uniform, 300, 5);
    QuadTree tree;
    tree.build(pts);
    CHECK(tree.name() == "QuadTree");
    CHECK(tree.kNearest(0, 0, 0).empty());
    CHECK(tree.kNearest(0, 0, -3).empty());
    CHECK(tree.withinRadius(0, 0, -1.0).empty());
    CHECK(tree.kNearest(0, 0, 100000).size() == pts.size());
    CHECK(tree.withinRadius(0, 0, 25000.0).size() == pts.size());

    // Smaller buckets mean more nodes, so more memory for the same points.
    QuadTree coarse(64, 20);
    QuadTree fine(4, 20);
    const std::vector<Location> many = generateSynthetic(SyntheticKind::Uniform, 20000, 6);
    coarse.build(many);
    fine.build(many);
    CHECK(coarse.memoryBytes() > many.size() * sizeof(uint32_t));
    CHECK(fine.memoryBytes() > coarse.memoryBytes());
}

}  // namespace

int main() {
    testDefaultContract();
    testContractAcrossSettings();
    testLargerRandomData();
    testDepthLimitStopsIdenticalPoints();
    testTiesPreferSmallerId();
    testBuildTwiceReplacesTheTree();
    testArgumentsAndBasics();

    if (failures != 0) {
        std::cerr << failures << " quadtree check(s) failed\n";
        return 1;
    }
    std::cout << "all quadtree tests passed\n";
    return 0;
}
