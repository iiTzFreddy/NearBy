// Tests for KdTree beyond the shared contract in test_main.cpp: more seeds and
// sizes, the inputs that break a naive quickselect, rebuilding, and bad arguments.

#include <algorithm>
#include <chrono>
#include <iostream>
#include <string>
#include <vector>

#include "index_contract.hpp"
#include "kdtree.hpp"

namespace {

int failures = 0;

#define CHECK(condition)                                                                  \
    do {                                                                                  \
        if (!(condition)) {                                                               \
            ++failures;                                                                   \
            std::cerr << "FAILED " << __FILE__ << ":" << __LINE__ << ": " #condition "\n"; \
        }                                                                                 \
    } while (false)

double secondsToBuild(KdTree& tree, const std::vector<Location>& pts) {
    const auto start = std::chrono::steady_clock::now();
    tree.build(pts);
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
}

// Several seeds on bigger data, so a pruning mistake that needs a deeper tree or an
// unlucky tie has a chance to show up.
void testMoreSeedsAndSizes() {
    using contract_detail::checkDataset;

    for (uint64_t seed : {1, 2, 3}) {
        const std::vector<Location> clustered = generateSynthetic(SyntheticKind::Clustered, 4000, seed);
        const std::vector<Location> uniform = generateSynthetic(SyntheticKind::Uniform, 4000, seed);
        KdTree tree;
        failures += checkDataset(tree, "clustered 4k", clustered, 100, seed);
        failures += checkDataset(tree, "uniform 4k", uniform, 100, seed);
    }

    // Around the leaf size of 8: a single leaf, exactly one full leaf, the first
    // split, and a split into uneven halves.
    for (size_t n : {size_t{1}, size_t{2}, size_t{3}, size_t{8}, size_t{9}, size_t{17}}) {
        const std::vector<Location> pts = generateSynthetic(SyntheticKind::Clustered, n, 9);
        KdTree tree;
        failures += checkDataset(tree, "tiny n=" + std::to_string(n), pts, 100, 9);
    }
}

// Identical coordinates are the classic way to make quickselect quadratic: every
// key equals the pivot. The three-way partition must keep the build fast and the
// answers exact.
void testIdenticalPoints() {
    std::vector<Location> pts;
    for (int i = 0; i < 5000; ++i) pts.push_back({100000 - i, 29.65, -82.32});

    KdTree tree;
    tree.build(pts);
    BruteForce brute;
    brute.build(pts);
    CHECK(firstMismatch(brute.kNearest(29.65, -82.32, 50), tree.kNearest(29.65, -82.32, 50)).empty());
    CHECK(firstMismatch(brute.kNearest(30.0, -82.0, 7), tree.kNearest(30.0, -82.0, 7)).empty());
    CHECK(firstMismatch(brute.withinRadius(29.65, -82.32, 0.0), tree.withinRadius(29.65, -82.32, 0.0)).empty());
    CHECK(tree.withinRadius(29.65, -82.32, 0.0).size() == pts.size());

    // A quadratic build on 200k identical points would take minutes; a linear one
    // per level takes milliseconds. The limit leaves a wide margin for slow machines.
    std::vector<Location> many;
    for (int i = 0; i < 200000; ++i) many.push_back({i, 29.65, -82.32});
    KdTree big;
    const double seconds = secondsToBuild(big, many);
    CHECK(seconds < 5.0);
    if (seconds >= 5.0) std::cerr << "identical-points build took " << seconds << " s\n";
}

// Already sorted input (ascending and descending on both axes at once) breaks a
// first-element pivot. Median-of-three must keep it fast and exact.
void testSortedInput() {
    using contract_detail::checkDataset;

    std::vector<Location> ascending;
    for (int i = 0; i < 3000; ++i) {
        ascending.push_back({i, -89.0 + 0.059 * i, -179.0 + 0.119 * i});
    }
    std::vector<Location> descending(ascending.rbegin(), ascending.rend());

    KdTree tree;
    failures += checkDataset(tree, "sorted ascending", ascending, 100, 4);
    failures += checkDataset(tree, "sorted descending", descending, 100, 4);

    std::vector<Location> big;
    for (int i = 0; i < 200000; ++i) big.push_back({i, -89.0 + 0.0009 * i, -179.0 + 0.0018 * i});
    KdTree fast;
    const double seconds = secondsToBuild(fast, big);
    CHECK(seconds < 5.0);
    if (seconds >= 5.0) std::cerr << "sorted-input build took " << seconds << " s\n";
}

// The tie-break must pick the smaller id even when the query sits exactly on a
// pile of duplicates, where every bound and the k-th distance are all 0.
void testTiesPreferSmallerId() {
    std::vector<Location> pts;
    for (int i = 0; i < 40; ++i) pts.push_back({500 - i, 10.0, 20.0});  // ids 461..500, same spot
    for (int i = 0; i < 40; ++i) pts.push_back({i, 10.0 + 0.001 * (i + 1), 20.0});

    KdTree tree;
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

    KdTree tree;
    BruteForce brute;

    tree.build(first);
    tree.build(second);  // must forget `first` completely
    brute.build(second);

    CHECK(firstMismatch(brute.kNearest(30.0, -90.0, 25), tree.kNearest(30.0, -90.0, 25)).empty());
    CHECK(firstMismatch(brute.withinRadius(30.0, -90.0, 800.0), tree.withinRadius(30.0, -90.0, 800.0)).empty());

    // Going from more points to fewer must not leave old nodes behind.
    tree.build(first);
    brute.build(first);
    CHECK(firstMismatch(brute.kNearest(-10.0, 45.0, 30), tree.kNearest(-10.0, 45.0, 30)).empty());

    const std::vector<Location> none;
    tree.build(none);
    CHECK(tree.kNearest(30.0, -90.0, 5).empty());
    CHECK(tree.withinRadius(30.0, -90.0, 1e6).empty());
}

void testArgumentsAndBasics() {
    KdTree neverBuilt;
    CHECK(neverBuilt.name() == "KdTree");
    CHECK(neverBuilt.kNearest(0, 0, 5).empty());
    CHECK(neverBuilt.withinRadius(0, 0, 100.0).empty());

    const std::vector<Location> pts = generateSynthetic(SyntheticKind::Uniform, 300, 5);
    KdTree tree;
    tree.build(pts);
    CHECK(tree.kNearest(0, 0, 0).empty());
    CHECK(tree.kNearest(0, 0, -3).empty());
    CHECK(tree.withinRadius(0, 0, -1.0).empty());
    CHECK(tree.withinRadius(0, 0, -1e-12).empty());
    CHECK(tree.kNearest(0, 0, 100000).size() == pts.size());
    CHECK(tree.kNearest(0, 0, 2147483647).size() == pts.size());
    CHECK(tree.withinRadius(0, 0, 25000.0).size() == pts.size());
}

void testMemoryBytes() {
    KdTree empty;
    const std::vector<Location> none;
    empty.build(none);
    const size_t emptyBytes = empty.memoryBytes();
    CHECK(emptyBytes >= sizeof(KdTree));

    const std::vector<Location> small = generateSynthetic(SyntheticKind::Uniform, 1000, 6);
    const std::vector<Location> large = generateSynthetic(SyntheticKind::Uniform, 20000, 6);
    KdTree smallTree;
    KdTree largeTree;
    smallTree.build(small);
    largeTree.build(large);

    // At least the index array, and growing with n.
    CHECK(smallTree.memoryBytes() > small.size() * sizeof(uint32_t));
    CHECK(largeTree.memoryBytes() > smallTree.memoryBytes());
    CHECK(largeTree.memoryBytes() > emptyBytes);

    // Leaves of up to 8 points keep it well under 64 bytes per point.
    CHECK(largeTree.memoryBytes() < large.size() * 64);
}

}  // namespace

int main() {
    testMoreSeedsAndSizes();
    testIdenticalPoints();
    testSortedInput();
    testTiesPreferSmallerId();
    testBuildTwiceReplacesTheTree();
    testArgumentsAndBasics();
    testMemoryBytes();

    if (failures != 0) {
        std::cerr << failures << " k-d tree check(s) failed\n";
        return 1;
    }
    std::cout << "all k-d tree tests passed\n";
    return 0;
}
