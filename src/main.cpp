// Week 1 command-line tools for preparing datasets. The interactive menu from the
// project plan replaces this in week 3.

#include <cstdint>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "loader.hpp"

namespace {

int usage() {
    std::cerr << "Usage:\n"
                 "  nearby clean <raw_gowalla.txt> <out.bin>\n"
                 "  nearby generate <uniform|clustered> <n> <seed> <out.bin>\n";
    return 2;
}

int runClean(const std::string& rawPath, const std::string& outPath) {
    CleanStats stats;
    const std::vector<Location> pts = parseGowalla(rawPath, stats);
    saveLocations(outPath, pts);

    std::cout << "rows read:    " << stats.rowsRead << '\n'
              << "malformed:    " << stats.malformed << '\n'
              << "out of range: " << stats.outOfRange << '\n'
              << "at (0, 0):    " << stats.zeroZero << '\n'
              << "duplicates:   " << stats.duplicates << '\n'
              << "kept:         " << stats.kept << '\n'
              << "wrote " << outPath << '\n';
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

}  // namespace

int main(int argc, char** argv) {
    const std::vector<std::string> args(argv + 1, argv + argc);

    try {
        if (args.size() == 3 && args[0] == "clean") return runClean(args[1], args[2]);
        if (args.size() == 5 && args[0] == "generate") {
            return runGenerate(args[1], args[2], args[3], args[4]);
        }
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
    return usage();
}
