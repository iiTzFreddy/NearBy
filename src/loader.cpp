#include "loader.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <random>
#include <stdexcept>
#include <unordered_set>

namespace {

constexpr char kMagic[4] = {'N', 'B', 'Y', '1'};

static_assert(sizeof(Location) == 24, "the cache file format assumes 24-byte records");

// Reads fields 2-4 of a row. Returns false if the row doesn't have that shape.
bool parseRow(const char* line, double& lat, double& lon, int64_t& id) {
    const char* cursor = line;
    for (int skip = 0; skip < 2; ++skip) {
        cursor = std::strchr(cursor, '\t');
        if (cursor == nullptr) return false;
        ++cursor;
    }

    char* end = nullptr;
    lat = std::strtod(cursor, &end);
    if (end == cursor || *end != '\t') return false;

    cursor = end + 1;
    lon = std::strtod(cursor, &end);
    if (end == cursor || *end != '\t') return false;

    cursor = end + 1;
    id = std::strtoll(cursor, &end, 10);
    return end != cursor && (*end == '\0' || *end == '\r');
}

// The standard distributions differ between compilers, so these are written out
// by hand: a fixed seed must give teammates on different toolchains the same data.
double uniform01(std::mt19937_64& rng) {
    return static_cast<double>(rng() >> 11) * (1.0 / 9007199254740992.0);  // 2^-53
}

double gaussian(std::mt19937_64& rng) {
    constexpr double kTwoPi = 6.283185307179586;
    const double radius = std::sqrt(-2.0 * std::log(1.0 - uniform01(rng)));
    return radius * std::cos(kTwoPi * uniform01(rng));
}

void uniformOnSphere(std::mt19937_64& rng, double& lat, double& lon) {
    constexpr double kRadToDeg = 57.29577951308232;
    lat = std::asin(2.0 * uniform01(rng) - 1.0) * kRadToDeg;
    lon = uniform01(rng) * 360.0 - 180.0;
}

}  // namespace

std::vector<Location> parseGowalla(const std::string& path, CleanStats& stats) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open " + path);

    stats = CleanStats{};
    std::vector<Location> pts;
    std::unordered_set<int64_t> seen;

    std::string line;
    while (std::getline(in, line)) {
        if (line.empty() || line == "\r") continue;
        ++stats.rowsRead;

        double lat = 0.0;
        double lon = 0.0;
        int64_t id = 0;
        if (!parseRow(line.c_str(), lat, lon, id)) {
            ++stats.malformed;
        } else if (!(lat >= -90.0 && lat <= 90.0 && lon >= -180.0 && lon <= 180.0)) {
            ++stats.outOfRange;
        } else if (lat == 0.0 && lon == 0.0) {
            ++stats.zeroZero;
        } else if (!seen.insert(id).second) {
            ++stats.duplicates;
        } else {
            pts.push_back({id, lat, lon});
        }
    }

    stats.kept = pts.size();
    std::sort(pts.begin(), pts.end(),
              [](const Location& a, const Location& b) { return a.id < b.id; });
    return pts;
}

void saveLocations(const std::string& path, const std::vector<Location>& pts) {
    std::ofstream out(path, std::ios::binary);
    if (!out) throw std::runtime_error("cannot write " + path);

    const uint64_t count = pts.size();
    out.write(kMagic, sizeof(kMagic));
    out.write(reinterpret_cast<const char*>(&count), sizeof(count));
    out.write(reinterpret_cast<const char*>(pts.data()),
              static_cast<std::streamsize>(count * sizeof(Location)));
    if (!out) throw std::runtime_error("failed while writing " + path);
}

std::vector<Location> loadLocations(const std::string& path) {
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    if (!in) throw std::runtime_error("cannot open " + path);
    const uint64_t fileSize = static_cast<uint64_t>(in.tellg());
    in.seekg(0);

    char magic[sizeof(kMagic)] = {};
    uint64_t count = 0;
    in.read(magic, sizeof(magic));
    in.read(reinterpret_cast<char*>(&count), sizeof(count));
    constexpr uint64_t kHeaderSize = sizeof(kMagic) + sizeof(count);

    // Checking the size first means a corrupt count can't trigger a huge allocation.
    if (!in || std::memcmp(magic, kMagic, sizeof(kMagic)) != 0 ||
        (fileSize - kHeaderSize) / sizeof(Location) != count ||
        (fileSize - kHeaderSize) % sizeof(Location) != 0) {
        throw std::runtime_error(path + " is not a valid NearBy locations file");
    }

    std::vector<Location> pts(count);
    in.read(reinterpret_cast<char*>(pts.data()),
            static_cast<std::streamsize>(count * sizeof(Location)));
    if (!in) throw std::runtime_error("failed while reading " + path);
    return pts;
}

std::vector<Location> generateSynthetic(SyntheticKind kind, size_t n, uint64_t seed) {
    constexpr size_t kClusterCount = 20;
    constexpr double kClusterSpreadDeg = 2.0;

    std::mt19937_64 rng(seed);
    std::vector<Location> pts;
    pts.reserve(n);

    if (kind == SyntheticKind::Uniform) {
        for (size_t i = 0; i < n; ++i) {
            double lat = 0.0;
            double lon = 0.0;
            uniformOnSphere(rng, lat, lon);
            pts.push_back({static_cast<int64_t>(i), lat, lon});
        }
        return pts;
    }

    std::vector<Location> centers(kClusterCount);
    for (Location& center : centers) uniformOnSphere(rng, center.lat, center.lon);

    for (size_t i = 0; i < n; ++i) {
        const Location& center = centers[rng() % kClusterCount];
        const double lat = std::clamp(center.lat + gaussian(rng) * kClusterSpreadDeg, -90.0, 90.0);
        double lon = center.lon + gaussian(rng) * kClusterSpreadDeg;
        if (lon >= 180.0) lon -= 360.0;
        if (lon < -180.0) lon += 360.0;
        pts.push_back({static_cast<int64_t>(i), lat, lon});
    }
    return pts;
}
