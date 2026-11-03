#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "geo.hpp"

// Row counts from cleaning, kept for the report.
struct CleanStats {
    uint64_t rowsRead = 0;
    uint64_t malformed = 0;   // missing or unparsable fields
    uint64_t outOfRange = 0;  // lat outside [-90, 90] or lon outside [-180, 180]
    uint64_t zeroZero = 0;    // exactly (0, 0), a placeholder rather than a place
    uint64_t duplicates = 0;  // valid rows repeating a location_id already kept
    uint64_t kept = 0;
};

// Reads the tab-separated Gowalla check-in file (user, check-in time, lat, lon,
// location_id) and returns one Location per location_id, sorted by id. The first
// valid row for an id wins. Throws std::runtime_error if the file can't be read.
std::vector<Location> parseGowalla(const std::string& path, CleanStats& stats);

// Compact binary cache of cleaned locations, so later runs load in seconds.
// loadLocations throws std::runtime_error on a missing or corrupt file.
void saveLocations(const std::string& path, const std::vector<Location>& pts);
std::vector<Location> loadLocations(const std::string& path);

enum class SyntheticKind {
    Uniform,    // evenly spread over the sphere
    Clustered,  // Gaussian blobs, imitating city clusters
};

// Deterministic for a given (kind, n, seed), on every compiler. Ids are 0..n-1.
std::vector<Location> generateSynthetic(SyntheticKind kind, size_t n, uint64_t seed);
