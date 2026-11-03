#pragma once

#include <cstdint>
#include <vector>

// One entry per unique Gowalla location_id.
struct Location {
    int64_t id;
    double lat;
    double lon;
};

// idx points into the shared `points` array.
struct Result {
    uint32_t idx;
    double dist_km;
};

// A latitude/longitude box in degrees. Requires minLat <= maxLat and
// minLon <= maxLon, so a box never spans the antimeridian itself. The root box
// is [-90, 90] x [-180, 180].
struct Rect {
    double minLat;
    double maxLat;
    double minLon;
    double maxLon;
};

constexpr double kEarthRadiusKm = 6371.0;

// The only distance function in the project, so that every structure produces
// bit-identical distances. Always pass the query first, then the data point.
double haversineKm(double lat1, double lon1, double lat2, double lon2);

// Lower bound on the distance from a query to any point inside `box`, used for
// pruning by both trees. Longitude differences wrap, so a query at -179 sees a
// box ending at +179 as 2 degrees away and antimeridian searches need no special
// case. Returns 0 if the query is inside the box.
//
// The result is the exact sphere distance minus a small slack, which keeps it
// below any value haversineKm can round to.
//
// Prune only when the bound is strictly greater than the current limit:
//     kNN:    skip if minDistToRectKm(...) >  kthDist   (when the heap is full)
//     radius: skip if minDistToRectKm(...) >  r_km
// A subtree whose bound equals kthDist can still hold a tie with a smaller id.
// That happens whenever the query sits exactly on duplicate coordinates, where
// the bound and kthDist are both 0.
double minDistToRectKm(double lat, double lon, const Rect& box);

// The project-wide result order: (dist_km, id) ascending. Location ids are
// unique after cleaning, so this is a total order and every structure can match
// brute force exactly.
struct ResultOrder {
    const std::vector<Location>* points;

    bool operator()(const Result& a, const Result& b) const {
        if (a.dist_km != b.dist_km) return a.dist_km < b.dist_km;
        return (*points)[a.idx].id < (*points)[b.idx].id;
    }
};
