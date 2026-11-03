#include "geo.hpp"

#include <algorithm>
#include <cmath>

namespace {

constexpr double kPi = 3.14159265358979323846;
constexpr double kDegToRad = kPi / 180.0;
constexpr double kRadToDeg = 180.0 / kPi;

// Haversine loses precision near antipodal points, where rounding can shift a
// distance by about 0.1 m. One metre of slack covers that with room to spare and
// costs almost nothing in pruning.
constexpr double kBoundSlackKm = 1e-3;

double wrappedLonDiffDeg(double lonA, double lonB) {
    double diff = std::fabs(lonA - lonB);
    return diff > 180.0 ? 360.0 - diff : diff;
}

}  // namespace

double haversineKm(double lat1, double lon1, double lat2, double lon2) {
    const double sinHalfLat = std::sin((lat2 - lat1) * kDegToRad / 2.0);
    const double sinHalfLon = std::sin((lon2 - lon1) * kDegToRad / 2.0);
    const double a = sinHalfLat * sinHalfLat +
                     std::cos(lat1 * kDegToRad) * std::cos(lat2 * kDegToRad) *
                         sinHalfLon * sinHalfLon;
    return 2.0 * kEarthRadiusKm * std::asin(std::sqrt(std::min(1.0, a)));
}

double minDistToRectKm(double lat, double lon, const Rect& box) {
    double exactKm;

    if (box.minLon <= lon && lon <= box.maxLon) {
        // Same meridian as part of the box: the nearest point is straight north
        // or south, because no point can be closer than its latitude gap.
        const double nearestLat = std::clamp(lat, box.minLat, box.maxLat);
        exactKm = std::fabs(lat - nearestLat) * kDegToRad * kEarthRadiusKm;
    } else {
        // At a fixed latitude, distance grows with the wrapped longitude gap, so
        // the nearest point lies on whichever edge meridian has the smaller gap.
        const double gapMin = wrappedLonDiffDeg(lon, box.minLon);
        const double gapMax = wrappedLonDiffDeg(lon, box.maxLon);
        const double edgeLon = gapMin <= gapMax ? box.minLon : box.maxLon;
        const double gap = std::min(gapMin, gapMax);

        // Latitude on that meridian closest to the query. Past a 90 degree gap
        // this falls beyond the pole, so the range test below rejects it.
        const double phi = lat * kDegToRad;
        const double closestLat =
            std::atan2(std::sin(phi), std::cos(phi) * std::cos(gap * kDegToRad)) * kRadToDeg;

        if (box.minLat <= closestLat && closestLat <= box.maxLat) {
            exactKm = haversineKm(lat, lon, closestLat, edgeLon);
        } else {
            exactKm = std::min(haversineKm(lat, lon, box.minLat, edgeLon),
                               haversineKm(lat, lon, box.maxLat, edgeLon));
        }
    }

    return std::max(0.0, exactKm - kBoundSlackKm);
}
