# NearBy

NearBy answers one question: how quickly can you find the k nearest places, or every place within a radius, of a latitude/longitude point? It compares a k-d tree and a quadtree, both built from scratch in C++, against a brute-force scan on 1.28 million real locations from the [SNAP Gowalla dataset](https://snap.stanford.edu/data/loc-gowalla.html).

**Team Coding Serpents:** Freddy Rives, Pedro Guevara, Alfonso Castano

**Current status:** the data pipeline, brute force, k-d tree, quadtree, benchmark, graphs and menu are all in place, and both trees match brute force in the tests.

## Requirements

- A C++17 compiler (MSVC 2022 and clang 21 are tested)
- [CMake](https://cmake.org/download/) 3.16 or newer
- Python 3 with matplotlib, for the graphs: `pip install matplotlib`

## Build

```sh
cmake -S . -B build
cmake --build build --config Release
```

The program is `build/Release/nearby.exe` with Visual Studio, and `build/nearby` with other generators. The examples below use `nearby` for either one, and all commands run from the repository root.

## Get the data

The raw dataset isn't stored in this repository.

1. Create a `data/` folder.
2. Download [loc-gowalla_totalCheckins.txt.gz](https://snap.stanford.edu/data/loc-gowalla_totalCheckins.txt.gz) (105 MB).
3. Decompress it to `data/gowalla.txt` (395 MB).
4. Clean it, which takes under a minute and only needs doing once:

```sh
nearby clean data/gowalla.txt data/gowalla_clean.bin
```

This drops invalid rows, keeps one row per `location_id`, and writes a 31 MB cache that loads in well under a second.

## Run the program

Run `nearby` with no arguments to open the menu:

```text
NearBy
  1. Load dataset
  2. Find the k nearest places
  3. Find places within a radius
  4. Run benchmark
  5. Exit
```

Start with option 1 and press Enter to load `data/gowalla_clean.bin`. Options 2 and 3 ask for a location and a k or radius, let you pick a structure or all of them, and show the results with the query time. To try it, search from Gainesville at latitude 29.65, longitude -82.32.

## Run the tests

```sh
ctest --test-dir build -C Release --output-on-failure
```

Brute force is the reference: every structure must return the same results in the same order, with identical distances.

To run the same check on the real data, after cleaning it:

```sh
nearby verify                  # uses data/gowalla_clean.bin
nearby verify path/to/data.bin
```

This compares every structure against brute force on 300 queries with a fixed seed, half on real locations and half spread over the whole globe, for k = 1, 5, 10, 100 and radii of 0.5, 5, 50 and 500 km. It prints the mismatch count per structure and exits with a nonzero code if any answer differs. Brute force is slow at full size, so this takes a few minutes.

## Run the benchmark and make the graphs

```sh
nearby bench            # every size, on real and synthetic data
nearby bench --quick    # up to 100,000 points, for checking things work
python scripts/plot.py
```

The benchmark also runs from menu option 4. It writes `results/benchmark.csv`, and `plot.py` writes the graphs to `results/`:

| File | Shows |
| --- | --- |
| `build_time.png` | Build time as the dataset grows |
| `knn_query_time.png` | kNN query time for each k |
| `radius_query_time.png` | Radius query time for each radius |
| `memory.png` | Memory used by each structure |
| `quadtree_tuning.png` | Quadtree bucket size against depth limit |
| `uniform_vs_clustered.png` | The same kNN query on each kind of data |
| `report_summary.png` | A 2×2 summary for the report |

A slow structure stops a measurement after 1 second, once it has run at least 20 queries. The `queries` column in the CSV records how many ran. Publishable numbers need a Release build, with every run on the same machine.

## Project layout

```text
src/spatial_index.hpp   Interface every structure implements
src/geo.hpp, geo.cpp    Haversine distance, pruning bound, result ordering
src/loader.hpp, .cpp    Gowalla parsing and cleaning, cache files, synthetic data
src/brute.hpp, .cpp     Brute-force baseline
src/bench.hpp, .cpp     Benchmark harness
src/main.cpp            Menu and command-line tools
tests/                  Correctness tests against brute force
scripts/plot.py         Graphs from the benchmark CSVs
```

## Adding a structure

1. Implement `SpatialIndex`, following the rules at the top of `src/spatial_index.hpp`.
2. Add the `.cpp` file to `nearby_core` in `CMakeLists.txt`.
3. Add it to `makeAllIndexes()` in `src/bench.cpp`, which adds it to the menu and benchmark.
4. Call `runIndexContract` on it in `tests/test_main.cpp`.

For the quadtree, also return it from `quadTreeFactory()` in `src/bench.cpp` to turn on the tuning sweep.

## References

- E. Cho, S. A. Myers, J. Leskovec. Friendship and Mobility: User Movement in Location-Based Social Networks. ACM SIGKDD, 2011.
- J. L. Bentley. Multidimensional binary search trees used for associative searching. *Communications of the ACM*, 18(9), 509–517, 1975.
- R. A. Finkel and J. L. Bentley. Quad trees: a data structure for retrieval on composite keys. *Acta Informatica*, 4(1), 1–9, 1974.
