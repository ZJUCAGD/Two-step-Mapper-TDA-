# mapper-cpp

A **Two-step Mapper** (Topological Data Analysis / TDA) library and command-line tool implemented in C++17.

## Features

- Two-step Mapper pipeline: a 1D cover along the first principal direction + an orthogonal split
- Two built-in clusterers: `EpsGraph` (connected components of the ε-neighborhood graph) and `DBSCAN` (backed by a nanoflann KD-tree)
- Automatic estimation of `r_max` from the local geometry of singular nodes (degree ≥ 3), used to derive the cover `overlap`
- Interactive HTML visualization (Mapper graph + point cloud)
- Depends only on Eigen3 (header-only); nanoflann is bundled

## Dependencies

| Dependency | Version | Notes |
|---|---|---|
| CMake | ≥ 3.15 | Build system |
| C++ compiler | C++17 | GCC / Clang / MSVC |
| [Eigen3](https://eigen.tuxfamily.org) | ≥ 3.3 | Header-only linear algebra library |

> [nanoflann](https://github.com/jlblancoc/nanoflann) is bundled under `third_party/nanoflann`; no separate installation is required.

## Building

### Windows (Visual Studio 2019)

Run from an "x64 Native Tools Command Prompt for VS 2019":

```bat
cmake -G "Visual Studio 16 2019" -A x64 -S . -B build
cmake --build build --config Release
```

> If Eigen3 is not found automatically, provide its location via `-DCMAKE_PREFIX_PATH=<eigen-prefix>` or `-DEigen3_DIR=<prefix>/share/eigen3/cmake`.

## Usage

```text
two_step_mapper_cli <data.txt> --eps <value>
    [--out <dir>] [--overlap <value>]
    [--no-pause] [-h|--help]
```

| Option | Description |
|---|---|
| `<data.txt>` | Input data file (required) |
| `--eps <value>` | Neighborhood radius; must be positive (required; value depends on the data scale) |
| `--out <dir>` | Output directory (default: `output`) |
| `--overlap <value>` | Cover overlap in (0, 0.5); when omitted, it is estimated automatically from `r_max` |
| `--no-pause` | Do not wait for Enter on exit (double-click launches) |
| `-h`, `--help` | Print help |

> The command-line tool uses the `EpsGraph` clusterer by default.

### Input format

One point per line; coordinates are separated by whitespace (space / tab):

```text
0.0 0.0
1.0 0.5
0.5 1.0
```

### Example

```bash
./two_step_mapper_cli data/uv_point/multi_singular_uv_point.txt --eps 0.00883883476483228 --out output
```

## Output files

| File | Description |
|---|---|
| `graph.html` | Interactive Mapper graph visualization |
| `points.html` | Normalized 2D point cloud |
| `nodes.csv` | Element indices per node |
| `edges.csv` | Edge list |
| `summary.txt` | Run summary (input, eps, overlap, node/edge counts, ...) |

## Project structure

```text
mapper-cpp/
├── apps/                    # Command-line program
│   └── two_step_mapper_cli.cpp
├── include/mapper/          # Public headers
├── src/                     # Library implementation
├── third_party/nanoflann/   # Bundled nanoflann
├── data/                    # B-spline intersection test data
├── CMakeLists.txt
└── build.bat                # Windows one-click build script
```

## Algorithm overview

The two-step Mapper builds the cover and graph in two stages:

1. **First stage**: build a uniform 1D cover along the first principal direction to obtain the initial Mapper graph;
2. **Second stage**: re-split nodes whose orthogonal spread reaches the split threshold `S_i` along the orthogonal direction.

When `--overlap` is omitted, the program first runs an initial Mapper pass and estimates the maximum local radius `r_max` from the geometry of singular nodes (degree ≥ 3) and their adjacent strips, then derives the cover overlap from `r_max`, adapting the overlap automatically.

## Sample data

The `data/` directory contains B-spline intersection test data produced by a subdivision-based intersection method:

| Path | Description |
|---|---|
| `uv_rectangle/` | UV parameter-domain rectangles; each row is `u_min u_max v_min v_max` |
| `uv_rectangle_max_diagonal.txt` | Maximum rectangle diagonal length per dataset (`<name> <diagonal>`) |
| `uv_point/` | Center of each parameter-domain rectangle — the final intersection result; each row is `u v` |

For the two-step Mapper, the recommended neighborhood radius is **2 × the maximum rectangle diagonal length** (listed in `uv_rectangle_max_diagonal.txt`). For example, `multi_singular` has a max diagonal of `0.00441941738241614`, so use `--eps 0.00883883476483228`.
