
// Prevent Windows.h from defining min/max macros, which would clash with
// std::min / std::max and std::numeric_limits<...>::max().
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#endif

#include "mapper/cover.hpp"
#include "mapper/embedding.hpp"
#include "mapper/mapper.hpp"
#include "mapper/two_step_mapper.hpp"

#include <Eigen/Dense>

#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#endif

namespace fs = std::filesystem;

namespace {

// ---------------------------------------------------------------------------
// Small utilities
// ---------------------------------------------------------------------------

std::string trim(const std::string& s) {
    const auto start = s.find_first_not_of(" \t\r\n");
    if (start == std::string::npos) return "";
    const auto end = s.find_last_not_of(" \t\r\n");
    return s.substr(start, end - start + 1);
}

// Returns true when the process owns its own console, i.e. it was most
// likely launched by double-clicking the executable rather than from an
// existing shell.
bool is_launched_by_double_click() {
#ifdef _WIN32
    DWORD processes[2] = {0, 0};
    const DWORD count  = ::GetConsoleProcessList(processes, 2);
    return count <= 1;
#else
    return false;
#endif
}

void pause_if_standalone() {
    if (!is_launched_by_double_click()) return;

    std::cout << "\nPress Enter to exit..." << std::flush;
    std::cin.clear();
    // Parenthesise the member call so a leftover "max" macro from any
    // Windows header cannot swallow the identifier.
    std::cin.ignore((std::numeric_limits<std::streamsize>::max)(), '\n');
    std::cin.get();
}

// ---------------------------------------------------------------------------
// Data loading
// ---------------------------------------------------------------------------

// Loads a txt file where each non-empty line is one point. Coordinates are
// separated by whitespace (space, tab). 
bool load_data(const std::string& path, mapper::Dataset& data) {
    std::ifstream in(path);
    if (!in) {
        std::cerr << "Cannot open " << path << "\n";
        return false;
    }

    std::vector<std::vector<double>> rows;
    std::string line;
    std::size_t line_no = 0;
    while (std::getline(in, line)) {
        ++line_no;
        const std::string t = trim(line);
        if (t.empty()) continue;

        std::stringstream ss(t);
        std::string       tok;
        std::vector<double> row;
        while (ss >> tok) {
            try {
                row.push_back(std::stod(tok));
            } catch (...) {
                std::cerr << "Line " << line_no << ": cannot parse '" << tok
                          << "' as a number.\n";
                return false;
            }
        }
        if (row.empty()) continue;
        rows.push_back(std::move(row));
    }

    if (rows.empty()) {
        std::cerr << "No data read from " << path << "\n";
        return false;
    }

    const int n = static_cast<int>(rows.size());
    const int d = static_cast<int>(rows[0].size());
    for (int i = 1; i < n; ++i) {
        if (static_cast<int>(rows[i].size()) != d) {
            std::cerr << "Row " << (i + 1) << " has " << rows[i].size()
                      << " values; expected " << d << "\n";
            return false;
        }
    }
    if (d < 1) {
        std::cerr << "Data must have at least one dimension.\n";
        return false;
    }

    data.resize(n, d);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < d; ++j) {
            data(i, j) = rows[i][j];
        }
    }
    return true;
}

// ---------------------------------------------------------------------------
// Output writers
// ---------------------------------------------------------------------------

void write_nodes_csv(const mapper::MapperGraph& g, const std::string& path) {
    std::ofstream out(path);
    if (!out) {
        std::cerr << "Cannot write " << path << "\n";
        return;
    }
    out << "node_id,n_elements,elements\n";
    for (std::size_t i = 0; i < g.num_nodes(); ++i) {
        const auto& e = g.nodes()[i].elements;
        out << i << "," << e.size() << ",\"";
        for (std::size_t k = 0; k < e.size(); ++k) {
            if (k) out << ";";
            out << e[k];
        }
        out << "\"\n";
    }
}

void write_edges_csv(const mapper::MapperGraph& g, const std::string& path) {
    std::ofstream out(path);
    if (!out) {
        std::cerr << "Cannot write " << path << "\n";
        return;
    }
    out << "src,dst\n";
    for (const auto& e : g.edges()) {
        out << e.first << "," << e.second << "\n";
    }
}

void write_summary(const mapper::MapperGraph& g,
                   const std::string&         path,
                   const std::string&         data_path,
                   double                     eps) {
    std::ofstream out(path);
    if (!out) {
        std::cerr << "Cannot write " << path << "\n";
        return;
    }
    out << "Two-step Mapper summary\n";
    out << "========================\n";
    out << "input          : " << data_path << "\n";
    out << "eps            : " << eps << "\n";
    out << "num_nodes      : " << g.num_nodes() << "\n";
    out << "num_edges      : " << g.num_edges() << "\n";
    out << "\nFiles produced:\n";
    out << "  graph.html              : interactive Mapper visualization\n";
    out << "  nodes.csv               : per-node element indices\n";
    out << "  edges.csv               : edge list\n";
    out << "  summary.txt             : this file\n";


}   
// ---------------------------------------------------------------------------
// Command-line parsing
// ---------------------------------------------------------------------------

struct Options {
    std::string data_path;
    std::string out_dir    = "output";
    double      eps        = -1.0;    // < 0 means "not provided"
    double      overlap    = -1.0;    // < 0 means "compute automatically"
    bool        no_pause   = false;
    bool        show_help  = false;
};

void print_usage() {
    std::cout
        << "Usage:\n"
        << "  two_step_mapper_cli <data.txt> --eps <value>\n"
        << "      [--out <dir>] [--overlap <value>]\n"
        << "      [--no-pause] [-h|--help]\n"
        << "\n"
        << "Arguments:\n"
        << "  <data.txt>          Input file. One point per line, whitespace\n"
        << "                      separated (e.g. \"x y\" or \"x y z\").\n"
        << "  --eps <value>       Neighbourhood radius (required).\n"
        << "  --out <dir>         Output directory (default: output).\n"
        << "  --overlap <value>   Cover overlap in (0, 0.5). If omitted, the\n"
        << "                      overlap is derived from r_max automatically.\n"
        << "  --no-pause          Do not wait for Enter on exit.\n"
        << "  -h, --help          Print this help.\n";
}

bool parse_args(int argc, char** argv, Options& opt) {
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];

        // Accept both "--key value" and "--key=value".
        auto try_value = [&](const std::string& key, std::string& out) -> bool {
            if (a == key) {
                if (i + 1 >= argc) return false;
                out = argv[++i];
                return true;
            }
            const std::string prefix = key + "=";
            if (a.rfind(prefix, 0) == 0) {
                out = a.substr(prefix.size());
                return true;
            }
            return false;
        };

        std::string v;
        if (a == "-h" || a == "--help") {
            opt.show_help = true;
            return false;
        } else if (a == "--no-pause") {
            opt.no_pause = true;
        } else if (try_value("--eps", v)) {
            try {
                opt.eps = std::stod(v);
            } catch (...) {
                std::cerr << "Invalid --eps value: '" << v << "'\n";
                return false;
            }
        } else if (try_value("--out", v)) {
            opt.out_dir = v;
        } else if (try_value("--overlap", v)) {
            try {
                opt.overlap = std::stod(v);
            } catch (...) {
                std::cerr << "Invalid --overlap value: '" << v << "'\n";
                return false;
            }
        } else if (!a.empty() && a[0] != '-') {
            if (opt.data_path.empty()) {
                opt.data_path = a;
            } else {
                std::cerr << "Unexpected positional argument: '" << a << "'\n";
                return false;
            }
        } else {
            std::cerr << "Unknown option: '" << a << "'\n";
            return false;
        }
    }
    return true;
}

} // namespace

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main(int argc, char** argv) {
    // Keeps the console alive when the executable is double-clicked, but is a
    // no-op when launched from an existing shell.
    struct ExitPause {
        bool enabled = true;
        ~ExitPause() {
            if (enabled) pause_if_standalone();
        }
    } pause_guard;

    Options opt;
    if (!parse_args(argc, argv, opt)) {
        if (opt.no_pause) pause_guard.enabled = false;
        if (opt.show_help) {
            pause_guard.enabled = false;   // help is usually piped/read
            print_usage();
            return 0;
        }
        std::cerr << "\n";
        print_usage();
        return 1;
    }
    if (opt.no_pause) pause_guard.enabled = false;

    // Required arguments.
    if (opt.data_path.empty()) {
        std::cerr << "Error: missing input data file.\n\n";
        print_usage();
        return 1;
    }
    if (opt.eps <= 0.0) {
        std::cerr << "Error: --eps must be provided and positive.\n\n";
        print_usage();
        return 1;
    }
    if (opt.overlap >= 0.5 || (opt.overlap > 0.0 && opt.overlap < 1e-12)) {
        std::cerr << "Error: --overlap must lie strictly between 0 and 0.5 "
                     "(or be omitted for auto mode).\n";
        return 1;
    }

    // Load data.
    mapper::Dataset data;
    if (!load_data(opt.data_path, data)) return 1;
    std::cout << "Loaded " << data.rows() << " points, "
              << data.cols() << " dimensions from '" << opt.data_path << "'.\n";

    // Prepare output directory.
    {
        std::error_code ec;
        fs::create_directories(opt.out_dir, ec);
        if (ec) {
            std::cerr << "Cannot create directory '" << opt.out_dir
                      << "': " << ec.message() << "\n";
            return 1;
        }
    }

    // Fixed algorithm parameters (not exposed as command-line options).
    const int min_samples     = 4;
    const int split_threshold = 2;

    // Resolve overlap.
    const bool overlap_auto = (opt.overlap <= 0.0);
    double     overlap      = overlap_auto ? (1.0 / 3.0) : opt.overlap;

    if (overlap_auto) {
        overlap = mapper::estimate_overlap_from_r_max(
            data, opt.eps, min_samples, split_threshold, overlap);
    }

    // Final two-step Mapper.
    mapper::TwoStepMapperParams tp_final;
    tp_final.overlap         = overlap;
    tp_final.eps             = opt.eps;
    tp_final.min_samples     = min_samples;
    tp_final.split_threshold = split_threshold;
    tp_final.method          = mapper::ClusterMethod::EpsGraph;

    mapper::TwoStepMapper             tsm_final(tp_final);
    const mapper::TwoStepMapperResult result = tsm_final.fit_transform(data);

    const auto& g = result.final_graph;
    std::cout << "Final graph: nodes = " << g.num_nodes()
              << ", edges = " << g.num_edges() << "\n";

    // Save outputs.
    const std::string html_path    = (fs::path(opt.out_dir) / "graph.html").string();
    const std::string points_html  = (fs::path(opt.out_dir) / "points.html").string();
    const std::string nodes_path   = (fs::path(opt.out_dir) / "nodes.csv").string();
    const std::string edges_path   = (fs::path(opt.out_dir) / "edges.csv").string();
    const std::string summary_path = (fs::path(opt.out_dir) / "summary.txt").string();

    // Visualization uses the first two original features; embedding_2d pads
    // missing columns with zeros for 1D inputs.
    const Eigen::MatrixXd embedding = mapper::embedding_2d(data);

    {
        std::ofstream html(html_path);
        if (!html) {
            std::cerr << "Cannot write " << html_path << "\n";
            return 1;
        }
        g.write_html(html, embedding, "Two-step Mapper");
    }
    {
        std::ofstream html(points_html);
        if (!html) {
            std::cerr << "Cannot write " << points_html << "\n";
            return 1;
        }
        mapper::write_points_html(html, data, "Point Cloud");
    }

    write_nodes_csv(g, nodes_path);
    write_edges_csv(g, edges_path);
    write_summary(g, summary_path, opt.data_path, opt.eps);

    std::cout << "\nOutputs written to '" << opt.out_dir << "':\n"
              << "  " << html_path << "\n"
              << "  " << points_html << "\n"
              << "  " << nodes_path << "\n"
              << "  " << edges_path << "\n"
              << "  " << summary_path << "\n";
            

    return 0;
}