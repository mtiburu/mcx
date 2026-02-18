#!/bin/bash
# MCX Build and Benchmark Script

CUGENCODE="sm_86"
MCX_DIR="$(cd "$(dirname "$0")" && pwd)"
SRC_DIR="$MCX_DIR/src"
BIN_DIR="$MCX_DIR/bin"
PLOT_SCRIPT="$HOME/SurfaceNets/tools"

usage() {
    echo "Usage: $0 <command> [options]"
    echo ""
    echo "Commands:"
    echo "  mcx          Build mcx binary"
    echo "  mex          Build MATLAB mex file"
    echo "  bench        Run benchmark (requires mcx build)"
    echo "  plot         Launch MATLAB to plot SVMC results"
    echo "  clean        Clean build artifacts"
    echo ""
    echo "Benchmark options (with 'bench' command):"
    echo "  --mc         Run MC mode (--svmc 1) [default]"
    echo "  --sn         Run SN mode (--svmc 2)"
    echo "  --both       Run both MC and SN"
    echo "  -n <num>     Number of photons (default: 1e7)"
    echo "  -b <name>    Benchmark name (default: skinvessel)"
    echo ""
    echo "Plot options (with 'plot' command):"
    echo "  --mc         Plot MC results (choice=1)"
    echo "  --sn         Plot SN results (choice=2) [default]"
    echo "  --both       Plot both MC and SN"
    echo "  --gui        Launch MATLAB with GUI (default: no GUI)"
    echo ""
    echo "Examples:"
    echo "  $0 mcx                    # Build mcx binary"
    echo "  $0 mex                    # Build mex file"
    echo "  $0 bench --sn             # Run SN benchmark"
    echo "  $0 bench --both -n 1e8    # Run both with 1e8 photons"
    echo "  $0 plot --sn              # Plot SN results"
    echo "  $0 plot --both --gui      # Plot both with MATLAB GUI"
}

build_mcx() {
    echo "Building mcx binary..."
    cd "$SRC_DIR"
    make clean && make CUGENCODE=-arch=$CUGENCODE -j
    echo "Done. Binary at: $BIN_DIR/mcx"
}

build_mex() {
    echo "Building mex file..."
    cd "$SRC_DIR"
    make clean && make mex CUGENCODE=-arch=$CUGENCODE -j
    echo "Done. Mex file at: $MCX_DIR/mcxlab/"
}

run_bench() {
    local mode="mc"
    local nphotons="1e7"
    local bench="skinvessel"
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            --mc)   mode="mc"; shift ;;
            --sn)   mode="sn"; shift ;;
            --both) mode="both"; shift ;;
            -n)     nphotons="$2"; shift 2 ;;
            -b)     bench="$2"; shift 2 ;;
            *)      echo "Unknown option: $1"; exit 1 ;;
        esac
    done
    
    run_single() {
        local svmc=$1
        local label=$2
        echo ""
        echo "=== Running $label (--svmc $svmc) ==="
        "$BIN_DIR/mcx" --bench "$bench" --svmc "$svmc" -n "$nphotons" --session "${label}_test" -F mc2
    }
    
    case $mode in
        mc)   run_single 1 "MC" ;;
        sn)   run_single 2 "SN" ;;
        both) run_single 1 "MC"; run_single 2 "SN" ;;
    esac
}

run_plot() {
    local mode="sn"
    local gui=""
    local matlab_opts="-nodesktop -nosplash"
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            --mc)   mode="mc"; shift ;;
            --sn)   mode="sn"; shift ;;
            --both) mode="both"; shift ;;
            --gui)  gui=1; shift ;;
            *)      echo "Unknown option: $1"; exit 1 ;;
        esac
    done
    
    [[ -z "$gui" ]] && matlab_opts="$matlab_opts -batch"
    
    local cmd="addpath('$PLOT_SCRIPT'); addpath('$MCX_DIR/mcxlab');"
    
    case $mode in
        mc)   cmd+="plot_svmc([],1);" ;;
        sn)   cmd+="plot_svmc([],2);" ;;
        both) cmd+="figure; plot_svmc([],1); figure; plot_svmc([],2);" ;;
    esac
    
    [[ -n "$gui" ]] && cmd+="pause;"  # keep figures open in GUI mode
    
    echo "Launching MATLAB to plot $mode..."
    if [[ -n "$gui" ]]; then
        matlab -nodesktop -nosplash -r "$cmd"
    else
        matlab -batch "$cmd"
    fi
}

# Main
case "${1:-}" in
    mcx)    build_mcx ;;
    mex)    build_mex ;;
    bench)  shift; run_bench "$@" ;;
    plot)   shift; run_plot "$@" ;;
    clean)  cd "$SRC_DIR" && make clean ;;
    -h|--help|"") usage ;;
    *)      echo "Unknown command: $1"; usage; exit 1 ;;
esac
