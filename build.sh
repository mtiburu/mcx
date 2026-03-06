#!/bin/bash
# MCX Build and Benchmark Script

CUGENCODE="sm_86"
MCX_DIR="$(cd "$(dirname "$0")" && pwd)"
SRC_DIR="$MCX_DIR/src"
BIN_DIR="$MCX_DIR/bin"
PLOT_SCRIPT="$HOME/SurfaceNets/tools"
MATLAB_CMD="${MATLAB_CMD:-matlab2022b}"

usage() {
    echo "Usage: $0 <command> [options]"
    echo ""
    echo "Commands:"
    echo "  mcx          Build mcx binary"
    echo "  mex          Build MATLAB mex file"
    echo "  bench        Run benchmark (requires mcx build)"
    echo "  plot         Launch MATLAB to plot SVMC results"
    echo "  vectors      Plot centroid/normal vectors"
    echo "  demo         Run mcxlab SVMC demo scripts"
    echo "  clean        Clean build artifacts"
    echo ""
    echo "Benchmark options (with 'bench' command):"
    echo "  --mc         Run MC mode (--svmc 1) [default]"
    echo "  --sn         Run SN mode (--svmc 2)"
    echo "  --both       Run both MC and SN"
    echo "  -n <num>     Number of photons (default: 1e7)"
    echo "  -b <name>    Benchmark name (default: skinvessel)"
    echo "               Available: tspheres, mallet, cube60, zlayer_sphere60,"
    echo "               multisphere60, touch60t, touch60, cube60b, cube60planar,"
    echo "               cubesph60b, skinvessel, sphshells, spherebox, colin27"
    echo "  --list       List all available benchmarks"
    echo ""
    echo "Plot options (with 'plot' command):"
    echo "  --mc         Plot MC results (choice=1)"
    echo "  --sn         Plot SN results (choice=2) [default]"
    echo "  --both       Plot both MC and SN"
    echo "  --gui        Launch MATLAB with full GUI (default: -nodesktop)"
    echo ""
    echo "Environment variables:"
    echo "  MATLAB_CMD   MATLAB executable (default: matlab)"
    echo "               e.g. MATLAB_CMD=matlab2022b ./build.sh plot --sn"
    echo ""
    echo "Vectors options (with 'vectors' command):"
    echo "  --mc         Plot MC vectors (default)"
    echo "  --sn         Plot SN vectors"
    echo "  --both       Plot both MC and SN vectors"
    echo "  -s <scale>   Scale factor (default: 0.056)"
    echo "  -o <file>    Save to file (.png at 200 dpi, or .pdf)"
    echo "  --gui        Launch MATLAB with full GUI"
    echo ""
    echo "Demo options (with 'demo' command):"
    echo "  cubesph      Run demo_svmc_cubesph.m"
    echo "  sphshells    Run demo_svmc_sphshells.m"
    echo "  brain        Run demo_svmc_brain19_5.m"
    echo "  --gui        Launch MATLAB with full GUI (default: -nodesktop)"
    echo ""
    echo "Examples:"
    echo "  $0 mcx                    # Build mcx binary"
    echo "  $0 mex                    # Build mex file"
    echo "  $0 bench --sn             # Run SN benchmark (skinvessel)"
    echo "  $0 bench --both -n 1e8    # Run both with 1e8 photons"
    echo "  $0 bench -b colin27 --sn  # Run colin27 benchmark with SN"
    echo "  $0 bench --list           # List available benchmarks"
    echo "  $0 plot --sn              # Plot SN results"
    echo "  $0 plot --both --gui      # Plot both with MATLAB GUI"
    echo "  $0 vectors --sn           # Plot SN vectors"
    echo "  $0 vectors --both -s 0.05 # Plot both with custom scale"
    echo "  $0 vectors --sn -o normals.png  # Save to 200 dpi PNG"
    echo "  $0 vectors --sn -o normals.pdf  # Save to PDF"
    echo "  $0 demo cubesph           # Run cubesph demo"
    echo "  $0 demo brain --gui       # Run brain demo with GUI"
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

BENCHMARKS=("tspheres" "mallet" "cube60" "zlayer_sphere60" "multisphere60" "touch60t" "touch60" "cube60b" "cube60planar" "cubesph60b" "skinvessel" "sphshells" "spherebox" "colin27")

list_benchmarks() {
    echo "Available benchmarks:"
    for b in "${BENCHMARKS[@]}"; do
        echo "  $b"
    done
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
            --list) list_benchmarks; exit 0 ;;
            *)      echo "Unknown option: $1"; exit 1 ;;
        esac
    done
    
    # Validate benchmark name
    local valid=0
    for b in "${BENCHMARKS[@]}"; do
        [[ "$bench" == "$b" ]] && valid=1 && break
    done
    if [[ $valid -eq 0 ]]; then
        echo "Error: Unknown benchmark '$bench'"
        list_benchmarks
        exit 1
    fi
    
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

run_vectors() {
    local mode="mc"
    local scale="0.056"
    local outfile=""
    local gui=""
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            --mc)   mode="mc"; shift ;;
            --sn)   mode="sn"; shift ;;
            --both) mode="both"; shift ;;
            -s)     scale="$2"; shift 2 ;;
            -o)     outfile="$2"; shift 2 ;;
            --gui)  gui=1; shift ;;
            *)      echo "Unknown option: $1"; exit 1 ;;
        esac
    done
    
    local cmd="addpath('$PLOT_SCRIPT');"
    
    if [[ -n "$outfile" ]]; then
        case $mode in
            mc)   cmd+=" plot_vectors([], 1, $scale, '$outfile');" ;;
            sn)   cmd+=" plot_vectors([], 2, $scale, '$outfile');" ;;
            both) cmd+=" plot_vectors([], 1, $scale, 'mc_$outfile'); plot_vectors([], 2, $scale, 'sn_$outfile');" ;;
        esac
        cmd+=" exit;"
    else
        case $mode in
            mc)   cmd+=" plot_vectors([], 1, $scale);" ;;
            sn)   cmd+=" plot_vectors([], 2, $scale);" ;;
            both) cmd+=" plot_vectors([], 1, $scale); plot_vectors([], 2, $scale);" ;;
        esac
        cmd+=" disp('Press any key to exit...'); pause; exit;"
    fi
    
    echo "Plotting $mode vectors (scale=$scale)..."
    if [[ -n "$gui" ]]; then
        $MATLAB_CMD -nosplash -r "$cmd"
    else
        $MATLAB_CMD -nodesktop -nosplash -r "$cmd"
    fi
}

run_demo() {
    local demo=""
    local gui=""
    
    # Check for mex binary, build if missing
    local mex_file
    if [[ -f "$MCX_DIR/mcxlab/mcx.mexmaca64" ]]; then
        mex_file="$MCX_DIR/mcxlab/mcx.mexmaca64"
    elif [[ -f "$MCX_DIR/mcxlab/mcx.mexmaci64" ]]; then
        mex_file="$MCX_DIR/mcxlab/mcx.mexmaci64"
    elif [[ -f "$MCX_DIR/mcxlab/mcx.mexa64" ]]; then
        mex_file="$MCX_DIR/mcxlab/mcx.mexa64"
    else
        echo "Mex binary not found in $MCX_DIR/mcxlab/"
        echo "Building mex first..."
        echo ""
        build_mex
        if [[ $? -ne 0 ]]; then
            echo "Error: mex build failed"
            exit 1
        fi
        # Re-check for the built binary
        if [[ -f "$MCX_DIR/mcxlab/mcx.mexmaca64" ]]; then
            mex_file="$MCX_DIR/mcxlab/mcx.mexmaca64"
        elif [[ -f "$MCX_DIR/mcxlab/mcx.mexmaci64" ]]; then
            mex_file="$MCX_DIR/mcxlab/mcx.mexmaci64"
        elif [[ -f "$MCX_DIR/mcxlab/mcx.mexa64" ]]; then
            mex_file="$MCX_DIR/mcxlab/mcx.mexa64"
        else
            echo "Error: mex build did not produce expected binary"
            exit 1
        fi
        echo ""
    fi
    echo "Using mex: $mex_file"
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            cubesph)    demo="demo_svmc_cubesph"; shift ;;
            sphshells)  demo="demo_svmc_sphshells"; shift ;;
            brain)      demo="demo_svmc_brain19_5"; shift ;;
            --gui)      gui=1; shift ;;
            *)          echo "Unknown demo: $1"; exit 1 ;;
        esac
    done
    
    if [[ -z "$demo" ]]; then
        echo "Error: specify a demo (cubesph, sphshells, brain)"
        exit 1
    fi
    
    echo "Running $demo.m..."
    local cmd="addpath('$MCX_DIR/mcxlab'); addpath('$MCX_DIR/mcxlab/examples'); $demo; disp('Press any key to exit...'); pause; exit;"
    if [[ -n "$gui" ]]; then
        $MATLAB_CMD -nosplash -r "$cmd"
    else
        $MATLAB_CMD -nodesktop -nosplash -r "$cmd"
    fi
}

run_plot() {
    local mode="sn"
    local gui=""
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            --mc)   mode="mc"; shift ;;
            --sn)   mode="sn"; shift ;;
            --both) mode="both"; shift ;;
            --gui)  gui=1; shift ;;
            *)      echo "Unknown option: $1"; exit 1 ;;
        esac
    done
    
    local cmd="addpath('$PLOT_SCRIPT'); addpath('$MCX_DIR/mcxlab');"
    
    case $mode in
        mc)   cmd+="plot_svmc([],1);" ;;
        sn)   cmd+="plot_svmc([],2);" ;;
        both) cmd+="figure; plot_svmc([],1); figure; plot_svmc([],2);" ;;
    esac
    
    cmd+="disp('Press any key to exit...'); pause; exit;"
    
    echo "Launching MATLAB to plot $mode..."
    if [[ -n "$gui" ]]; then
        $MATLAB_CMD -nosplash -r "$cmd"
    else
        $MATLAB_CMD -nodesktop -nosplash -r "$cmd"
    fi
}

# Main
case "${1:-}" in
    mcx)    build_mcx ;;
    mex)    build_mex ;;
    bench)  shift; run_bench "$@" ;;
    demo)   shift; run_demo "$@" ;;
    vectors) shift; run_vectors "$@" ;;
    plot)   shift; run_plot "$@" ;;
    clean)  cd "$SRC_DIR" && make clean ;;
    -h|--help|"") usage ;;
    *)      echo "Unknown command: $1"; usage; exit 1 ;;
esac
