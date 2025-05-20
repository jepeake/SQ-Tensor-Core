#!/bin/bash

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' 

TESTBENCH=""
DEBUG=0
VERBOSE=0
SEED=0
MAX_CYCLES=100
VIEW_WAVES=0
SEED_RANDOM=0
BUILD_ONLY=0
CLEAN_FIRST=0

OBJ_DIR="obj_dir"
WAVE_VIEWER="gtkwave"
PE_WAVE="waveform.vcd"
PE_ARRAY_WAVE="waveform_array.vcd"
PE_ARRAY_VERIFY_WAVE="waveform_array_verify.vcd"

function print_usage() {
    echo -e "${BLUE}Usage:${NC} $0 [options]"
    echo ""
    echo "Options:"
    echo "  -t, --testbench <name>  Run specific testbench (pe, pe_array, pe_array_verify, all)"
    echo "  -d, --debug             Enable debug output"
    echo "  -v, --verbose           Enable verbose output"
    echo "  -s, --seed <number>     Set random seed (default: random)"
    echo "  -r, --random-seed       Use random seed (default behavior)"
    echo "  -c, --cycles <number>   Set maximum cycles (default: 100)"
    echo "  -w, --view-waves        Open waveform viewer after simulation"
    echo "  -b, --build-only        Only build, don't run the simulation"
    echo "  --clean                 Clean build before running"
    echo "  -h, --help              Show this help message"
    echo ""
}

function log() {
    local level="$1"
    local message="$2"
    local color="${NC}"
    
    case "$level" in
        "INFO")    color="${GREEN}";;
        "WARNING") color="${YELLOW}";;
        "ERROR")   color="${RED}";;
        *)         color="${BLUE}";;
    esac
    
    echo -e "${color}[$level]${NC} $message"
}

function clean_build() {
    log "INFO" "Cleaning Build Directory..."
    make clean
}

function build_testbench() {
    local tb="$1"
    
    if [[ "$tb" == "pe" ]]; then
        log "INFO" "Building Processing Element Testbench..."
        make build_pe
    elif [[ "$tb" == "pe_array" ]]; then
        log "INFO" "Building Processing Element Array Testbench..."
        make build_pe_array
    elif [[ "$tb" == "pe_array_verify" ]]; then
        log "INFO" "Building Processing Element Array Verification Testbench..."
        make build_pe_array_verify
    elif [[ "$tb" == "all" ]]; then
        log "INFO" "Building All Testbenches..."
        make build_all
    else
        log "ERROR" "Unknown Testbench: $tb"
        return 1
    fi
    
    return $?
}

function run_testbench() {
    local tb="$1"
    local sim_flags=""
    
    if [[ $SEED_RANDOM -eq 1 ]]; then
        SEED=$RANDOM
    fi
    
    if [[ $SEED -ne 0 ]]; then
        sim_flags="$sim_flags --seed $SEED"
    fi
    
    if [[ $MAX_CYCLES -ne 100 ]]; then
        sim_flags="$sim_flags --cycles $MAX_CYCLES"
    fi
    
    if [[ $VERBOSE -eq 1 ]]; then
        sim_flags="$sim_flags --verbose"
    fi
    
    if [[ $DEBUG -eq 1 ]]; then
        sim_flags="$sim_flags --debug"
    fi
    
    if [[ "$tb" == "pe" ]]; then
        log "INFO" "Running Processing Element Testbench with Flags: $sim_flags"
        make run_pe SIM_FLAGS="$sim_flags"
        result=$?
        wave_file="$PE_WAVE"
    elif [[ "$tb" == "pe_array" ]]; then
        log "INFO" "Running Processing Element Array Testbench with Flags: $sim_flags"
        make run_pe_array SIM_FLAGS="$sim_flags"
        result=$?
        wave_file="$PE_ARRAY_WAVE"
    elif [[ "$tb" == "pe_array_verify" ]]; then
        log "INFO" "Running Processing Element Array Verification Testbench with Flags: $sim_flags"
        make run_pe_array_verify SIM_FLAGS="$sim_flags"
        result=$?
        wave_file="$PE_ARRAY_VERIFY_WAVE"
    else
        log "ERROR" "Unknown Testbench: $tb"
        return 1
    fi
    
    if [[ $result -eq 0 ]]; then
        log "INFO" "Testbench $tb Completed Successfully"
    else
        log "ERROR" "Testbench $tb Failed with Exit Code $result"
    fi
    
    if [[ $VIEW_WAVES -eq 1 && -f "$wave_file" ]]; then
        log "INFO" "Opening Waveform Viewer for $wave_file"
        $WAVE_VIEWER "$wave_file" &
    fi
    
    return $result
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -t|--testbench)
            TESTBENCH="$2"
            shift 2
            ;;
        -d|--debug)
            DEBUG=1
            shift
            ;;
        -v|--verbose)
            VERBOSE=1
            shift
            ;;
        -s|--seed)
            SEED="$2"
            SEED_RANDOM=0
            shift 2
            ;;
        -r|--random-seed)
            SEED_RANDOM=1
            shift
            ;;
        -c|--cycles)
            MAX_CYCLES="$2"
            shift 2
            ;;
        -w|--view-waves)
            VIEW_WAVES=1
            shift
            ;;
        -b|--build-only)
            BUILD_ONLY=1
            shift
            ;;
        --clean)
            CLEAN_FIRST=1
            shift
            ;;
        -h|--help)
            print_usage
            exit 0
            ;;
        *)
            log "ERROR" "Unknown option: $1"
            print_usage
            exit 1
            ;;
    esac
done

if [[ -z "$TESTBENCH" ]]; then
    echo -e "${YELLOW}Select Testbench to Run:${NC}"
    echo "1) Processing Element (PE)"
    echo "2) Processing Element Array (PE Array)"
    echo "3) Processing Element Array Verification (PE Array Verify)"
    echo "4) All Testbenches"
    echo -n "Enter Choice [1-4]: "
    read choice
    
    case "$choice" in
        1) TESTBENCH="pe";;
        2) TESTBENCH="pe_array";;
        3) TESTBENCH="pe_array_verify";;
        4) TESTBENCH="all";;
        *) 
            log "ERROR" "Invalid Choice"
            exit 1
            ;;
    esac
fi

if [[ $CLEAN_FIRST -eq 1 ]]; then
    clean_build
fi

if [[ "$TESTBENCH" == "all" ]]; then
    build_testbench "all"
    build_result=$?
    
    if [[ $BUILD_ONLY -eq 1 ]]; then
        exit $build_result
    fi
    
    if [[ $build_result -eq 0 && $BUILD_ONLY -eq 0 ]]; then
        run_testbench "pe"
        pe_result=$?
        
        run_testbench "pe_array"
        pe_array_result=$?
        
        run_testbench "pe_array_verify"
        pe_array_verify_result=$?

        if [[ $pe_result -ne 0 || $pe_array_result -ne 0 || $pe_array_verify_result -ne 0 ]]; then
            exit 1
        fi
    else
        exit $build_result
    fi
else
    build_testbench "$TESTBENCH"
    build_result=$?
    
    if [[ $BUILD_ONLY -eq 1 ]]; then
        exit $build_result
    fi
    
    if [[ $build_result -eq 0 && $BUILD_ONLY -eq 0 ]]; then
        run_testbench "$TESTBENCH"
        exit $?
    else
        exit $build_result
    fi
fi

exit 0 