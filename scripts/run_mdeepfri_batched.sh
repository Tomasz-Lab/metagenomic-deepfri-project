#!/bin/bash

# =============================================================================
# mDeepFRI predict-function runner script with FASTA batching
# =============================================================================
# This script splits large FASTA files into batches of max 10k sequences,
# runs mDeepFRI on each batch, and merges results to appear as a single run.
# =============================================================================

# -----------------------------------------------------------------------------
# CONFIGURATION - Modify these variables as needed
# -----------------------------------------------------------------------------

# Input FASTA file
INPUT_FILE="/home/FilipS/2025/metagenomic_deepfri/data/source/uhgp_mags/uhgp_50_mags/uhgp_50_mags.faa"

# Path to model weights folder
WEIGHTS_PATH="weights_folder/"

# Output directory
OUTPUT_DIR="results/uhgp_50_mags"

# Maximum sequences per batch (default: 10000)
MAX_SEQUENCES_PER_BATCH=10000

# Timing log file (will be created in OUTPUT_DIR)
TIMING_LOG="${OUTPUT_DIR}/timing.log"

# Terminal output log file (will be created in OUTPUT_DIR)
# Set to empty string to disable saving terminal output
TERMINAL_LOG="${OUTPUT_DIR}/terminal_output.log"
# Set to true to save all terminal output (stdout + stderr) to TERMINAL_LOG
SAVE_TERMINAL_OUTPUT=true

# Database paths (can specify multiple databases)
# Example: DATABASES=("database1" "database2")
# Leave empty to use only PDB100: DATABASES=()
# Note: PDB100 is searched by default, then these databases are searched
DATABASES=("/mnt/storage_10T/afdb_foldcomp/afdb_uniprot_v4" "/mnt/storage_10T/software/Metagenomic-DeepFRI/highquality_clust30/highquality_clust30")

# Number of top MMSeqs2 hits to save per query
# Then PER_QUERY (defaults to "topbits") selects the best hit from these
# Recommended: Set to 100-200 to ensure you get the best hit
# If not set, defaults to 1 (which means only 1 hit per query)
TOP_K=0

# Identity bins (format: "low,high")
# These are used for organizing results, not for filtering MMSeqs2 hits
# If you want to filter only with pyopal (cmap-identity/cmap-coverage), use a single bin covering all identities
# Example: IDENTITY_BINS=("0.9,1.0" "0.8,0.9" "0.7,0.8" "0.6,0.7")
# NOTE: Upper bound is EXCLUSIVE (<), so use 1.01 to include perfect 1.0 identity hits
# For no identity-based filtering, use a single bin covering all (including 1.0):
IDENTITY_BINS=("0.00,1.01")

# Generate contacts values
# Example: GENERATE_CONTACTS=(0 1 2 3 4)
GENERATE_CONTACTS=(2)

# Processing modes (bp, cc, ec, mf)
PROCESSING_MODES=("bp" "cc" "mf")

# Additional options
SAVE_CMAPS=false
SAVE_STRUCTURES=false
SAVE_RAW_ALIGNMENTS=false  # Set to true to save raw alignments (gapped sequences) from pyopal
THREADS=8
SKIP_PDB=false

# -----------------------------------------------------------------------------
# ADVANCED OPTIONS (optional - uncomment and modify if needed)
# -----------------------------------------------------------------------------

# MMSeqs2 search parameters
# SENSITIVITY=5.7
# MIN_BITSCORE=0
# MAX_EVAL=0.001
# Set MIN_IDENTITY=0.0 to not filter MMSeqs2 hits by identity (filtering happens later with pyopal)
MIN_IDENTITY=0.0
# MIN_COVERAGE=0.9

# Contact map alignment parameters (used for pyopal filtering)
# ANGSTROM_CONTACT_THRESH=6
# ALIGNMENT_GAP_OPEN=10
# ALIGNMENT_GAP_EXTEND=1
# These filter alignments after pyopal alignment (not MMSeqs2 hits)
CMAP_IDENTITY=0.2
CMAP_COVERAGE=0.9

# Other options
# PER_QUERY defaults to "topbits" (best hit by bitscore) - this is what you want
# PER_QUERY="random"  # or "topbits" - uncomment to change from default
DROP_SELF_HITS=false
# SEED=42  # Random seed (important when PER_QUERY="random" for reproducibility)
# REMOVE_INTERMEDIATE=false

# =============================================================================
# SCRIPT EXECUTION - Do not modify below unless you know what you're doing
# =============================================================================

set -e  # Exit on error

# -----------------------------------------------------------------------------
# Timing functions
# -----------------------------------------------------------------------------

# Function to get current timestamp
get_timestamp() {
    date '+%Y-%m-%d %H:%M:%S'
}

# Function to get elapsed time in seconds
get_elapsed() {
    local start=$1
    local end=$2
    echo "$((end - start))"
}

# Function to format seconds into human-readable format
format_time() {
    local seconds=$1
    local hours=$((seconds / 3600))
    local minutes=$(((seconds % 3600) / 60))
    local secs=$((seconds % 60))
    
    if [ $hours -gt 0 ]; then
        printf "%dh %dm %ds" $hours $minutes $secs
    elif [ $minutes -gt 0 ]; then
        printf "%dm %ds" $minutes $secs
    else
        printf "%ds" $secs
    fi
}

# Function to log message with timestamp
log_timing() {
    local message="$1"
    local timestamp=$(get_timestamp)
    echo "[$timestamp] $message" | tee -a "$TIMING_LOG"
}

# Function to merge TSV files (skip header after first file)
merge_tsv_files() {
    local output_file="$1"
    shift
    local input_files=("$@")
    
    if [ ${#input_files[@]} -eq 0 ]; then
        return
    fi
    
    # Create output directory if it doesn't exist
    mkdir -p "$(dirname "$output_file")"
    
    # Copy first file with header
    if [ -f "${input_files[0]}" ] && [ -s "${input_files[0]}" ]; then
        cat "${input_files[0]}" > "$output_file"
        
        # Append remaining files without header
        for i in $(seq 1 $((${#input_files[@]} - 1))); do
            if [ -f "${input_files[$i]}" ] && [ -s "${input_files[$i]}" ]; then
                # Skip first line (header) and append
                tail -n +2 "${input_files[$i]}" >> "$output_file"
            fi
        done
    fi
}

# Function to merge directories (copy files, merge TSV files)
merge_directories() {
    local target_dir="$1"
    local source_dir="$2"
    
    if [ ! -d "$source_dir" ]; then
        return
    fi
    
    mkdir -p "$target_dir"
    
    # Find all files in source directory
    find "$source_dir" -type f | while read -r source_file; do
        # Get relative path from source directory
        rel_path="${source_file#$source_dir/}"
        target_file="$target_dir/$rel_path"
        target_file_dir="$(dirname "$target_file")"
        
        mkdir -p "$target_file_dir"
        
        # If target file exists and both are TSV files, merge them
        if [ -f "$target_file" ] && [[ "$source_file" == *.tsv ]] && [[ "$target_file" == *.tsv ]]; then
            # Merge TSV files
            local temp_file="${target_file}.tmp"
            merge_tsv_files "$temp_file" "$target_file" "$source_file"
            mv "$temp_file" "$target_file"
        else
            # Copy file (will overwrite if exists, but that's OK for most files)
            cp "$source_file" "$target_file"
        fi
    done
}

# Initialize timing log (create output directory and log file)
mkdir -p "$OUTPUT_DIR"
> "$TIMING_LOG"  # Clear/create log file

# Initialize terminal output log if enabled
if [ "$SAVE_TERMINAL_OUTPUT" = true ] && [ -n "$TERMINAL_LOG" ]; then
    > "$TERMINAL_LOG"  # Clear/create terminal log file
    log_timing "Terminal output will be saved to: $TERMINAL_LOG"
    # Function to echo and log to terminal log file
    echo_and_log() {
        echo "$@"
        echo "$@" >> "$TERMINAL_LOG"
    }
else
    # If not logging, just echo normally
    echo_and_log() {
        echo "$@"
    }
fi

# Record script start time
SCRIPT_START=$(date +%s)
SCRIPT_START_STR=$(get_timestamp)

log_timing "================================================================================"
log_timing "mDeepFRI predict-function - Batched Execution Started"
log_timing "================================================================================"
log_timing "Start time:         $SCRIPT_START_STR"
log_timing "Input file:         $INPUT_FILE"
log_timing "Weights:            $WEIGHTS_PATH"
log_timing "Output directory:   $OUTPUT_DIR"
log_timing "Max sequences/batch: $MAX_SEQUENCES_PER_BATCH"
log_timing "Top-K:              $TOP_K"
log_timing "Databases:          ${DATABASES[@]:-PDB100 only}"
log_timing "Identity bins:      ${IDENTITY_BINS[@]}"
log_timing "Generate contacts:  ${GENERATE_CONTACTS[@]}"
log_timing "Processing modes:   ${PROCESSING_MODES[@]}"
log_timing "Threads:            $THREADS"
log_timing "Skip PDB:           $SKIP_PDB"
log_timing "================================================================================"
log_timing ""

echo_and_log "================================================================================"
echo_and_log "mDeepFRI predict-function - Batched Execution"
echo_and_log "================================================================================"
echo_and_log "Input file:        $INPUT_FILE"
echo_and_log "Weights:           $WEIGHTS_PATH"
echo_and_log "Output directory:  $OUTPUT_DIR"
echo_and_log "Max sequences/batch: $MAX_SEQUENCES_PER_BATCH"
echo_and_log "Top-K:             $TOP_K"
echo_and_log "Databases:         ${DATABASES[@]:-PDB100 only}"
echo_and_log "Identity bins:     ${IDENTITY_BINS[@]}"
echo_and_log "Generate contacts: ${GENERATE_CONTACTS[@]}"
echo_and_log "Processing modes:  ${PROCESSING_MODES[@]}"
echo_and_log "Threads:           $THREADS"
echo_and_log "Timing log:        $TIMING_LOG"
if [ "$SAVE_TERMINAL_OUTPUT" = true ] && [ -n "$TERMINAL_LOG" ]; then
    echo_and_log "Terminal log:       $TERMINAL_LOG"
fi
echo_and_log "================================================================================"
echo_and_log ""

# Check if input file exists
if [ ! -f "$INPUT_FILE" ]; then
    echo_and_log "ERROR: Input file not found: $INPUT_FILE"
    exit 1
fi

# Check if weights directory exists
if [ ! -d "$WEIGHTS_PATH" ]; then
    echo_and_log "ERROR: Weights directory not found: $WEIGHTS_PATH"
    exit 1
fi

# Count total sequences in input file
echo_and_log "Counting sequences in input file..."
TOTAL_SEQUENCES=$(grep -c "^>" "$INPUT_FILE" || echo "0")
echo_and_log "Total sequences found: $TOTAL_SEQUENCES"

if [ "$TOTAL_SEQUENCES" -eq 0 ]; then
    echo_and_log "ERROR: No sequences found in input file: $INPUT_FILE"
    exit 1
fi

# Calculate number of batches needed
NUM_BATCHES=$(( (TOTAL_SEQUENCES + MAX_SEQUENCES_PER_BATCH - 1) / MAX_SEQUENCES_PER_BATCH ))
echo_and_log "Will split into $NUM_BATCHES batch(es) of max $MAX_SEQUENCES_PER_BATCH sequences each"
echo_and_log ""

# Create temporary directory for batch files
TMP_DIR=$(mktemp -d)
trap "rm -rf $TMP_DIR" EXIT
echo_and_log "Temporary directory: $TMP_DIR"
echo_and_log ""

# Split FASTA file into batches
echo_and_log "Splitting FASTA file into batches..."
BATCH_COUNT=0
CURRENT_BATCH=0
CURRENT_BATCH_FILE=""

while IFS= read -r line; do
    if [[ "$line" =~ ^\> ]]; then
        # This is a header line
        if [ $CURRENT_BATCH -ge $MAX_SEQUENCES_PER_BATCH ]; then
            # Start a new batch
            CURRENT_BATCH=0
        fi
        
        if [ $CURRENT_BATCH -eq 0 ]; then
            # Create new batch file
            BATCH_COUNT=$((BATCH_COUNT + 1))
            CURRENT_BATCH_FILE="${TMP_DIR}/batch_${BATCH_COUNT}.fasta"
            echo_and_log "Creating batch $BATCH_COUNT: $CURRENT_BATCH_FILE"
        fi
        
        CURRENT_BATCH=$((CURRENT_BATCH + 1))
    fi
    
    # Write line to current batch file
    echo "$line" >> "$CURRENT_BATCH_FILE"
done < "$INPUT_FILE"

echo_and_log "Created $BATCH_COUNT batch file(s)"
echo_and_log ""

# Initialize merged output directory
mkdir -p "$OUTPUT_DIR"
OVERALL_EXIT_CODE=0

# Process each batch
for BATCH_NUM in $(seq 1 $BATCH_COUNT); do
    BATCH_FILE="${TMP_DIR}/batch_${BATCH_NUM}.fasta"
    BATCH_OUTPUT_DIR="${OUTPUT_DIR}/batch_${BATCH_NUM}"
    
    if [ ! -f "$BATCH_FILE" ]; then
        echo_and_log "WARNING: Batch file $BATCH_FILE not found, skipping..."
        continue
    fi
    
    BATCH_SEQ_COUNT=$(grep -c "^>" "$BATCH_FILE" || echo "0")
    echo_and_log "================================================================================"
    echo_and_log "Processing batch $BATCH_NUM/$BATCH_COUNT ($BATCH_SEQ_COUNT sequences)"
    echo_and_log "================================================================================"
    log_timing "Processing batch $BATCH_NUM/$BATCH_COUNT ($BATCH_SEQ_COUNT sequences)"
    
    # Build the command for this batch
    CMD="mDeepFRI predict-function"
    CMD="$CMD --input \"$BATCH_FILE\""
    CMD="$CMD --weights \"$WEIGHTS_PATH\""
    CMD="$CMD --output \"$BATCH_OUTPUT_DIR\""
    
    # Add top-k if set
    if [ -n "${TOP_K:-}" ]; then
        CMD="$CMD --top-k $TOP_K"
    fi
    
    # Add database paths
    if [ ${#DATABASES[@]} -gt 0 ]; then
        for db in "${DATABASES[@]}"; do
            CMD="$CMD --db-path \"$db\""
        done
    fi
    
    # Add skip-pdb flag if requested
    if [ "$SKIP_PDB" = true ]; then
        CMD="$CMD --skip-pdb"
    fi
    
    # Add processing modes
    if [ ${#PROCESSING_MODES[@]} -gt 0 ]; then
        for mode in "${PROCESSING_MODES[@]}"; do
            CMD="$CMD --processing-modes $mode"
        done
    else
        CMD="$CMD --skip-prediction"
    fi
    
    # Add identity bins
    for bin in "${IDENTITY_BINS[@]}"; do
        CMD="$CMD --identity-bin \"$bin\""
    done
    
    # Add generate contacts
    for gc in "${GENERATE_CONTACTS[@]}"; do
        CMD="$CMD --generate-contacts $gc"
    done
    
    # Add flags
    if [ "$SAVE_CMAPS" = true ]; then
        CMD="$CMD --save-cmaps"
    fi
    
    if [ "$SAVE_STRUCTURES" = true ]; then
        CMD="$CMD --save-structures"
    fi
    
    if [ "$SAVE_RAW_ALIGNMENTS" = true ]; then
        CMD="$CMD --save-raw-alignments"
    fi
    
    CMD="$CMD --threads $THREADS"
    
    # Add optional advanced parameters if they are set
    if [ -n "${SENSITIVITY:-}" ]; then
        CMD="$CMD --sensitivity $SENSITIVITY"
    fi
    
    if [ -n "${MIN_BITSCORE:-}" ]; then
        CMD="$CMD --min-bitscore $MIN_BITSCORE"
    fi
    
    if [ -n "${MAX_EVAL:-}" ]; then
        CMD="$CMD --max-eval $MAX_EVAL"
    fi
    
    if [ -n "${MIN_IDENTITY:-}" ]; then
        CMD="$CMD --min_identity $MIN_IDENTITY"
    fi
    
    if [ -n "${MIN_COVERAGE:-}" ]; then
        CMD="$CMD --min-coverage $MIN_COVERAGE"
    fi
    
    if [ -n "${ANGSTROM_CONTACT_THRESH:-}" ]; then
        CMD="$CMD --angstrom-contact-thresh $ANGSTROM_CONTACT_THRESH"
    fi
    
    if [ -n "${ALIGNMENT_GAP_OPEN:-}" ]; then
        CMD="$CMD --alignment-gap-open $ALIGNMENT_GAP_OPEN"
    fi
    
    if [ -n "${ALIGNMENT_GAP_EXTEND:-}" ]; then
        CMD="$CMD --alignment-gap-extend $ALIGNMENT_GAP_EXTEND"
    fi
    
    if [ -n "${CMAP_IDENTITY:-}" ]; then
        CMD="$CMD --cmap-identity $CMAP_IDENTITY"
    fi
    
    if [ -n "${CMAP_COVERAGE:-}" ]; then
        CMD="$CMD --cmap-coverage $CMAP_COVERAGE"
    fi
    
    if [ -n "${PER_QUERY:-}" ]; then
        CMD="$CMD --per-query $PER_QUERY"
    fi
    
    if [ -n "${DROP_SELF_HITS:-}" ]; then
        if [ "$DROP_SELF_HITS" = true ]; then
            CMD="$CMD --drop-self-hits"
        else
            CMD="$CMD --keep-self-hits"
        fi
    fi
    
    if [ -n "${SEED:-}" ]; then
        CMD="$CMD --seed $SEED"
    fi
    
    if [ -n "${REMOVE_INTERMEDIATE:-}" ] && [ "$REMOVE_INTERMEDIATE" = true ]; then
        CMD="$CMD --remove-intermediate"
    fi
    
    # Record batch execution start time
    BATCH_START=$(date +%s)
    BATCH_START_STR=$(get_timestamp)
    log_timing "Batch $BATCH_NUM started at: $BATCH_START_STR"
    
    # Execute the command and capture exit status
    set +e  # Temporarily disable exit on error
    
    if [ "$SAVE_TERMINAL_OUTPUT" = true ] && [ -n "$TERMINAL_LOG" ]; then
        echo_and_log "[BATCH $BATCH_NUM] Running command..."
        eval $CMD 2>&1 | tee -a "$TERMINAL_LOG"
        BATCH_EXIT_CODE=${PIPESTATUS[0]}
    else
        echo_and_log "[BATCH $BATCH_NUM] Running command..."
        eval $CMD
        BATCH_EXIT_CODE=$?
    fi
    
    set -e  # Re-enable exit on error
    
    # Record batch execution end time
    BATCH_END=$(date +%s)
    BATCH_END_STR=$(get_timestamp)
    BATCH_ELAPSED=$(get_elapsed $BATCH_START $BATCH_END)
    BATCH_ELAPSED_STR=$(format_time $BATCH_ELAPSED)
    
    log_timing "Batch $BATCH_NUM completed at: $BATCH_END_STR"
    log_timing "Batch $BATCH_NUM execution time: $BATCH_ELAPSED_STR ($BATCH_ELAPSED seconds)"
    log_timing "Batch $BATCH_NUM exit code: $BATCH_EXIT_CODE"
    
    if [ $BATCH_EXIT_CODE -ne 0 ]; then
        echo_and_log "WARNING: Batch $BATCH_NUM failed with exit code $BATCH_EXIT_CODE"
        OVERALL_EXIT_CODE=$BATCH_EXIT_CODE
    else
        echo_and_log "Batch $BATCH_NUM completed successfully in $BATCH_ELAPSED_STR"
        
        # Merge batch results into main output directory
        echo_and_log "Merging batch $BATCH_NUM results..."
        
        # Merge each subdirectory
        if [ -d "$BATCH_OUTPUT_DIR" ]; then
            for subdir in mmseqs2_search mmseqs2_filtered pyopal_alignments results structures contact_maps; do
                if [ -d "$BATCH_OUTPUT_DIR/$subdir" ]; then
                    merge_directories "$OUTPUT_DIR/$subdir" "$BATCH_OUTPUT_DIR/$subdir"
                fi
            done
            
            # Merge terminal output logs (append)
            if [ -f "$BATCH_OUTPUT_DIR/terminal_output.log" ]; then
                echo_and_log "" >> "$TERMINAL_LOG"
                echo_and_log "=== BATCH $BATCH_NUM OUTPUT ===" >> "$TERMINAL_LOG"
                cat "$BATCH_OUTPUT_DIR/terminal_output.log" >> "$TERMINAL_LOG"
            fi
            
            # Merge timing logs (append)
            if [ -f "$BATCH_OUTPUT_DIR/timing.log" ]; then
                echo "" >> "$TIMING_LOG"
                echo "=== BATCH $BATCH_NUM TIMING ===" >> "$TIMING_LOG"
                cat "$BATCH_OUTPUT_DIR/timing.log" >> "$TIMING_LOG"
            fi
        fi
        
        echo_and_log "Batch $BATCH_NUM results merged successfully"
    fi
    
    echo_and_log ""
done

# Record total script time
SCRIPT_END=$(date +%s)
SCRIPT_END_STR=$(get_timestamp)
SCRIPT_ELAPSED=$(get_elapsed $SCRIPT_START $SCRIPT_END)
SCRIPT_ELAPSED_STR=$(format_time $SCRIPT_ELAPSED)

log_timing ""
log_timing "================================================================================"
log_timing "Total script execution time: $SCRIPT_ELAPSED_STR ($SCRIPT_ELAPSED seconds)"
log_timing "Script started:  $SCRIPT_START_STR"
log_timing "Script finished: $SCRIPT_END_STR"
log_timing "Processed $BATCH_COUNT batch(es) with $TOTAL_SEQUENCES total sequences"
log_timing "================================================================================"

echo_and_log ""
echo_and_log "================================================================================"
if [ $OVERALL_EXIT_CODE -eq 0 ]; then
    echo_and_log "mDeepFRI completed successfully!"
else
    echo_and_log "mDeepFRI completed with exit code: $OVERALL_EXIT_CODE (some batches may have failed)"
fi
echo_and_log "Results saved to: $OUTPUT_DIR"
echo_and_log "Total execution time: $SCRIPT_ELAPSED_STR"
echo_and_log "Processed $BATCH_COUNT batch(es) with $TOTAL_SEQUENCES total sequences"
echo_and_log "Timing log saved to: $TIMING_LOG"
if [ "$SAVE_TERMINAL_OUTPUT" = true ] && [ -n "$TERMINAL_LOG" ]; then
    echo_and_log "Terminal output log saved to: $TERMINAL_LOG"
fi
echo_and_log "================================================================================"

# Clean up batch directories if desired (uncomment to enable)
# echo_and_log "Cleaning up batch directories..."
# for BATCH_NUM in $(seq 1 $BATCH_COUNT); do
#     BATCH_OUTPUT_DIR="${OUTPUT_DIR}/batch_${BATCH_NUM}"
#     if [ -d "$BATCH_OUTPUT_DIR" ]; then
#         rm -rf "$BATCH_OUTPUT_DIR"
#     fi
# done

# Exit with the overall exit code
exit $OVERALL_EXIT_CODE
