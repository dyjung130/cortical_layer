#!/usr/bin/env bash
#SBATCH --job-name=convert
#SBATCH --array=0-16
#SBATCH --time=00:25:00
#SBATCH --partition=RM-shared
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

# ====== LOGGING SETUP ======
LOG_DIR="./logs"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/convert_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}_$(date +%Y%m%d_%H%M%S).log"

# Function to log with timestamp
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_FILE"
}

# Function to log errors
log_error() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: $*" | tee -a "$LOG_FILE" >&2
}

# Function to log warnings
log_warn() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] WARNING: $*" | tee -a "$LOG_FILE"
}

# Function to check if file exists and log
check_file() {
    if [ -f "$1" ]; then
        log "✓ Found file: $1"
        return 0
    else
        log_error "✗ Missing file: $1"
        return 1
    fi
}

# Function to check if directory exists and log
check_dir() {
    if [ -d "$1" ]; then
        log "✓ Found directory: $1"
        return 0
    else
        log_error "✗ Missing directory: $1"
        return 1
    fi
}

# Function to log command execution
run_cmd() {
    log "Executing: $*"
    if "$@" 2>&1 | tee -a "$LOG_FILE"; then
        log "✓ Command completed successfully: $*"
        return 0
    else
        log_error "✗ Command failed: $*"
        return 1
    fi
}

# Start logging
log "========== CONVERSION SCRIPT STARTED =========="
log "Job ID: ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
log "Node: $(hostname)"
log "Working directory: $(pwd)"
log "Environment variables:"
log "  SLURM_ARRAY_TASK_ID: $SLURM_ARRAY_TASK_ID"
log "  PATH: $PATH"

export LANG=C.UTF-8
export LC_ALL=C.UTF-8

# Load required modules (adjust for your cluster)
CONDA_ENV_NAME_SURFACE_TOOLS="surfae_tools_wagstyl" #for the first part
CONDA_ENV_NAME_WB="niwrapenv" #for the second part

log "Loading anaconda3 module..."
if module load anaconda3 2>&1 | tee -a "$LOG_FILE"; then
    log "✓ anaconda3 module loaded successfully"
else
    log_error "✗ Failed to load anaconda3 module"
    exit 1
fi

log "Activating conda environment: $CONDA_ENV_NAME_SURFACE_TOOLS"
if conda activate ${CONDA_ENV_NAME_SURFACE_TOOLS} 2>&1 | tee -a "$LOG_FILE"; then
    log "✓ Conda environment activated: $CONDA_ENV_NAME_SURFACE_TOOLS"
    log "Current conda environment: $(conda info --envs | grep '*')"
else
    log_error "✗ Failed to activate conda environment: $CONDA_ENV_NAME_SURFACE_TOOLS"
    exit 1
fi

# ====== Configuration ======
SUBJECTS_DIR="../../../shared/incoming/exvivoMRI"
FREESURFER_IMG="../../../shared/singularity_images/tingsterx_brained-tools_fsl6fs7.simg"
template_dir="../../data/resample_fsaverage/"

log "Configuration:"
log "  SUBJECTS_DIR: $SUBJECTS_DIR"
log "  FREESURFER_IMG: $FREESURFER_IMG"
log "  template_dir: $template_dir"

# Check critical paths
check_dir "$SUBJECTS_DIR" || exit 1
check_file "$FREESURFER_IMG" || exit 1
check_dir "$template_dir" || exit 1

SUBJECT_LIST=($(ls -1d ${SUBJECTS_DIR}/*/)) # Get list of subject paths
log "Found ${#SUBJECT_LIST[@]} subjects total"
log "Subject list: ${SUBJECT_LIST[*]}"

if [ ${SLURM_ARRAY_TASK_ID} -ge ${#SUBJECT_LIST[@]} ]; then
    log_error "SLURM_ARRAY_TASK_ID ($SLURM_ARRAY_TASK_ID) exceeds number of subjects (${#SUBJECT_LIST[@]})"
    exit 1
fi

subj="${SUBJECT_LIST[$SLURM_ARRAY_TASK_ID]}"
subj_name=$(basename "$subj")
log "Processing subject: $subj_name (index: $SLURM_ARRAY_TASK_ID)"
log "Subject path: $subj"

# ====== Paths ======
surf_dir="${subj%/}/surf"
mri_dir="${subj%/}/mri"
out_surf_dir="./exvivo/$subj_name"
out_mri_dir="./exvivo/$subj_name/mri"

log "Subject-specific paths:"
log "  surf_dir: $surf_dir"
log "  mri_dir: $mri_dir"
log "  out_surf_dir: $out_surf_dir"
log "  out_mri_dir: $out_mri_dir"

check_dir "$surf_dir" || exit 1
check_dir "$mri_dir" || exit 1

log "Creating output directories..."
if mkdir -p "$out_surf_dir" "$out_mri_dir" 2>&1 | tee -a "$LOG_FILE"; then
    log "✓ Output directories created successfully"
else
    log_error "✗ Failed to create output directories"
    exit 1
fi

# ====== Step 1: FreeSurfer Operations ======
log "========== STEP 1: FREESURFER OPERATIONS =========="
log "Executing singularity container: $FREESURFER_IMG"

singularity exec --cleanenv "$FREESURFER_IMG" bash <<EOF 2>&1 | tee -a "$LOG_FILE"
echo "[SINGULARITY] Setting up FreeSurfer environment..."
export FS_LICENSE=\$HOME/freesurfer/license.txt

if [ ! -f "\$FS_LICENSE" ]; then
    echo "[SINGULARITY] ERROR: FreeSurfer license not found at \$FS_LICENSE" 
    exit 1
fi
echo "[SINGULARITY] FreeSurfer license found: \$FS_LICENSE"

# Get the equidistant surfaces 
for hemi in lh rh; do
    echo "[SINGULARITY] Processing hemisphere: \$hemi"
    if [ -f "$surf_dir/\${hemi}.white" ]; then
        echo "[SINGULARITY] Getting equidistant surfaces for \$hemi..."
        pial_src="$surf_dir/\${hemi}.pial"
        white_src="$surf_dir/\${hemi}.white"
        out_equi_dir="$out_surf_dir/\${hemi}.equi"
        out_qui_file="\${hemi}.equi"
        n_surfs=11
        
        echo "[SINGULARITY] Checking required surface files..."
        echo "[SINGULARITY]   pial_src: \$pial_src"
        echo "[SINGULARITY]   white_src: \$white_src"
        echo "[SINGULARITY]   out_equi_dir: \$out_equi_dir"
        
        if [ -f "\$pial_src" ] && [ -f "\$white_src" ]; then
            echo "[SINGULARITY] ✓ Both pial and white surfaces found"
            echo "[SINGULARITY] Executing generate_equivolumetric_surfaces..."

            # Set FREESURFER_HOME environment variable
            export FREESURFER_HOME=/ocean/projects/bio220042p/djung2/jupyter-notebook/090525_surface_tool_wagstyl/exvivo
            echo "[SINGULARITY] FREESURFER_HOME set to: \$FREESURFER_HOME"
            export SUBJECTS_DIR="\$FREESURFER_HOME"
            echo "[SINGULARITY] SUBJECTS_DIR set to: \$SUBJECTS_DIR"

            generate_equivolumetric_surfaces \
                --smoothing 0 \
                "\$pial_src" \
                "\$white_src" \
                \$n_surfs \
                "\$out_equi_file" \
                --software freesurfer \
                --subject_id "$subj_name"
            
            if [ \$? -eq 0 ]; then
                echo "[SINGULARITY] ✓ generate_equivolumetric_surfaces completed successfully"
            else
                echo "[SINGULARITY] ✗ generate_equivolumetric_surfaces failed"
                exit 1
            fi
            
            # Convert generated equivolumetric surfaces to GIFTI
            echo "[SINGULARITY] Converting equivolumetric surfaces to GIFTI format..."
            for surf_val in 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0; do
                native_surf="\$out_equi_dir/\${surf_val}"
                gii_surf="$out_surf_dir/\${hemi}.equi\${surf_val}.pial.surf.gii"
                if [ -f "\$native_surf" ]; then
                    echo "[SINGULARITY] Converting \$native_surf -> \$gii_surf"
                    mris_convert "\$native_surf" "\$gii_surf"
                    if [ \$? -eq 0 ]; then
                        echo "[SINGULARITY] ✓ Conversion successful: \$gii_surf"
                    else
                        echo "[SINGULARITY] ✗ Conversion failed: \$gii_surf"
                    fi
                else
                    echo "[SINGULARITY] ✗ Missing: \$native_surf"
                fi
            done
            
        else
            echo "[SINGULARITY] ✗ Missing \$pial_src or \$white_src"
        fi
    else
        echo "[SINGULARITY] ✗ Missing $surf_dir/\${hemi}.white"
    fi
done

# Convert sphere.reg files for resampling (CRITICAL ADDITION)
echo "[SINGULARITY] Converting sphere.reg files for resampling..."
for hemi in lh rh; do
    sphere_reg_file="$surf_dir/\${hemi}.sphere.reg"
    output_file="$out_surf_dir/\${hemi}.sphere.reg.surf.gii"
    
    if [ -f "\$sphere_reg_file" ]; then
        echo "[SINGULARITY] Converting \${hemi}.sphere.reg to GIFTI..."
        echo "[SINGULARITY]   Input: \$sphere_reg_file"
        echo "[SINGULARITY]   Output: \$output_file"
        
        mris_convert "\$sphere_reg_file" "\$output_file"
        if [ \$? -eq 0 ]; then
            echo "[SINGULARITY] ✓ Successfully converted \${hemi}.sphere.reg"
        else
            echo "[SINGULARITY] ✗ Failed to convert \${hemi}.sphere.reg"
        fi
    else
        echo "[SINGULARITY] ✗ Missing \$sphere_reg_file - resampling will fail"
    fi
done


# Convert other FreeSurfer surfaces to GIFTI format for metadata setting
for hemi in lh rh; do
    for surf in white pial inflated sphere sphere.reg; do
        if [ -f "$surf_dir/\${hemi}.\${surf}" ]; then
            echo "Converting \${hemi}.\${surf} to GIFTI..."
            mris_convert "$surf_dir/\${hemi}.\${surf}" "$out_surf_dir/\${hemi}.\${surf}.surf.gii"
        fi
    done
done
echo "[SINGULARITY] FreeSurfer operations completed"
EOF

if [ $? -eq 0 ]; then
    log "✓ Singularity FreeSurfer operations completed successfully"
else
    log_error "✗ Singularity FreeSurfer operations failed"
    exit 1
fi

# ====== Step 2: Set Workbench Metadata & Resample ======
log "========== STEP 2: WORKBENCH METADATA & RESAMPLE =========="

log "Switching conda environments..."
eval "$(conda shell.bash hook)"
conda deactivate 2>&1 | tee -a "$LOG_FILE"

log "Activating conda environment: $CONDA_ENV_NAME_WB"
if conda activate ${CONDA_ENV_NAME_WB} 2>&1 | tee -a "$LOG_FILE"; then
    log "✓ Conda environment activated: $CONDA_ENV_NAME_WB"
    log "Current conda environment: $(conda info --envs | grep '*')"
else
    log_error "✗ Failed to activate conda environment: $CONDA_ENV_NAME_WB"
    exit 1
fi

log "Setting metadata and resampling for subject: $subj_name"

# Define associative arrays
declare -A hemi_structure=(["lh"]="CORTEX_LEFT" ["rh"]="CORTEX_RIGHT")
declare -A surface_type=(
    ["white"]="ANATOMICAL" ["pial"]="ANATOMICAL" ["inf"]="ANATOMICAL"
    ["inflated"]="INFLATED" ["orig"]="ANATOMICAL" ["smoothwm"]="ANATOMICAL"
    ["sphere"]="SPHERICAL" ["sphere.reg"]="SPHERICAL"
)
declare -A secondary_type=(
    ["white"]="GRAY_WHITE" ["pial"]="PIAL" ["inf"]="MIDTHICKNESS"
    ["inflated"]="INVALID" ["orig"]="GRAY_WHITE" ["smoothwm"]="GRAY_WHITE"
    ["sphere"]="INVALID" ["sphere.reg"]="INVALID"
)

log "Defined associative arrays for hemisphere structures and surface types"

# Define surface files to process
surf_files=("0.0" "0.1" "0.2" "0.3" "0.4" "0.5" "0.6" "0.7" "0.8" "0.9" "1.0")
log "Processing surface files: ${surf_files[*]}"

for hemi in lh rh; do
    log "---------- Processing hemisphere: $hemi ----------"
    structure="${hemi_structure[$hemi]}"
    hemi_upper=$(echo "${hemi:0:1}" | tr 'a-z' 'A-Z')
    
    log "Hemisphere structure: $structure"
    log "Hemisphere upper: $hemi_upper"
    
    # ----- Set metadata for equivolumetric surfaces -----
    log "Setting metadata for equivolumetric surfaces..."
    for surf in "${surf_files[@]}"; do
        surf_file="$out_surf_dir/${hemi}.equi${surf}.pial.surf.gii"
        if check_file "$surf_file"; then
            log "Setting metadata for: $surf_file"
            if run_cmd wb_command -set-structure "$surf_file" "$structure" \
                -surface-type "ANATOMICAL" \
                -surface-secondary-type "PIAL"; then
                log "✓ Metadata set successfully for: $surf_file"
            else
                log_error "✗ Failed to set metadata for: $surf_file"
            fi
        else
            log_warn "Skipping metadata for missing file: $surf_file"
        fi
    done
    
    # ----- Check if sphere.reg exists for resampling -----
    sphere_in="$out_surf_dir/${hemi}.sphere.reg.surf.gii"
    if ! check_file "$sphere_in"; then
        log_error "Missing sphere registration file: $sphere_in"
        log_error "Cannot proceed with resampling for $hemi hemisphere"
        continue
    fi
    
    # ----- Template files -----
    sphere_out="${template_dir}/fs_LR-deformed_to-fsaverage.${hemi_upper}.sphere.32k_fs_LR.surf.gii"
    area_target="${template_dir}/fs_LR.${hemi_upper}.midthickness_va_avg.32k_fs_LR.shape.gii"
    
    log "Template files:"
    log "  sphere_out: $sphere_out"
    log "  area_target: $area_target"
    
    if ! check_file "$sphere_out"; then
        log_error "Missing template sphere: $sphere_out"
        continue
    fi
   
    # ----- Resample equivolumetric surfaces -----
    log "Resampling equivolumetric surfaces..."
    for surf in "${surf_files[@]}"; do
        in_file="$out_surf_dir/${hemi}.equi${surf}.pial.surf.gii"
        out_file="$out_surf_dir/${hemi}.equi${surf}.32k_fs_LR.pial.surf.gii"
        if check_file "$in_file"; then
            log "Resampling: $in_file -> $out_file"
            if run_cmd wb_command -surface-resample "$in_file" "$sphere_in" "$sphere_out" BARYCENTRIC "$out_file"; then
                log "✓ Resampling successful: $out_file"
            else
                log_error "✗ Resampling failed: $out_file"
            fi
        else
            log_warn "Skipping resampling for missing file: $in_file"
        fi
    done
done

log "========== PROCESSING COMPLETE =========="
log "Processing complete for subject: $subj_name"
log "Log file saved to: $LOG_FILE"
log "Script execution time: $SECONDS seconds"

# Final summary
log "========== SUMMARY =========="
log "Generated files in $out_surf_dir:"
if ls -la "$out_surf_dir" 2>&1 | tee -a "$LOG_FILE"; then
    log "✓ Output directory listing completed"
else
    log_warn "Could not list output directory contents"
fi

log "========== SCRIPT COMPLETED =========="