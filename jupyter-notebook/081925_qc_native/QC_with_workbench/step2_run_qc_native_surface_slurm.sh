#!/bin/bash
#SBATCH -N 1
#SBATCH --job-name=surface_analysis
#SBATCH --output=logs/surface_analysis_%A_%a.out
#SBATCH --error=logs/surface_analysis_%A_%a.err
#SBATCH --array=1-17 # Replace N with actual number of subjects
#SBATCH --time=00:30:00
#SBATCH --ntasks-per-node=10
#SBATCH --partition=RM-shared

CONDA_ENV_NAME="niwrapenv" #this has workbench 
# Set base path where all subject folders are located
'''
#For exvivo old data quality control
BASE_PATH="/ocean/projects/bio220042p/djung2/data/exvivo_mod"
PYTHON_SCRIPT="/ocean/projects/bio220042p/djung2/code/091025_exvivo_quality_control_wb/quality_control_native_surface_slurm_wb.py"
# Set paths required for workbench analysis
WB_SURFACES_PATH="/ocean/projects/bio220042p/djung2/data/exvivo_mod/exvivo_voxel_up_and_down/"
'''
#For exvivo new data quality control
BASE_PATH="/ocean/projects/bio220042p/djung2/data/exvivo_final"
PYTHON_SCRIPT="/ocean/projects/bio220042p/djung2/code/091025_exvivo_quality_control_wb/quality_control_native_surface_slurm_wb.py"
# Set paths required for workbench analysis
WB_SURFACES_PATH="/ocean/projects/bio220042p/djung2/data/exvivo_final/surface_voxel_up_and_down/"
# Load required modules (adjust for your cluster)
module load anaconda3
conda activate ${CONDA_ENV_NAME}    

# Create logs directory if it doesn't exist
#mkdir -p logs

# Validate paths exist
if [[ ! -d "${BASE_PATH}" ]]; then
    echo "ERROR: Base path does not exist: ${BASE_PATH}"
    exit 1
fi

if [[ ! -f "${PYTHON_SCRIPT}" ]]; then
    echo "ERROR: Python script not found: ${PYTHON_SCRIPT}"
    exit 1
fi

# Get all subject directories
SUBJECTS=($(find ${BASE_PATH} -maxdepth 1 -type d -name "*" | grep -v "^${BASE_PATH}$" | sort))

# Validate array bounds
if [[ ${SLURM_ARRAY_TASK_ID} -gt ${#SUBJECTS[@]} ]]; then
    echo "ERROR: Array task ID (${SLURM_ARRAY_TASK_ID}) exceeds number of subjects (${#SUBJECTS[@]})"
    exit 1
fi

# Get current subject based on SLURM array task ID
SUBJECT_DIR=${SUBJECTS[$((SLURM_ARRAY_TASK_ID - 1))]}
SUBJECT_NAME=$(basename ${SUBJECT_DIR})

echo "Processing subject: ${SUBJECT_NAME}"
echo "Subject directory: ${SUBJECT_DIR}"
echo "SLURM Job ID: ${SLURM_JOB_ID}"
echo "Array Task ID: ${SLURM_ARRAY_TASK_ID}"

# Check if required files exist for each hemisphere
HEMISPHERES=("lh" "rh")
echo "Checking required files..."

# Always check if native.nii exists first
if [[ ! -f "${SUBJECT_DIR}/mri/native.nii.gz" ]]; then
    echo "ERROR: Required file not found: ${SUBJECT_DIR}/mri/native.nii.gz"
    exit 1
else
    echo "Found: ${SUBJECT_DIR}/mri/native.nii.gz"
fi

# Check each hemisphere
PROCESSED_COUNT=0
for hemi in "${HEMISPHERES[@]}"; do
    # FIXED: Added missing coordinate files
    HEMI_FILES=(
        "${SUBJECT_DIR}/${hemi}.inf.32k_fs_LR.surfnorm.func.gii"
        "${SUBJECT_DIR}/${hemi}.white.32k_fs_LR.surfnorm.func.gii" 
        "${SUBJECT_DIR}/${hemi}.pial.32k_fs_LR.surfnorm.func.gii"
        "${SUBJECT_DIR}/${hemi}.inf.32k_fs_LR.coord.func.gii"
        "${SUBJECT_DIR}/${hemi}.white.32k_fs_LR.coord.func.gii"
        "${SUBJECT_DIR}/${hemi}.pial.32k_fs_LR.coord.func.gii"
    )
    
    # Check if files exist for this hemisphere
    HEMI_EXISTS=true
    MISSING_FILES=()
    for file in "${HEMI_FILES[@]}"; do
        if [[ ! -f "$file" ]]; then
            HEMI_EXISTS=false
            MISSING_FILES+=("$file")
        fi
    done
    
    # If files exist for this hemisphere, process it
    if [[ "$HEMI_EXISTS" = true ]]; then
        echo "Processing hemisphere: ${hemi}"
        for file in "${HEMI_FILES[@]}"; do
            echo "Found: $file"
        done
        
        # FIXED: Better error handling for directory change and script execution
        echo "Changing to script directory..."
        if ! cd $(dirname ${PYTHON_SCRIPT}); then
            echo "ERROR: Failed to change to script directory: $(dirname ${PYTHON_SCRIPT})"
            exit 1
        fi
        
        echo "Running: python3 ${PYTHON_SCRIPT} ${SUBJECT_NAME} ${BASE_PATH} ${hemi} ${WB_SURFACES_PATH}"
        if python3 ${PYTHON_SCRIPT} ${SUBJECT_NAME} ${BASE_PATH} ${hemi} ${WB_SURFACES_PATH}; then
            echo "SUCCESS: Completed processing for subject: ${SUBJECT_NAME}, hemisphere: ${hemi}"
            ((PROCESSED_COUNT++))
        else
            echo "ERROR: Python script failed for subject: ${SUBJECT_NAME}, hemisphere: ${hemi}"
            exit 1
        fi
    else
        echo "Skipping ${hemi} hemisphere - required files not found:"
        printf '  Missing: %s\n' "${MISSING_FILES[@]}"
    fi
done

# Final status check
if [[ ${PROCESSED_COUNT} -eq 0 ]]; then
    echo "WARNING: No hemispheres were processed for subject ${SUBJECT_NAME}"
    exit 1
else
    echo "SUCCESS: Processed ${PROCESSED_COUNT} hemisphere(s) for subject ${SUBJECT_NAME}"
fi
