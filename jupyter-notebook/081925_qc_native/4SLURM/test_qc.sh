#!/bin/bash

# Set base path where all subject folders are located
BASE_PATH="/ocean/projects/bio220042p/djung2/data/exvivo_mod"

echo "=========================================="
echo "SUBJECT AND HEMISPHERE CHECKER"
echo "=========================================="
echo "Base path: ${BASE_PATH}"
echo ""

# Check if base path exists
if [[ ! -d "${BASE_PATH}" ]]; then
    echo "ERROR: Base path does not exist: ${BASE_PATH}"
    exit 1
fi

echo "✓ Base path exists"
echo ""

# Get all subject directories
SUBJECTS=($(find ${BASE_PATH} -maxdepth 1 -type d -name "*" | grep -v "^${BASE_PATH}$" | sort))

echo "Found ${#SUBJECTS[@]} potential subject directories:"
echo "------------------------------------------"

if [[ ${#SUBJECTS[@]} -eq 0 ]]; then
    echo "No subject directories found in ${BASE_PATH}"
    exit 1
fi

# List all subjects
for i in "${!SUBJECTS[@]}"; do
    echo "$((i+1)): $(basename ${SUBJECTS[i]})"
done

echo ""
echo "=========================================="
echo "CHECKING REQUIRED FILES FOR EACH SUBJECT"
echo "=========================================="

VALID_LH=0
VALID_RH=0
VALID_BOTH=0
INVALID_SUBJECTS=0
TOTAL_VALID_HEMISPHERES=0

# Function to check hemisphere files
check_hemisphere() {
    local subject_dir=$1
    local hemi=$2
    
    local files=(
        "${subject_dir}/mri/native.nii"
        "${subject_dir}/${hemi}.inf.32k_fs_LR.surfnorm.func.gii"
        "${subject_dir}/${hemi}.white.32k_fs_LR.surfnorm.func.gii" 
        "${subject_dir}/${hemi}.pial.32k_fs_LR.surfnorm.func.gii"
    )
    
    local missing=0
    for file in "${files[@]}"; do
        if [[ ! -f "$file" ]]; then
            missing=$((missing + 1))
        fi
    done
    
    return $missing
}

for SUBJECT_DIR in "${SUBJECTS[@]}"; do
    SUBJECT_NAME=$(basename ${SUBJECT_DIR})
    echo ""
    echo "Subject: ${SUBJECT_NAME}"
    echo "Path: ${SUBJECT_DIR}"
    
    # Check native.nii first (required for both hemispheres)
    if [[ ! -f "${SUBJECT_DIR}/mri/native.nii" ]]; then
        echo "  ✗ MISSING: mri/native.nii (required for all processing)"
        echo "  → Status: INVALID (no native volume)"
        INVALID_SUBJECTS=$((INVALID_SUBJECTS + 1))
        continue
    else
        echo "  ✓ Found: mri/native.nii"
    fi
    
    # Check left hemisphere
    echo "  Checking left hemisphere (lh):"
    check_hemisphere "$SUBJECT_DIR" "lh"
    LH_MISSING=$?
    
    if [[ $LH_MISSING -eq 0 ]]; then
        echo "    ✓ LH: All files present"
        LH_VALID=true
        VALID_LH=$((VALID_LH + 1))
        TOTAL_VALID_HEMISPHERES=$((TOTAL_VALID_HEMISPHERES + 1))
    else
        echo "    ✗ LH: $LH_MISSING files missing"
        LH_VALID=false
    fi
    
    # Check right hemisphere
    echo "  Checking right hemisphere (rh):"
    check_hemisphere "$SUBJECT_DIR" "rh"
    RH_MISSING=$?
    
    if [[ $RH_MISSING -eq 0 ]]; then
        echo "    ✓ RH: All files present"
        RH_VALID=true
        VALID_RH=$((VALID_RH + 1))
        TOTAL_VALID_HEMISPHERES=$((TOTAL_VALID_HEMISPHERES + 1))
    else
        echo "    ✗ RH: $RH_MISSING files missing"
        RH_VALID=false
    fi
    
    # Determine overall status
    if [[ $LH_VALID == true && $RH_VALID == true ]]; then
        echo "  → Status: VALID (both hemispheres)"
        VALID_BOTH=$((VALID_BOTH + 1))
    elif [[ $LH_VALID == true || $RH_VALID == true ]]; then
        if [[ $LH_VALID == true ]]; then
            echo "  → Status: VALID (left hemisphere only)"
        else
            echo "  → Status: VALID (right hemisphere only)"
        fi
    else
        echo "  → Status: INVALID (no valid hemispheres)"
        INVALID_SUBJECTS=$((INVALID_SUBJECTS + 1))
    fi
done

echo ""
echo "=========================================="
echo "SUMMARY"
echo "=========================================="
echo "Total subjects found: ${#SUBJECTS[@]}"
echo "Subjects with valid LH: ${VALID_LH}"
echo "Subjects with valid RH: ${VALID_RH}"
echo "Subjects with both hemispheres: ${VALID_BOTH}"
echo "Total valid hemispheres: ${TOTAL_VALID_HEMISPHERES}"
echo "Invalid subjects: ${INVALID_SUBJECTS}"
echo ""

if [[ $TOTAL_VALID_HEMISPHERES -gt 0 ]]; then
    echo "✓ Ready to run SLURM job with --array=1-${TOTAL_VALID_HEMISPHERES}"
    echo ""
    echo "Valid hemisphere processing list:"
    echo "--------------------------------"
    
    COUNTER=1
    for SUBJECT_DIR in "${SUBJECTS[@]}"; do
        SUBJECT_NAME=$(basename ${SUBJECT_DIR})
        
        # Skip if no native.nii
        if [[ ! -f "${SUBJECT_DIR}/mri/native.nii" ]]; then
            continue
        fi
        
        # Check LH
        check_hemisphere "$SUBJECT_DIR" "lh"
        if [[ $? -eq 0 ]]; then
            echo "  ${COUNTER}: ${SUBJECT_NAME} (lh)"
            COUNTER=$((COUNTER + 1))
        fi
        
        # Check RH
        check_hemisphere "$SUBJECT_DIR" "rh"
        if [[ $? -eq 0 ]]; then
            echo "  ${COUNTER}: ${SUBJECT_NAME} (rh)"
            COUNTER=$((COUNTER + 1))
        fi
    done
else
    echo "✗ No valid hemispheres found. Cannot run SLURM job."
fi

echo ""
echo "Note: Each array job will process one hemisphere."
echo "Subjects with both hemispheres will have 2 jobs."
echo "=========================================="