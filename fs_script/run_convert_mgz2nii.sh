#!/usr/bin/env bash
#SBATCH --job-name=convert
#SBATCH --array=0-16
#SBATCH --time=00:25:00
#SBATCH --partition=RM-shared
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

# Create logs directory FIRST (before any other commands)
#mkdir -p logs

export LANG=C.UTF-8
export LC_ALL=C.UTF-8

echo "=== JOB START ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Date: $(date)"

# ====== Configuration ======
SUBJECTS_DIR="../../shared/incoming/exvivoMRI"
FREESURFER_IMG="../../shared/singularity_images/tingsterx_brained-tools_fsl6fs7.simg"
OUTPUT_BASE_DIR="../exvivo_mod"
TARGET_FILE="aparc+aseg.upsampled.mgz"

# Create base output directory
mkdir -p "$OUTPUT_BASE_DIR"

# Get list of subject paths and validate array index
SUBJECT_LIST=($(ls -1d ${SUBJECTS_DIR}/*/))
TOTAL_SUBJECTS=${#SUBJECT_LIST[@]}

echo "Total subjects found: $TOTAL_SUBJECTS"
echo "Current array task ID: $SLURM_ARRAY_TASK_ID"

# Check if array task ID is valid
if [ $SLURM_ARRAY_TASK_ID -ge $TOTAL_SUBJECTS ]; then
    echo "ERROR: Array task ID $SLURM_ARRAY_TASK_ID exceeds number of subjects ($TOTAL_SUBJECTS). Exiting."
    exit 1
fi

subj="${SUBJECT_LIST[$SLURM_ARRAY_TASK_ID]}"
subj_name=$(basename "$subj")
echo "Processing subject: $subj_name"

# ====== Paths ======
mri_dir="${subj%/}/mri"
input_file="$mri_dir/$TARGET_FILE"
out_mri_dir="$OUTPUT_BASE_DIR/$subj_name/mri"
output_file="$out_mri_dir/aparc+aseg.upsampled.nii"

echo "Input file: $input_file"
echo "Output file: $output_file"

mkdir -p "$out_mri_dir"

# Check if input directory exists
if [ ! -d "$mri_dir" ]; then
    echo "ERROR: Input directory $mri_dir does not exist"
    exit 1
fi

# Check if the specific native.mgz file exists
if [ ! -f "$input_file" ]; then
    echo "ERROR: Input file $input_file does not exist"
    echo "Files in $mri_dir:"
    ls -la "$mri_dir"
    exit 1
fi

# Check if Singularity image exists
if [ ! -f "$FREESURFER_IMG" ]; then
    echo "ERROR: Singularity image $FREESURFER_IMG does not exist"
    exit 1
fi

echo "Input file size: $(du -h "$input_file" | cut -f1)"
echo "Starting conversion..."

# ====== Conversion ======
singularity exec --cleanenv "$FREESURFER_IMG" bash <<EOF
export FS_LICENSE=\$HOME/freesurfer/license.txt

echo "Inside Singularity container"

# Check if FreeSurfer license exists
if [ ! -f "\$FS_LICENSE" ]; then
    echo "ERROR: FreeSurfer license not found at \$FS_LICENSE"
    exit 1
fi

echo "Running: mri_convert $input_file $output_file"
mri_convert "$input_file" "$output_file"

if [ \$? -eq 0 ]; then
    echo "SUCCESS: Conversion completed"
    echo "Output file size: \$(du -h "$output_file" | cut -f1)"
else
    echo "ERROR: mri_convert failed"
    exit 1
fi
EOF

# Verify the output file was created
if [ -f "$output_file" ]; then
    echo "SUCCESS: File created at $output_file"
    ls -lh "$output_file"
    echo "Job completed successfully for subject: $subj_name"
else
    echo "ERROR: Output file was not created"
    exit 1
fi

echo "=== JOB END ==="
