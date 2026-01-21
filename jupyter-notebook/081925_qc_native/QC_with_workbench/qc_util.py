import os
import re

# Example usage:
# calculate_surfnorm_and_coords('exvivo2', "/Users/dennis.jungchildmind.org/Desktop/To-be-uploaded-to-CMI-server/data/thickness_data/")

def calculate_surfnorm_and_coords(data_type, base_path):
    """
    Calculate surface normals and coordinates for surface files found inside folders.

    Args:
        data_type (str): e.g., "exvivo2", "bigbrain", or "exvivo"
        base_path (str): Base path containing the data directories

    Raises:
        Exception: If wb_command is not found or if data_type is invalid.
    """

    if 'exvivo' in data_type.lower():
        path = f"{base_path}/{data_type}"
        layer_types = ["pial", "white", "inf"]
        regex_postfx = ".32k_fs_LR.surf.gii"
    elif 'bigbrain' in data_type.lower():
        path = f"{base_path}/BigBrain/PlosBiology2020gii/"
        layer_types = ["layer3"]
        regex_postfx = "_327680.surf.gii"
    else:
        raise ValueError(f"Invalid DATA_TYPE: {data_type}")

    folders = [f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]

    for folder in folders:
        for layer in layer_types:
            regex_target = f"{layer}.*{re.escape(regex_postfx)}"
            print(regex_target)
            files = os.listdir(os.path.join(path, folder))
            matching_files = [f for f in files if re.search(regex_target, f)]
            print(matching_files)
            if matching_files:
                for file in matching_files:
                    surface_file = os.path.join(path, folder, file)
                    print(f"Found {surface_file} in {folder}")
                    
                    surfnorm_output = surface_file.replace('.surf.gii', '.surfnorm.func.gii')
                    print(f"Surface normal output file: {surfnorm_output}")
                    cmd_surface_normals = f"wb_command -surface-normals {surface_file} {surfnorm_output}"
                    
                    coord_output = surface_file.replace('.surf.gii', '.coord.func.gii')
                    print(f"Coordinate output file: {coord_output}")
                    cmd_coordinates = f"wb_command -surface-coordinates-to-metric {surface_file} {coord_output}"

                    # Check if wb_command exists
                    if os.system("which wb_command > /dev/null") != 0:
                        raise Exception("wb_command not found in PATH")

                    # Execute commands and check return codes
                    if os.system(cmd_surface_normals) != 0:
                        print(f"Error calculating surface normals for {surface_file}")
                        continue

                    if os.system(cmd_coordinates) != 0:
                        print(f"Error calculating coordinates for {surface_file}")
                        continue

                    print(f"Successfully calculated surface normals and coordinates for {folder}")