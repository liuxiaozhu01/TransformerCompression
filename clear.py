import os
import glob

def delete_specific_files(parent_directory, extensions):
    # Walk through all subdirectories of the parent directory
    for root, dirs, files in os.walk(parent_directory):
        # For each file extension
        for extension in extensions:
            # Find all files with the specified extension
            for file in glob.glob(os.path.join(root, f'*.{extension}')):
                # Delete the file
                os.remove(file)

# Call the function with the list of extensions
delete_specific_files('/root/home/workspace/GPT_StrucPGPruning/prune_channel/exp/summary', ['pt', 'model'])