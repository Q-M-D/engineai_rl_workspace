from engineai_rl_lib.files_and_dirs import (
    get_module_path_from_files_in_dir,
    get_folder_paths_from_dir,
)
import os
from engineai_rl_workspace import ENGINEAI_WORKSPACE_ROOT_DIR

file_directory = os.path.dirname(os.path.abspath(__file__))
folders = get_folder_paths_from_dir(file_directory)
for folder in folders:
    import_modules = get_module_path_from_files_in_dir(
        ENGINEAI_WORKSPACE_ROOT_DIR, folder
    )
    for module in import_modules.values():
        exec(f"import {module}")
