"""
Update the logs git info to avoid chechout.

Usage:
--exp_name <exp_name> --load_run <run_name> --commit <commit_hash>
"""
import os
import argparse
from git import Repo
from engineai_rl_workspace.scripts.sim_mujoco import get_log_root_and_log_dir
from engineai_rl_workspace import ENGINEAI_WORKSPACE_ROOT_DIR
from engineai_rl_lib.git import get_current_commit_and_branch

def update_git_info_file(log_dir, commit_hash, repo):
    """Update the git_info.txt file with the specified commit hash."""
    git_info_path = os.path.join(log_dir, "git_info.txt")
    
    if not os.path.exists(git_info_path):
        print(f"Warning: git_info.txt not found at {git_info_path}")
        return False
    
    try:
        # Get commit information
        commit = repo.commit(commit_hash)
        short_hash = commit.hexsha[:8]
        author = commit.author.name
        date = commit.committed_datetime.strftime("%Y-%m-%d %H:%M:%S%z")
        message = commit.message.strip()
        
        # Create new git info content
        new_content = f"""--- Git Repository Information ---
Repository: engineai_rl_workspace

--- Current Commit ---
Hash: {commit.hexsha}
Short Hash: {short_hash}
Author: {author}
Date: {date}
Message: {message}"""
        
        # Write to file
        with open(git_info_path, 'w') as f:
            f.write(new_content)
        
        print(f"Successfully updated git_info.txt with commit {short_hash}")
        return True
        
    except Exception as e:
        print(f"Error updating git_info.txt: {e}")
        return False

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exp_name", type=str, default="dora2_shoes_rough_ppo_contact", help="Name of the experiment"
    )
    parser.add_argument(
        "--load_run", type=str, default="2025-08-26_20-34-44", help="Name of the run to load"
    )
    parser.add_argument(
        "--commit", type=str, default="", help="Commit hash to set in the logs"
    )
    args = parser.parse_args()
    args.sub_exp_name = "default"
    args.log_root = None
    
    if args.commit == "":
        # take the last commit id
        args.commit = Repo(ENGINEAI_WORKSPACE_ROOT_DIR).head.commit.hexsha
    
    _, log_dir = get_log_root_and_log_dir(args)
    print(f"Log dir: {log_dir}")

    repo = Repo(ENGINEAI_WORKSPACE_ROOT_DIR)
    current_commit, current_branch = get_current_commit_and_branch(repo)
    print(f"Current commit: {current_commit}, branch: {current_branch}")
    
    # Validate that the provided commit exists
    try:
        target_commit = repo.commit(args.commit)
        print(f"Target commit: {target_commit.hexsha[:8]} - {target_commit.message.strip()}")
    except Exception as e:
        print(f"Error: Invalid commit hash {args.commit}: {e}")
        return
    
    # Update the git_info.txt file
    success = update_git_info_file(log_dir, args.commit, repo)
    
    if success:
        print(f"Git info updated successfully for run {args.load_run}")
    else:
        print(f"Failed to update git info for run {args.load_run}")

if __name__ == "__main__":
    main()