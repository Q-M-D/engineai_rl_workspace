import os
import subprocess
import pathlib


def store_code_state(logdir, repo) -> list:
    # get the name of the repository
    repo_name = pathlib.Path(repo.working_dir).name
    git_info_file = os.path.join(logdir, "git_info.txt")
    # check if the diff file already exists
    if os.path.isfile(git_info_file):
        return
    # write the diff file
    with open(git_info_file, "x", encoding="utf-8") as f:
        content = ["--- Git Repository Information ---", f"Repository: {repo_name}"]
        current_commit = repo.head.commit
        content.extend(
            [
                f"\n--- Current Commit ---",
                f"Hash: {current_commit.hexsha}",
                f"Short Hash: {current_commit.hexsha[:8]}",
                f"Author: {current_commit.author}",
                f"Date: {current_commit.authored_datetime}",
                f"Message: {current_commit.message.strip()}",
            ]
        )
        f.write("\n".join(content))


def get_commit_hash(logdir):
    file_path = os.path.join(logdir, "git_info.txt")
    with open(file_path) as file:
        for line in file:
            if line.startswith("Hash:"):
                commit_hash = line.split()[1]
    return commit_hash


def get_current_commit_and_branch(repo):
    current_commit = repo.head.commit.hexsha
    branches = [head for head in repo.heads if head.commit.hexsha == current_commit]
    try:
        current_branch = repo.active_branch
    except:
        current_branch = None
    if current_branch in branches:
        return current_commit, current_branch
    else:
        return current_commit, None


def checkout_commit_or_branch(repo, commit, branch):
    if branch is not None:
        repo.git.checkout(branch, force=True)
    else:
        repo.git.checkout(commit, force=True)


def save_patch(file):
    result = subprocess.run(["git", "diff"], capture_output=True, text=True)

    if result.returncode == 0:
        # Write the diff output to the patch file
        with open(file, "w") as f:
            f.write(result.stdout)
        print(f"Patch saved to {file}")
    else:
        print("Error running git diff:")
        print(result.stderr)


def apply_patch(file, repo_path):
    subprocess.run(["git", "apply", file], cwd=repo_path)


def stash_files(repo):
    if repo.is_dirty(untracked_files=True):
        # Stash changes
        repo.git.stash("save")


def unstash_files(repo):
    if repo.git.stash("list"):
        repo.git.stash("pop")
