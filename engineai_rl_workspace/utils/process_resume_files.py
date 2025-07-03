import os
from shutil import copyfile
from .helpers import (
    get_load_run_path,
    get_class_and_parent_paths,
    get_resume_path_from_original_path,
    get_original_path_from_resume_path,
)
from engineai_rl_workspace import ENGINEAI_WORKSPACE_ROOT_DIR
from engineai_rl_lib.git import get_commit_hash, stash_files


def save_resume_files_from_file_paths(files, resume_dir):
    for file in files:
        resume_path = get_resume_path_from_original_path(file, resume_dir)
        resume_file_dir = resume_path.rsplit("/", 1)[0]
        if not os.path.exists(resume_file_dir):
            os.makedirs(resume_file_dir)
        copyfile(file, resume_path + ".txt")


def save_resume_classes_and_parents_files(*classes, log_dir):
    for class_name in classes:
        class_items = get_class_and_parent_paths(class_name)
        save_resume_files_from_file_paths(class_items, get_resume_dir(log_dir))


def get_resume_dir(log_dir):
    resume_dir = os.path.join(log_dir, "resume")
    return resume_dir


def restore_resume_files(args):
    log_root = get_log_root(args.exp_name, args.sub_exp_name, args.log_root)
    if args.load_run is None:
        args.load_run = -1
    log_dir, load_run = get_load_run_path(log_root, load_run=args.load_run)
    resume_dir = os.path.join(log_dir, "resume")
    original_items = []
    restore_items = []
    for root, _, files in os.walk(resume_dir):
        for file in files:
            full_file_path = os.path.join(root, file)
            restore_items.append(full_file_path)
            original_items.append(
                get_original_path_from_resume_path(
                    full_file_path.replace(".txt", ""), resume_dir
                )
            )
    for original_item, restore_item in zip(original_items, restore_items):
        copyfile(original_item, original_item + ".bak")
        copyfile(restore_item, original_item)
    import engineai_rl_workspace.exps

    return original_items


def checkout_resume_commit(log_dir, repo):
    commit_hash = get_commit_hash(log_dir)
    stash_files(repo)
    repo.git.checkout(commit_hash)


def get_log_root_and_log_dir(args):
    log_root = get_log_root(args.exp_name, args.sub_exp_name, args.log_root)
    if args.load_run is None:
        args.load_run = -1
    log_dir, load_run = get_load_run_path(log_root, load_run=args.load_run)
    return log_root, log_dir


def get_log_root(exp_name, sub_exp_name, log_root=None):
    if exp_name is None:
        raise ValueError("Please specify an experiment name!")
    if log_root is None:
        log_root = os.path.join(
            ENGINEAI_WORKSPACE_ROOT_DIR, "logs", exp_name, sub_exp_name
        )
    else:
        log_root = log_root
    return log_root


def restore_original_files(original_items):
    for original_item in original_items:
        os.rename(original_item + ".bak", original_item)
