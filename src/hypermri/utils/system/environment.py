"""
System environment utilities.

Contains functions for interacting with the system environment and version control.
"""

import git


def get_git_commit_hash():
    """
    Get the current git commit hash of the repository.

    Returns
    -------
    str
        The hex SHA of the current commit, or "unknown" if not in a git repo.
    """
    try:
        repo = git.Repo(search_parent_directories=True)
        return repo.head.object.hexsha
    except BaseException:
        return "unknown"
