from typing import List, Optional, Any, Dict, Union
import base64
import os
import requests

from github import Github, GithubException, Auth

"""github_tools.py
Utility‑grade wrappers around the ten most‑used GitHub REST endpoints.

Pattern:
  • Minimal typed signatures that mirror the REST parameters.
  • Shared global `gh: Github` client (call `init_github()` once at start‑up).
  • Consistent error handling → raises RuntimeError with a helpful message.

Functions:
  1. list_my_repos()                   – GET   /user/repos
  2. read_file()                       – GET   /repos/{owner}/{repo}/contents/{path}
  3. commit_file()                     – PUT   /repos/{owner}/{repo}/contents/{path}
  4. list_issues()                     – GET   /repos/{owner}/{repo}/issues
  5. open_issue()                      – POST  /repos/{owner}/{repo}/issues
  6. list_prs()                        – GET   /repos/{owner}/{repo}/pulls
  7. create_pr()                       – POST  /repos/{owner}/{repo}/pulls
  8. list_commits()                    – GET   /repos/{owner}/{repo}/commits
  9. search_repos()                    – GET   /search/repositories
 10. search_issues()                   – GET   /search/issues
"""

# --------------------------------------------------------------------------- #
# Globals & configuration                                                     #
# --------------------------------------------------------------------------- #

gh: Optional[Github] = None               # Will hold the authenticated client
DEFAULT_OWNER: str = os.getenv("GITHUB_REPO_OWNER", "Shreyas-Yadav")
DEFAULT_PER_PAGE: int = 100

def init_github(token: str) -> None:
    """Initialise the global PyGitHub client (run once)."""
    global gh
    gh = Github(auth=Auth.Token(token))


def _client() -> Github:
    if gh is None:
        raise RuntimeError("GitHub client not initialized — call init_github(<token>) first")
    return gh

# --------------------------------------------------------------------------- #
# 1. GET /user/repos                                                          #
# --------------------------------------------------------------------------- #

def list_my_repos(*args) -> List[str]:
    """Return the full names ("owner/repo") of all repos visible to the user."""
    try:
        return [r.full_name for r in _client().get_user().get_repos()]
    except GithubException as e:
        raise RuntimeError(f"GitHub API error listing user repos: {e.data.get('message', str(e))}")
    except Exception as e:
        raise RuntimeError(f"Unexpected error listing user repos: {e}")

# --------------------------------------------------------------------------- #
# 2. GET /repos/{owner}/{repo}/contents/{path}                                #
# --------------------------------------------------------------------------- #

def read_file(repo: str, path: str, branch: str = "main", owner: Optional[str] = None) -> str:
    """Fetch a text/binary file and return its decoded content as str."""
    repo_owner = owner or DEFAULT_OWNER
    try:
        repository = _client().get_repo(f"{repo_owner}/{repo}")
        blob = repository.get_contents(path, ref=branch)
        return base64.b64decode(blob.content).decode()
    except GithubException as e:
        raise RuntimeError(
            f"GitHub API error reading {path} in {repo_owner}/{repo}@{branch}: "
            f"{e.data.get('message', str(e))}"
        )
    except Exception as e:
        raise RuntimeError(f"Unexpected error reading file from {repo_owner}/{repo}: {e}")


def close_issue(repo: str, issue_number: int, owner: Optional[str] = None) -> str:
    """
    Close an existing GitHub issue.

    Args:
        repo (str): Repository name (e.g. "my-repo").
        issue_number (int): The number of the issue to close.
        owner (str, optional): Repository owner or organization. 
                               Defaults to DEFAULT_OWNER.

    Returns:
        str: Confirmation message.

    Raises:
        RuntimeError: On API errors or unexpected failures.
    """
    repo_owner = owner or DEFAULT_OWNER
    try:
        repository = _client().get_repo(f"{repo_owner}/{repo}")
        issue = repository.get_issue(number=issue_number)
        if issue.state.lower() == "closed":
            return f"Issue #{issue_number} is already closed."
        issue.edit(state="closed")
        return f"Issue #{issue_number} has been closed successfully."
    except GithubException as e:
        msg = e.data.get("message") if hasattr(e, "data") else str(e)
        raise RuntimeError(f"GitHub API error closing issue #{issue_number}: {msg}")
    except Exception as e:
        raise RuntimeError(f"Unexpected error closing issue #{issue_number}: {e}")

# --------------------------------------------------------------------------- #
# 3. PUT /repos/{owner}/{repo}/contents/{path}                                #
# --------------------------------------------------------------------------- #

def commit_file(
    repo: str,
    path: str,
    message: str,
    content: str,
    branch: str = "main",
    owner: Optional[str] = None,
) -> str:
    """Create or update *path* with *content* and return the new blob SHA."""
    repo_owner = owner or DEFAULT_OWNER
    try:
        repository = _client().get_repo(f"{repo_owner}/{repo}")
        try:
            existing = repository.get_contents(path, ref=branch)
            result = repository.update_file(path, message, content, sha=existing.sha, branch=branch)
        except GithubException as inner:
            if inner.status == 404:  # file absent → create
                result = repository.create_file(path, message, content, branch=branch)
            else:
                raise
        return result["content"].sha
    except GithubException as e:
        raise RuntimeError(
            f"GitHub API error committing {path} in {repo_owner}/{repo}: "
            f"{e.data.get('message', str(e))}"
        )
    except Exception as e:
        raise RuntimeError(f"Unexpected error committing file to {repo_owner}/{repo}: {e}")

# --------------------------------------------------------------------------- #
# 4. GET /repos/{owner}/{repo}/issues                                         #
# --------------------------------------------------------------------------- #

def list_issues(
    repo: str,
    owner: Optional[str] = None,
    state: str = "open"
) -> List[Dict[str, Union[int, str]]]:
    """
    Retrieve all issues from a GitHub repository.

    Args:
        repo (str): Repository name (e.g. "my-repo").
        owner (str, optional): Repository owner. Defaults to DEFAULT_OWNER.
        state (str, optional): Issue state filter: "open", "closed", or "all". Defaults to "open".

    Returns:
        List[Dict[str, Union[int, str]]]: A list of dicts each with:
            - number (int): Issue number
            - title (str): Issue title

    Raises:
        RuntimeError: On API errors or unexpected failures.
    """
    repo_owner = owner or DEFAULT_OWNER
    try:
        repository = _client().get_repo(f"{repo_owner}/{repo}")
        issues = repository.get_issues(state=state)
        return [{"number": issue.number, "title": issue.title} for issue in issues]
    except GithubException as e:
        msg = e.data.get("message") if hasattr(e, "data") else str(e)
        raise RuntimeError(f"GitHub API error listing issues: {msg}")
    except Exception as e:
        raise RuntimeError(f"Unexpected error listing issues: {e}")
    
# --------------------------------------------------------------------------- #
# 5. POST /repos/{owner}/{repo}/issues                                        #
# --------------------------------------------------------------------------- #

def open_issue(
    repo: str,
    title: str,
    body: str = "",
    labels: Optional[List[str]] = None,
    owner: Optional[str] = None,
) -> int:
    """Create a new issue and return its number."""
    repo_owner = owner or DEFAULT_OWNER
    try:
        repository = _client().get_repo(f"{repo_owner}/{repo}")
        issue = repository.create_issue(title=title, body=body, labels=labels or [])
        return issue.number
    except GithubException as e:
        raise RuntimeError(
            f"GitHub API error opening issue in {repo_owner}/{repo}: {e.data.get('message', str(e))}"
        )
    except Exception as e:
        raise RuntimeError(f"Unexpected error opening issue in {repo_owner}/{repo}: {e}")

# --------------------------------------------------------------------------- #
# 6. GET /repos/{owner}/{repo}/pulls                                          #
# --------------------------------------------------------------------------- #

def list_prs(
    repo: str,
    state: str = "open",
    owner: Optional[str] = None,
    base: Optional[str] = None,
) -> List[Any]:
    """Return pull requests matching *state* (and optional *base*)."""
    repo_owner = owner or DEFAULT_OWNER
    try:
        repository = _client().get_repo(f"{repo_owner}/{repo}")
        prs = repository.get_pulls(state=state, base=base)
        return list(prs)
    except GithubException as e:
        raise RuntimeError(
            f"GitHub API error listing PRs in {repo_owner}/{repo}: {e.data.get('message', str(e))}"
        )
    except Exception as e:
        raise RuntimeError(f"Unexpected error listing PRs in {repo_owner}/{repo}: {e}")

# --------------------------------------------------------------------------- #
# 7. POST /repos/{owner}/{repo}/pulls                                         #
# --------------------------------------------------------------------------- #

def create_pr(
    repo: str,
    title: str,
    head: str,
    base: str = "main",
    body: str = "",
    draft: bool = False,
    owner: Optional[str] = None,
) -> int:
    """Create a pull request and return its number."""
    repo_owner = owner or DEFAULT_OWNER
    try:
        repository = _client().get_repo(f"{repo_owner}/{repo}")
        pr = repository.create_pull(title=title, body=body, head=head, base=base, draft=draft)
        return pr.number
    except GithubException as e:
        raise RuntimeError(
            f"GitHub API error creating PR in {repo_owner}/{repo}: {e.data.get('message', str(e))}"
        )
    except Exception as e:
        raise RuntimeError(f"Unexpected error creating PR in {repo_owner}/{repo}: {e}")

# --------------------------------------------------------------------------- #
# 8. GET /repos/{owner}/{repo}/commits                                        #
# --------------------------------------------------------------------------- #

def list_commits(
    repo: str,
    branch: str = "main",
    owner: Optional[str] = None,
    per_page: int = DEFAULT_PER_PAGE,
) -> List[str]:
    """Return commit SHAs for *branch* (limited by *per_page*)."""
    repo_owner = owner or DEFAULT_OWNER
    try:
        repository = _client().get_repo(f"{repo_owner}/{repo}")
        commits = repository.get_commits(sha=branch)[:per_page]
        return [c.sha for c in commits]
    except GithubException as e:
        raise RuntimeError(
            f"GitHub API error listing commits in {repo_owner}/{repo}: {e.data.get('message', str(e))}"
        )
    except Exception as e:
        raise RuntimeError(f"Unexpected error listing commits in {repo_owner}/{repo}: {e}")

# --------------------------------------------------------------------------- #
# 9. GET /search/repositories                                                 #
# --------------------------------------------------------------------------- #

def search_repos(
    query: str,
    sort: str = "stars",
    order: str = "desc",
    per_page: int = 30,
) -> List[str]:
    """Search repositories and return their full names."""
    try:
        repos = _client().search_repositories(query=query, sort=sort, order=order)[:per_page]
        return [r.full_name for r in repos]
    except GithubException as e:
        raise RuntimeError(f"GitHub API error searching repositories: {e.data.get('message', str(e))}")
    except Exception as e:
        raise RuntimeError(f"Unexpected error searching repositories: {e}")

# --------------------------------------------------------------------------- #
# 10. GET /search/issues (issues + PRs)                                       #
# --------------------------------------------------------------------------- #

def search_issues(
    query: str,
    sort: str = "updated",
    order: str = "desc",
    per_page: int = 30,
) -> List[int]:
    """Search issues/PRs and return their numbers."""
    try:
        results = _client().search_issues(query=query, sort=sort, order=order)[:per_page]
        return [i.number for i in results]
    except GithubException as e:
        raise RuntimeError(f"GitHub API error searching issues: {e.data.get('message', str(e))}")
    except Exception as e:
        raise RuntimeError(f"Unexpected error searching issues: {e}")

# --------------------------------------------------------------------------- #
# 11. POST /user/repos                                                        #
# --------------------------------------------------------------------------- #

def create_repo(
    name: str,
    description: str = "",
    private: bool = False,
    has_issues: bool = True,
    has_wiki: bool = True,
    auto_init: bool = False,
) -> str:
    """Create a new repository and return its full name."""
    try:
        user = _client().get_user()
        repo = user.create_repo(
            name=name,
            description=description,
            private=private,
            has_issues=has_issues,
            has_wiki=has_wiki,
            auto_init=auto_init,
        )
        return repo.full_name
    except GithubException as e:
        raise RuntimeError(
            f"GitHub API error creating repository {name}: {e.data.get('message', str(e))}"
        )
    except Exception as e:
        raise RuntimeError(f"Unexpected error creating repository {name}: {e}")

def list_repo_files(repo: str, branch: str = "main", owner: Optional[str] = None) -> List[str]:
    """
    List all file paths in a given GitHub repository.

    Args:
        repo (str): Repository name (e.g., 'Hello-World')
        branch (str): Branch name (default: 'main')
        owner (str, optional): Owner of the repository. Defaults to DEFAULT_OWNER.
    
    Returns:
        List[str]: List of file paths
    """
    try:
        repo_owner = owner or DEFAULT_OWNER
        repository = _client().get_repo(f"{repo_owner}/{repo}") 
        git_tree = repository.get_git_tree(sha=branch, recursive=True).tree
        return [item.path for item in git_tree if item.type == "blob"]
    except GithubException as e:
        raise RuntimeError(f"GitHub API error listing files in {repo_owner}/{repo}: {e.data.get('message', str(e))}")
        return []
    except Exception as e:
        raise RuntimeError(f"Unexpected error listing files in {repo_owner}/{repo}: {e}")
        return []

def merge_branches(
    repo: str,
    head: str,
    base: str = "main",
    commit_message: Optional[str] = None,
    owner: Optional[str] = None,
    merge_method: str = "merge"
) -> Dict[str, Any]:
    """
    Merge a branch into another branch in a GitHub repository.
    
    Args:
        repo (str): Repository name (e.g. "my-repo").
        head (str): The name of the branch where changes are implemented (source branch).
        base (str): The name of the branch you want the changes pulled into (target branch). Defaults to "main".
        commit_message (Optional[str]): Commit message for the merge commit. If None, GitHub uses a default message.
        owner (Optional[str]): Repository owner or organization. Defaults to DEFAULT_OWNER.
        merge_method (str): The merge method to use: 'merge', 'squash', or 'rebase'. Defaults to 'merge'.
    
    Returns:
        Dict[str, Any]: Information about the merge including SHA of the resulting commit.
    """
    repo_owner = owner or DEFAULT_OWNER
    
    try:
        # Get the repository object
        repository = _client().get_repo(f"{repo_owner}/{repo}")
        
        # Perform the merge
        result = repository.merge(
            base=base,
            head=head,
            commit_message=commit_message or f"Merge branch '{head}' into {base}",
            merge_method=merge_method
        )
        
        # Return useful information about the merge
        return {
            "success": True,
            "sha": result.sha,
            "message": f"Successfully merged '{head}' into '{base}' using '{merge_method}' method",
            "commit_url": result.html_url
        }
        
    except GithubException as e:
        # Handle specific error cases
        if e.status == 409:  # Conflict
            raise RuntimeError(f"Merge conflict between '{head}' and '{base}'. Please resolve conflicts manually.")
        elif e.status == 404:  # Not Found
            raise RuntimeError(f"Branch '{head}' or '{base}' not found in repository {repo_owner}/{repo}.")
        else:
            message = e.data.get("message") if hasattr(e, "data") else str(e)
            raise RuntimeError(f"GitHub API error merging branches: {message}")
    except Exception as e:
        raise RuntimeError(f"Unexpected error merging branches: {e}")

def check_merge_status(
    repo: str,
    head: str,
    base: str = "main",
    owner: Optional[str] = None
) -> Dict[str, Union[bool, str]]:
    """
    Check if a branch can be merged into another branch without conflicts.
    
    Args:
        repo (str): Repository name (e.g. "my-repo").
        head (str): The name of the branch where changes are implemented (source branch).
        base (str): The name of the branch you want the changes pulled into (target branch). Defaults to "main".
        owner (Optional[str]): Repository owner or organization. Defaults to DEFAULT_OWNER.
    
    Returns:
        Dict[str, Union[bool, str]]: A dictionary containing:
            - mergeable (bool): Whether the branches can be merged automatically
            - message (str): Description of the merge status
    """
    repo_owner = owner or DEFAULT_OWNER
    
    try:
        # Get the repository object
        repository = _client().get_repo(f"{repo_owner}/{repo}")
        
        # Get the comparison between branches
        comparison = repository.compare(base, head)
        
        # Check for existing PRs between these branches
        existing_prs = repository.get_pulls(state='open', head=head, base=base)
        pr_count = existing_prs.totalCount
        
        # Get information about both branches to ensure they exist
        try:
            repository.get_branch(head)
            repository.get_branch(base)
        except GithubException:
            return {
                "mergeable": False,
                "message": f"One or both branches ('{head}' or '{base}') don't exist."
            }
        
        # Determine if they can be merged automatically
        if pr_count > 0:
            # If there's an open PR, we can check its mergeable status
            pr = existing_prs[0]
            mergeable = pr.mergeable
            
            if mergeable is None:
                # GitHub is still calculating mergeable status
                return {
                    "mergeable": None,
                    "message": "GitHub is still calculating merge status. Please try again in a few moments."
                }
            elif mergeable:
                return {
                    "mergeable": True,
                    "message": f"Branches '{head}' and '{base}' can be merged automatically. "
                               f"There are {comparison.ahead_by} commits ahead and {comparison.behind_by} commits behind."
                }
            else:
                return {
                    "mergeable": False,
                    "message": f"Branches '{head}' and '{base}' have conflicts that must be resolved manually."
                }
        else:
            # If no PR exists, make a best guess based on the comparison
            if comparison.ahead_by == 0:
                return {
                    "mergeable": True,
                    "message": f"Branch '{head}' has no new commits compared to '{base}'. Nothing to merge."
                }
            
            # We can't be 100% sure without a PR, so return a more cautious message
            return {
                "mergeable": None, 
                "message": f"Branch '{head}' is {comparison.ahead_by} commits ahead and "
                           f"{comparison.behind_by} commits behind '{base}'. "
                           f"Create a pull request to check for conflicts."
            }
            
    except GithubException as e:
        message = e.data.get("message") if hasattr(e, "data") else str(e)
        raise RuntimeError(f"GitHub API error checking merge status: {message}")
    except Exception as e:
        raise RuntimeError(f"Unexpected error checking merge status: {e}")

def create_branch(
    repo: str,
    branch_name: str,
    source_branch: str = "main",
    owner: Optional[str] = None
) -> dict:
    """
    Create a new branch in a GitHub repository based on an existing branch.
    
    Args:
        repo (str): Repository name (e.g. "my-repo").
        branch_name (str): Name of the new branch to create.
        source_branch (str): Name of the branch to base the new branch on. Defaults to "main".
        owner (Optional[str]): Repository owner or organization. Defaults to DEFAULT_OWNER.
    
    Returns:
        dict: Information about the created branch including ref and SHA.
    """
    repo_owner = owner or DEFAULT_OWNER
    
    try:
        # Get the repository object
        repository = _client().get_repo(f"{repo_owner}/{repo}")
        
        # Get the source branch to find its commit SHA
        try:
            source = repository.get_branch(source_branch)
        except GithubException as e:
            if e.status == 404:
                raise RuntimeError(f"Source branch '{source_branch}' not found in repository {repo_owner}/{repo}.")
            raise
            
        # Source branch SHA (commit to base the new branch on)
        sha = source.commit.sha
        
        # Check if branch already exists
        try:
            repository.get_branch(branch_name)
            # If we get here, branch exists
            return {
                "success": False,
                "message": f"Branch '{branch_name}' already exists in repository {repo_owner}/{repo}."
            }
        except GithubException as e:
            # 404 means branch doesn't exist, which is what we want
            if e.status != 404:
                raise
        
        # Create reference for the new branch
        # GitHub branches are refs in the form "refs/heads/branch-name"
        ref = f"refs/heads/{branch_name}"
        new_ref = repository.create_git_ref(ref=ref, sha=sha)
        
        return {
            "success": True,
            "ref": new_ref.ref,
            "sha": new_ref.object.sha,
            "message": f"Branch '{branch_name}' created successfully from '{source_branch}'.",
            "url": f"https://github.com/{repo_owner}/{repo}/tree/{branch_name}"
        }
        
    except GithubException as e:
        message = e.data.get("message") if hasattr(e, "data") else str(e)
        raise RuntimeError(f"GitHub API error creating branch: {message}")
    except Exception as e:
        raise RuntimeError(f"Unexpected error creating branch: {e}")

# --------------------------------------------------------------------------- #
# End of github_tools.py                                                      #
# --------------------------------------------------------------------------- #