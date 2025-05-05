from typing import List, Optional, Any
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
DEFAULT_OWNER: str = os.getenv("GH_DEFAULT_OWNER", "Shreyas-Yadav")
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

def list_my_repos() -> List[str]:
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
    state: str = "open",
    owner: Optional[str] = None,
    labels: Optional[List[str]] = None,
) -> List[Any]:
    """Return issues matching *state* and optional *labels*."""
    repo_owner = owner or DEFAULT_OWNER
    try:
        repository = _client().get_repo(f"{repo_owner}/{repo}")
        issues = repository.get_issues(state=state, labels=",".join(labels) if labels else None)
        return list(issues)
    except GithubException as e:
        raise RuntimeError(
            f"GitHub API error listing issues in {repo_owner}/{repo}: {e.data.get('message', str(e))}"
        )
    except Exception as e:
        raise RuntimeError(f"Unexpected error listing issues in {repo_owner}/{repo}: {e}")

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
# End of github_tools.py                                                      #
# --------------------------------------------------------------------------- #







tools = [list_my_repos, read_file, commit_file, list_issues, open_issue, list_prs, create_pr, list_commits, search_repos, search_issues]


