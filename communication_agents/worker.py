# worker.py
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from github import Github
import os

# Load your GitHub Personal Access Token from the environment
GITHUB_TOKEN = os.getenv("GITHUB_PERSONAL_ACCESS_TOKEN")
if not GITHUB_TOKEN:
    raise RuntimeError("Please set the GITHUB_PERSONAL_ACCESS_TOKEN environment variable")

gh = Github(GITHUB_TOKEN)
app = FastAPI()


class FileOut(BaseModel):
    content: str


@app.get("/file", response_model=FileOut)
def get_file_content(
    owner: str = Query(..., description="GitHub repo owner"),
    repo: str = Query(..., description="GitHub repo name"),
    path: str = Query(..., description="Path to file in the repo"),
    branch: str = Query("main", description="Branch or commit SHA"),
):
    """
    Fetch the raw content of a single file from GitHub.
    """
    try:
        repository = gh.get_repo(f"{owner}/{repo}")
        blob = repository.get_contents(path, ref=branch)
        content = blob.decoded_content.decode("utf-8", errors="ignore")
        return FileOut(content=content)
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Could not fetch file: {e}")
