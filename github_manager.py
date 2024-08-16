import os
import requests
from typing import List, Dict, Any


class GitHubManager:
    def __init__(self):
        self.base_url = "https://api.github.com"
        self.token = os.getenv("GITHUB_TOKEN")
        self.headers = {
            "Authorization": f"Bearer {self.token}",
            "Accept": "application/vnd.github.v3+json",
        }

    def list_repositories(self) -> List[Dict[str, Any]]:
        """List repositories for the authenticated user."""
        response = requests.get(f"{self.base_url}/user/repos", headers=self.headers)
        response.raise_for_status()
        return response.json()

    def get_repository_content(
        self, owner: str, repo: str, path: str
    ) -> Dict[str, Any]:
        """Get repository content."""
        response = requests.get(
            f"{self.base_url}/repos/{owner}/{repo}/contents/{path}",
            headers=self.headers,
        )
        response.raise_for_status()
        return response.json()

    def create_comment(
        self, owner: str, repo: str, issue_number: int, body: str
    ) -> Dict[str, Any]:
        """Create a comment on a pull request."""
        data = {"body": body}
        response = requests.post(
            f"{self.base_url}/repos/{owner}/{repo}/issues/{issue_number}/comments",
            headers=self.headers,
            json=data,
        )
        response.raise_for_status()
        return response.json()

    def create_pull_request(
        self, owner: str, repo: str, title: str, head: str, base: str, body: str
    ) -> Dict[str, Any]:
        """Create a pull request."""
        data = {"title": title, "head": head, "base": base, "body": body}
        response = requests.post(
            f"{self.base_url}/repos/{owner}/{repo}/pulls",
            headers=self.headers,
            json=data,
        )
        response.raise_for_status()
        return response.json()

    def merge_pull_request(
        self, owner: str, repo: str, pull_number: int
    ) -> Dict[str, Any]:
        """Merge a pull request."""
        response = requests.put(
            f"{self.base_url}/repos/{owner}/{repo}/pulls/{pull_number}/merge",
            headers=self.headers,
        )
        response.raise_for_status()
        return response.json()


github_manager = GitHubManager()
