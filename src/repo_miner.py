#!/usr/bin/env python3
"""
repo_miner.py

A command-line tool to:
  1) Fetch and normalize commit data from GitHub

Sub-commands:
  - fetch-commits
"""

import os
import argparse
import pandas as pd
from github import Github
import datetime

def fetch_commits(repo_name: str, max_commits: int = None) -> pd.DataFrame:
    """
    Fetch up to `max_commits` from the specified GitHub repository.
    Returns a DataFrame with columns: sha, author, email, date, message.
    """
    # 1) Read GitHub token from environment
    token = os.getenv('GITHUB_TOKEN')

    # 2) Initialize GitHub client and get the repo
    g = Github(token)
    repo = g.get_repo(repo_name)

    # 3) Fetch commit objects (paginated by PyGitHub)
    commits = repo.get_commits()

    # 4) Normalize each commit into a record dict
    record = []
    for i,commit in enumerate(commits):
      if max_commits is not None and i >= max_commits: break
      commit_record = {
        'sha': commit.sha,
        'author': commit.commit.author.name,
        'email': commit.commit.author.email,
        'date': commit.commit.author.date.isoformat(),
        'message': commit.commit.message
      }
      record.append(commit_record)

    # 5) Build DataFrame from records
    df = pd.DataFrame(record)
    
    return df


def fetch_issues(repo_name: str, state: str = "all", max_issues: int = None) -> pd.DataFrame:
    """
    Fetch up to `max_issues` from the specified GitHub repository (issues only).
    Returns a DataFrame with columns: id, number, title, user, state, created_at, closed_at, comments.
    """
    # 1) Read GitHub token
    token = os.getenv('GITHUB_TOKEN')

    # 2) Initialize client and get the repo
    g = Github(token)
    repo = g.get_repo(repo_name)

    # 3) Fetch issues, filtered by state ('all', 'open', 'closed')
    issues = repo.get_issues(state=state)

    # 4) Normalize each issue (skip PRs)
    records = []
    for idx, issue in enumerate(issues):
        if max_issues and idx >= max_issues:
            break
        # Skip pull requests
        if issue.pull_request is not None:
            continue

        # Append records
        record = {}
        record['id'] = issue.id
        record['number'] = issue.number
        record['title'] = issue.title
        record['user'] = issue.user
        record['state'] = issue.state
        record['created_at'] = issue.created_at.isoformat()
        record['open_duration_days'] = \
          (issue.closed_at - issue.created_at).days if issue.state == 'closed' else None
        record['closed_at'] = \
          issue.closed_at.isoformat() if issue.state == 'closed' else None
        record['comments'] = issue.comments
        records.append(record)

    # 5) Build DataFrame
    df = pd.DataFrame(records)
    return df


def main():
    """
    Parse command-line arguments and dispatch to sub-commands.
    """
    parser = argparse.ArgumentParser(
        prog="repo_miner",
        description="Fetch GitHub commits/issues and summarize them"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Sub-command: fetch-commits
    c1 = subparsers.add_parser("fetch-commits", help="Fetch commits and save to CSV")
    c1.add_argument("--repo", required=True, help="Repository in owner/repo format")
    c1.add_argument("--max",  type=int, dest="max_commits",
                    help="Max number of commits to fetch")
    c1.add_argument("--out",  required=True, help="Path to output commits CSV")

    # Sub-command: fetch-issues
    c2 = subparsers.add_parser("fetch-issues", help="Fetch issues and save to CSV")
    c2.add_argument("--repo",  required=True, help="Repository in owner/repo format")
    c2.add_argument("--state", choices=["all","open","closed"], default="all",
                    help="Filter issues by state")
    c2.add_argument("--max",   type=int, dest="max_issues",
                    help="Max number of issues to fetch")
    c2.add_argument("--out",   required=True, help="Path to output issues CSV")

    args = parser.parse_args()

    # Dispatch based on selected command
    if args.command == "fetch-commits":
        df = fetch_commits(args.repo, args.max_commits)
        df.to_csv(args.out, index=False)
        print(f"Saved {len(df)} commits to {args.out}")

    elif args.command == "fetch-issues":
        df = fetch_issues(args.repo, args.state, args.max_issues)
        df.to_csv(args.out, index=False)
        print(f"Saved {len(df)} issues to {args.out}")


if __name__ == "__main__":
    main()