import os
import subprocess
import sys
from pathlib import Path

def clone_repo(repo_name: str, version: str, target_dir: str = "data/repos"):
    """
    Clone a repository and checkout a specific version.
    
    Args:
        repo_name: Name of the repository (e.g., 'markedjs/marked')
        version: Version tag or commit hash (e.g., 'v1.2.0')
        target_dir: Directory to clone into
    """
    # Create target directory
    target_path = Path(target_dir)
    target_path.mkdir(parents=True, exist_ok=True)
    
    # Sanitize repo name for folder
    safe_repo_name = repo_name.replace("/", "__")
    safe_version = version.replace(".", "_")
    folder_name = f"{safe_repo_name}__{safe_version}"
    repo_dir = target_path / folder_name
    
    print(f"Preparing to clone {repo_name} (version {version}) into {repo_dir}...")
    
    if repo_dir.exists():
        print(f"Directory {repo_dir} already exists.")
        # Check if it's a git repo
        if (repo_dir / ".git").exists():
            print("Git repository found. Fetching updates...")
            subprocess.run(["git", "fetch", "--all"], cwd=repo_dir, check=True)
        else:
            print("Directory exists but is not a git repo. Please remove it manually.")
            return str(repo_dir)
    else:
        # Clone
        repo_url = f"https://github.com/{repo_name}.git"
        print(f"Cloning {repo_url}...")
        subprocess.run(["git", "clone", repo_url, str(repo_dir)], check=True)
        
    # Checkout version
    # Checkout version
    print(f"Checking out version {version}...")
    
    # 1. Try exact match
    try:
        subprocess.run(["git", "checkout", version], cwd=repo_dir, check=True, stderr=subprocess.DEVNULL)
        print(f"Successfully checked out {version}")
        return str(repo_dir)
    except subprocess.CalledProcessError:
        pass

    # 2. Try 'v' prefix
    try:
        subprocess.run(["git", "checkout", f"v{version}"], cwd=repo_dir, check=True, stderr=subprocess.DEVNULL)
        print(f"Successfully checked out v{version}")
        return str(repo_dir)
    except subprocess.CalledProcessError:
        pass

    # 3. Try fuzzy match on tags (e.g. 1.2 -> v1.2.0)
    print(f"Direct checkout failed. Searching tags for {version}...")
    try:
        result = subprocess.run(["git", "tag"], cwd=repo_dir, capture_output=True, text=True, check=True)
        tags = result.stdout.splitlines()
        
        # Look for tags that start with v{version}. or {version}.
        # We prefer v{version}.0 or {version}.0
        candidates = []
        for t in tags:
            if t == f"v{version}.0":
                candidates.insert(0, t) # Prioritize .0
            elif t == f"{version}.0":
                candidates.insert(0, t)
            elif t.startswith(f"v{version}.") or t.startswith(f"{version}."):
                candidates.append(t)
        
        if candidates:
            best_match = candidates[0]
            print(f"Found candidate tag: {best_match}")
            subprocess.run(["git", "checkout", best_match], cwd=repo_dir, check=True)
            return str(repo_dir)
            
    except Exception as e:
        print(f"Tag search failed: {e}")

    print(f"Checkout failed for {version}. Please check if the tag/commit exists.")
    raise Exception(f"Version {version} not found")

    print(f"Successfully prepared {repo_name} at {version}")
    return str(repo_dir)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("repo", help="Repository name (e.g. markedjs/marked)")
    parser.add_argument("version", help="Version tag or commit")
    parser.add_argument("--dir", default="data/repos", help="Target directory")
    
    args = parser.parse_args()
    
    clone_repo(args.repo, args.version, args.dir)
