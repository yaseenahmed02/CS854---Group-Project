
import json
import re
import glob
import os

CONSOLIDATED_DIR = "results/consolidated"

def clean_patch(text):
    if not isinstance(text, str):
        return ""
        
    # 1. Extract from Markdown code blocks if present
    # Look for ```diff ... ``` or ``` ... ```
    # We prioritize blocks that look like patches
    code_block_pattern = re.compile(r"```(?:diff|patch)?\s*(.*?)```", re.DOTALL)
    matches = code_block_pattern.findall(text)
    
    candidate = text
    if matches:
        # Heuristic: pick the block that contains "diff --git" or "--- a/"
        # If multiple, pick the longest one containing those
        valid_matches = [m for m in matches if "diff --git" in m or ("--- a/" in m and "+++ b/" in m)]
        if valid_matches:
            candidate = max(valid_matches, key=len)
        else:
            # If no obvious patch block, maybe it's just the longest block?
            # Or maybe the text wasn't in a block?
            # Let's fallback to original text if no block looks like a patch, 
            # UNLESS the original text is HUGE and the block is reasonable.
            # actually, if there are blocks but none look like patches, it's risky.
            # But usually the patch IS in a block.
            if len(matches) == 1:
                candidate = matches[0]
            else:
                 candidate = max(matches, key=len)

    # 2. Strip pre-patch text
    # Find start of patch
    # Patterns: "diff --git" or "--- a/"
    # We want the earliest occurrence
    
    patterns = ["diff --git", "--- a/"]
    start_indices = [candidate.find(p) for p in patterns]
    valid_indices = [i for i in start_indices if i != -1]
    
    if valid_indices:
        start_index = min(valid_indices)
        candidate = candidate[start_index:]
    
    # 3. Strip post-patch text?
    # This is harder. Usually looking for the last line that looks like code or diff syntax.
    # A simple valid patch ends with a hunk.
    # Just stripping trailing whitespace is usually enough, `patch` ignores trailing garbage often.
    # But user asked to remove it.
    # We can try to keep lines that look like patch lines:
    # starting with ' ', '+', '-', '@', 'diff', 'index', '---', '+++', 'old', 'new', 'binary'
    # But timestamps or file modes might be relevant.
    # Let's assume trimming whitespace is step 1.
    
    candidate = candidate.strip()
    
    return candidate

def main():
    files = glob.glob(os.path.join(CONSOLIDATED_DIR, "*_predictions.json"))
    print(f"Cleaning patches in {len(files)} files...")
    
    for fpath in files:
        print(f"Processing {os.path.basename(fpath)}...")
        try:
            with open(fpath, 'r') as f:
                data = json.load(f)
            
            modified = False
            for entry in data:
                original = entry.get("model_patch", "")
                cleaned = clean_patch(original)
                
                if cleaned != original:
                    entry["model_patch"] = cleaned
                    modified = True
            
            if modified:
                with open(fpath, 'w') as f:
                    json.dump(data, f, indent=2)
                print(f"  Saved updates to {os.path.basename(fpath)}")
            else:
                print(f"  No changes needed.")
                
        except Exception as e:
            print(f"Error processing {fpath}: {e}")

if __name__ == "__main__":
    main()
