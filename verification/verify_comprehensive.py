import subprocess
import sys
import time
from datetime import datetime

# Define the list of verification steps
# Each step has a name and a command to execute
STEPS = [
    {
        "name": "Unit Tests: Token Limit",
        "command": [sys.executable, "-m", "pytest", "verification/comprehensive_tests/test_token_limit.py", "-v"],
        "description": "Verifies token limit enforcement logic in retrieval."
    },
    {
        "name": "Unit Tests: Multi-Image Support",
        "command": [sys.executable, "-m", "pytest", "verification/comprehensive_tests/test_multi_image.py", "-v"],
        "description": "Verifies handling of multiple images in retrieval."
    },
    {
        "name": "Component: Chunking & Embedding",
        "command": [sys.executable, "verification/comprehensive_tests/verify_chunking_embedding.py"],
        "description": "Verifies SemanticChunker and EmbeddingGenerator (Dense/Sparse) outputs."
    },
    {
        "name": "Component: Path Sanitization & Chunking",
        "command": [sys.executable, "verification/comprehensive_tests/verify_step3.py"],
        "description": "Verifies path sanitization and basic chunking functionality."
    },
    {
        "name": "Integration: Retrieval (Local DB)",
        "command": [sys.executable, "verification/comprehensive_tests/verify_step5.py"],
        "description": "Verifies FlexibleRetriever against the local 'test_repo' Qdrant DB."
    },
    {
        "name": "Integration: RAG Pipeline (Code Gen)",
        "command": [sys.executable, "verification/comprehensive_tests/verify_step6.py"],
        "description": "Verifies RAGPipeline's code generation mode with mocked components."
    },
    {
        "name": "Debug: BM25 Initialization",
        "command": [sys.executable, "verification/comprehensive_tests/debug_bm25.py"],
        "description": "Verifies BM25 tokenizer and initialization."
    }
]

def log(section, message):
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] [{section}] {message}")

def run_step(step):
    name = step["name"]
    command = step["command"]
    description = step["description"]
    
    print("\n" + "="*80)
    log("START", f"Running: {name}")
    log("INFO", description)
    print("-" * 80)
    
    start_time = time.time()
    try:
        # Run command and stream output
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        
        output_lines = []
        for line in process.stdout:
            print(f"  | {line}", end="")
            output_lines.append(line)
            
        process.wait()
        duration = time.time() - start_time
        
        if process.returncode == 0:
            log("PASS", f"Finished in {duration:.2f}s")
            return True, output_lines
        else:
            log("FAIL", f"Failed with exit code {process.returncode} in {duration:.2f}s")
            return False, output_lines
            
    except Exception as e:
        duration = time.time() - start_time
        log("ERROR", f"Exception occurred: {e}")
        return False, [str(e)]

def main():
    print("="*80)
    print("COMPREHENSIVE SYSTEM VERIFICATION SUITE")
    print("="*80)
    print(f"Python Executable: {sys.executable}")
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    results = []
    all_passed = True
    
    for step in STEPS:
        success, _ = run_step(step)
        results.append({"name": step["name"], "success": success})
        if not success:
            all_passed = False
            # Optional: Break on failure? No, let's run all to see full state.
    
    print("\n" + "="*80)
    print("VERIFICATION SUMMARY")
    print("="*80)
    
    for res in results:
        status = "✅ PASS" if res["success"] else "❌ FAIL"
        print(f"{status} - {res['name']}")
        
    print("-" * 80)
    if all_passed:
        print("🎉 ALL TESTS PASSED. The system is healthy.")
        sys.exit(0)
    else:
        print("⚠️ SOME TESTS FAILED. Please check the logs above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
