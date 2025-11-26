import sys
import os
import subprocess
import time

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def run_script(script_path):
    print(f"\n>>> Running {script_path}...")
    start_time = time.time()
    try:
        result = subprocess.run(
            [sys.executable, script_path],
            capture_output=True,
            text=True,
            check=True
        )
        duration = time.time() - start_time
        print(f"✅ PASS ({duration:.2f}s)")
        return True, result.stdout
    except subprocess.CalledProcessError as e:
        duration = time.time() - start_time
        print(f"❌ FAIL ({duration:.2f}s)")
        print("Output:")
        print(e.stdout)
        print("Error:")
        print(e.stderr)
        return False, e.stderr

def verify_comprehensive():
    print("\n=== Comprehensive Regression Suite ===\n")
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    tests_dir = os.path.join(base_dir, "tests")
    
    # Define test order (Unit -> Integration -> System)
    test_order = [
        # Unit/Component
        "verify_chunking_embedding.py",
        "test_token_limit.py",
        
        # Integration
        "verify_step3.py",
        "verify_step5.py",
        "verify_step6.py",
        "test_multi_image.py",
        
        # System State
        "verify_image_count.py",
        "verify_ingestion_count.py"
    ]
    
    results = {}
    
    for script_name in test_order:
        script_path = os.path.join(tests_dir, script_name)
        if os.path.exists(script_path):
            success, output = run_script(script_path)
            results[script_name] = "PASS" if success else "FAIL"
        else:
            print(f"⚠️  Skipping {script_name} (Not found)")
            results[script_name] = "MISSING"

    print("\n" + "="*50)
    print("TEST SUMMARY")
    print("="*50)
    all_passed = True
    for name, status in results.items():
        color = "\033[92m" if status == "PASS" else "\033[91m"
        reset = "\033[0m"
        print(f"{name:.<40} [{color}{status}{reset}]")
        if status != "PASS":
            all_passed = False
            
    print("="*50)
    if all_passed:
        print("\033[92mALL TESTS PASSED\033[0m")
    else:
        print("\033[91mSOME TESTS FAILED\033[0m")
        sys.exit(1)

if __name__ == "__main__":
    verify_comprehensive()
