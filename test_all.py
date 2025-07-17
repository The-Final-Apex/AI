import subprocess
import sys

def run_python_tests():
    print("🧪 Running Python Tests...")
    result = subprocess.run(["python", "-m", "unittest", "discover", "-s", "tests", "-p", "test_python.py"], capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print("❌ Python tests failed.")
        print(result.stderr)
        sys.exit(1)

def run_cpp_tests():
    print("🧪 Compiling and Running C++ Tests...")
    cpp_file = "tests/test_cpp.cpp"
    exe_file = "tests/test_cpp_exec.exe"
    compile = subprocess.run(["g++", "-std=c++17", cpp_file, "-o", exe_file], capture_output=True, text=True)
    if compile.returncode != 0:
        print("❌ C++ compile error:\n", compile.stderr)
        sys.exit(1)
    run = subprocess.run([exe_file], capture_output=True, text=True)
    print(run.stdout)
    if run.returncode != 0:
        print("❌ C++ tests failed.")
        sys.exit(1)

if __name__ == "__main__":
    run_python_tests()
    run_cpp_tests()
    print("✅ All tests passed.")
