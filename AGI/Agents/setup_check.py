"""Pre-flight verification script."""

import importlib
import os
import sys
import glob


def check_python():
    print(f"Python: {sys.version.split()[0]} (need 3.10+)")
    assert sys.version_info >= (3, 10), "Python 3.10+ required"
    print("  OK")


def check_dependencies():
    for pkg in ["llama_cpp"]:
        try:
            importlib.import_module(pkg)
            print(f"  {pkg}: installed")
        except ImportError:
            print(f"  {pkg}: MISSING — run: pip install llama-cpp-python")
            return False
    return True


def check_models():
    models_dir = "models"
    if not os.path.isdir(models_dir):
        print(f"  models/: directory missing — create it and download a .gguf file")
        return False
    gguf_files = glob.glob(os.path.join(models_dir, "*.gguf"))
    if not gguf_files:
        print(f"  models/: no .gguf files found")
        return False
    print(f"  models/: {len(gguf_files)} .gguf file(s) found")
    for f in gguf_files:
        size_mb = os.path.getsize(f) / (1024 * 1024)
        print(f"    {os.path.basename(f)} ({size_mb:.0f} MB)")
    return True


def check_structure():
    for name in ["agents.py", "lessons", "diagrams"]:
        if os.path.exists(name):
            print(f"  {name}: found")
        else:
            print(f"  {name}: MISSING")


def main():
    print("=== Setup Check ===\n")
    print("[1/4] Python version:"),
    check_python()
    print()
    print("[2/4] Dependencies:")
    deps_ok = check_dependencies()
    print()
    print("[3/4] Models:")
    models_ok = check_models()
    print()
    print("[4/4] Project structure:")
    check_structure()
    print()
    if deps_ok and models_ok:
        print("All checks passed — ready to go!")
    else:
        print("Some checks failed — see above.")


if __name__ == "__main__":
    main()
