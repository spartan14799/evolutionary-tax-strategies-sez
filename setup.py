import sys
from pipreqs import pipreqs

def main():
    # Set the path to your repo (use "." for current directory)
    path = "."
    print(f"🔍 Scanning directory: {path}")

    try:
        # Equivalent to: python -m pipreqs . --force --encoding=utf-8
        sys.argv = ["pipreqs", path, "--force", "--encoding=utf-8"]
        pipreqs.main()
        print("✅ requirements.txt generated successfully!")
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
