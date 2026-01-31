"""Download refund policy PDF to data/policies/."""
import sys
from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
POLICIES_DIR = PROJECT_ROOT / "data" / "policies"
POLICIES_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_PATH = POLICIES_DIR / "refund_policy.pdf"
URL = "https://static.lightricks.com/legal/refund-policy.pdf"


def main() -> None:
    try:
        import urllib.request
        urllib.request.urlretrieve(URL, OUTPUT_PATH)
        print(f"Downloaded to {OUTPUT_PATH}")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
