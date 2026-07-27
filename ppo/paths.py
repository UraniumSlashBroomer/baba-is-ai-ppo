import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BABA_PATH = ROOT / "baba-is-ai"

if str(BABA_PATH) not in sys.path:
    sys.path.insert(0, str(BABA_PATH))
