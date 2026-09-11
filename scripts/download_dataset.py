"""Download the official UCI SMS Spam Collection into data/spam.csv."""
from pathlib import Path
from urllib.request import urlopen
import zipfile

URL = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
ZIP_PATH = DATA_DIR / "sms_spam_collection.zip"
RAW_PATH = DATA_DIR / "SMSSpamCollection"
CSV_PATH = DATA_DIR / "spam.csv"


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print("Downloading UCI SMS Spam Collection...")
    with urlopen(URL, timeout=30) as response:
        ZIP_PATH.write_bytes(response.read())

    with zipfile.ZipFile(ZIP_PATH) as archive:
        archive.extractall(DATA_DIR)

    lines = RAW_PATH.read_text(encoding="utf-8", errors="replace").splitlines()
    rows = ["label,text"]
    for line in lines:
        if "\t" not in line:
            continue
        label, text = line.split("\t", 1)
        escaped = text.replace('"', '""')
        rows.append(f'{label},"{escaped}"')
    CSV_PATH.write_text("\n".join(rows) + "\n", encoding="utf-8")
    ZIP_PATH.unlink(missing_ok=True)
    RAW_PATH.unlink(missing_ok=True)
    print(f"Saved {len(rows) - 1} messages to {CSV_PATH}")


if __name__ == "__main__":
    main()
