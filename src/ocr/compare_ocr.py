from paddleocr import PaddleOCR
import re
from pathlib import Path
import json
import statistics

# =========================
# OCR INITIALIZATION
# =========================
# Stable PaddleOCR API (v2.7.x)
ocr = PaddleOCR(
    use_angle_cls=True,   # text rotation correction
    lang='fr',            # French
    use_gpu=False         # CPU only
)

# =========================
# REGEX & KEYWORDS
# =========================
AMOUNT_RE = re.compile(
    r'(\d{1,3}(?:[ \u00A0,]\d{3})*(?:[.,]\d{2}))\s*(€|EUR|DT|TND)?',
    re.IGNORECASE
)

TOTAL_KEYWORDS = [
    'total t.t.c',
    'total ttc',
    'total à payer',
    'total t.t.c.',
    'total'
]

# =========================
# OCR FUNCTION
# =========================
def run_ocr_on_image(image_path: Path):
    """
    Run OCR using PaddleOCR stable API.
    Returns:
    - extracted lines
    - average confidence
    - full text
    """

    result = ocr.ocr(str(image_path), cls=True)

    extracted = []
    texts = []
    confs = []

    # result structure:
    # [
    #   [
    #     [box, (text, confidence)],
    #     ...
    #   ]
    # ]
    for page in result:
        for box, (text, conf) in page:
            conf = float(conf)
            extracted.append({
                "text": text,
                "conf": conf,
                "box": box
            })
            texts.append(text)
            confs.append(conf)

    avg_conf = statistics.mean(confs) if confs else 0.0
    full_text = "\n".join(texts)

    return extracted, avg_conf, full_text


# =========================
# TOTAL DETECTION
# =========================
def find_total_in_text(text_block: str):
    """
    Heuristic to detect TOTAL amount:
    1) Search 'total' keywords bottom-up
    2) Fallback: largest detected amount
    """

    lines = text_block.splitlines()

    # 1️⃣ Keyword-based search (bottom of invoice)
    for line in reversed(lines):
        low = line.lower()
        if any(k in low for k in TOTAL_KEYWORDS):
            match = AMOUNT_RE.search(line)
            if match:
                return (
                    match.group(1)
                    .replace(" ", "")
                    .replace("\u00A0", "")
                    .replace(",", ".")
                )

    # 2️⃣ Fallback: largest amount
    matches = AMOUNT_RE.findall(text_block)
    if not matches:
        return None

    cleaned = [
        m[0]
        .replace(" ", "")
        .replace("\u00A0", "")
        .replace(",", ".")
        for m in matches
    ]

    try:
        nums = [float(x) for x in cleaned]
        return str(max(nums))
    except ValueError:
        return cleaned[-1]


# =========================
# MAIN
# =========================
if __name__ == "__main__":

    img_processed = Path("data/processed/test2_processed.png")
    img_original = Path("data/raw/test2.png")

    for img in [img_processed, img_original]:

        if not img.exists():
            print(f"❌ File not found: {img}")
            continue

        print(f"\n--- OCR on: {img.name} ---")

        extracted, avg_conf, fulltext = run_ocr_on_image(img)

        print(f"Average confidence: {avg_conf:.3f}")

        # Show first 10 lines
        for i, e in enumerate(extracted[:10], start=1):
            print(f"[{i}] {e['text']} (conf={e['conf']:.2f})")

        total = find_total_in_text(fulltext)
        print("Detected total (heuristic):", total)

        # Save JSON output
        out_json = img.with_suffix(".ocr.json")
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "avg_conf": avg_conf,
                    "lines": extracted,
                    "detected_total": total
                },
                f,
                ensure_ascii=False,
                indent=2
            )

        print("Saved OCR JSON →", out_json)
