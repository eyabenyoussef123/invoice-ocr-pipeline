from pathlib import Path
import json
import sys

# Add root to path for module execution
root_path = Path(__file__).parent.parent.parent
if str(root_path) not in sys.path:
    sys.path.insert(0, str(root_path))

from src.ocr.compare_ocr import run_ocr_on_image, find_total_in_text


CONF_MARGIN = 0.02  # 2% threshold

def score_result(avg_conf, total, num_lines):
    score = avg_conf
    if total is not None:
        score += 0.05
    score += min(num_lines / 100, 0.05)
    return score


def select_best(image_original, image_processed):

    ext_o, conf_o, text_o = run_ocr_on_image(image_original)
    ext_p, conf_p, text_p = run_ocr_on_image(image_processed)

    total_o = find_total_in_text(text_o)
    total_p = find_total_in_text(text_p)

    score_o = score_result(conf_o, total_o, len(ext_o))
    score_p = score_result(conf_p, total_p, len(ext_p))

    if score_p > score_o + CONF_MARGIN:
        return {
            "chosen": "processed",
            "avg_conf": conf_p,
            "total": total_p,
            "lines": ext_p
        }
    else:
        return {
            "chosen": "original",
            "avg_conf": conf_o,
            "total": total_o,
            "lines": ext_o
        }


if __name__ == "__main__":

    img_original = Path("data/raw/test2.png")
    img_processed = Path("data/processed/test2_processed.png")

    result = select_best(img_original, img_processed)

    print("Chosen image:", result["chosen"])
    print("Confidence:", result["avg_conf"])
    print("Total:", result["total"])

    # Save result to final_decision.json
    out_path = Path("data/final/final_decision2.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    print(f"✅ Saved decision → {out_path}")
    with open("final_invoice.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
