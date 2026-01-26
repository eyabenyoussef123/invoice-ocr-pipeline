from pathlib import Path
import json
import sys

# Add root to path
root_path = Path(__file__).parent.parent
if str(root_path) not in sys.path:
    sys.path.insert(0, str(root_path))

from src.ocr.compare_ocr import run_ocr_on_image, find_total_in_text
from src.preprocess.image_preprocessor import (
    load_image,
    to_grayscale,
    enhance_contrast,
    otsu_binarization,
    deskew_image,
    denoise_image,
    save_processed_image
)

CONF_MARGIN = 0.02  # 2% threshold

def score_result(avg_conf, total, num_lines):
    """Calculate OCR result score"""
    score = avg_conf
    if total is not None:
        score += 0.05
    score += min(num_lines / 100, 0.05)
    return score

def preprocess_image(image_path):
    """Preprocess an image and return the path to the preprocessed image"""
    temp_dir = Path("data/orange_factures/.temp")
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    image = load_image(image_path)
    gray = to_grayscale(image)
    enhanced = enhance_contrast(gray)
    binary = otsu_binarization(enhanced)
    deskewed, angle = deskew_image(binary)
    denoised = denoise_image(deskewed)
    
    out_path = temp_dir / f"{image_path.stem}_processed.jpg"
    save_processed_image(denoised, out_path)
    
    return out_path

def select_best_orange(image_original):
    """Process an Orange facture image and select best OCR result"""
    
    # Preprocess
    image_processed = preprocess_image(image_original)
    
    # Run OCR on both versions
    try:
        ext_o, conf_o, text_o = run_ocr_on_image(image_original)
    except Exception as e:
        print(f"      ⚠️  OCR failed on original: {e}")
        ext_o, conf_o, text_o = [], 0.0, ""
    
    try:
        ext_p, conf_p, text_p = run_ocr_on_image(image_processed)
    except Exception as e:
        print(f"      ⚠️  OCR failed on preprocessed: {e}")
        ext_p, conf_p, text_p = [], 0.0, ""
    
    # Find totals (handle None returns)
    total_o = find_total_in_text(text_o) if text_o else None
    total_p = find_total_in_text(text_p) if text_p else None
    
    # Calculate scores
    score_o = score_result(conf_o, total_o, len(ext_o)) if ext_o else 0.0
    score_p = score_result(conf_p, total_p, len(ext_p)) if ext_p else 0.0
    
    # Choose best version (prefer original if scores are too close)
    if score_p > score_o + CONF_MARGIN and len(ext_p) > 0:
        chosen = "preprocessed"
        result = {
            "chosen": chosen,
            "image": image_original.name,
            "avg_conf": conf_p,
            "total": total_p,
            "lines": ext_p,
            "score": score_p,
            "scores": {
                "original": score_o,
                "preprocessed": score_p
            }
        }
    else:
        chosen = "original"
        result = {
            "chosen": chosen,
            "image": image_original.name,
            "avg_conf": conf_o,
            "total": total_o,
            "lines": ext_o,
            "score": score_o,
            "scores": {
                "original": score_o,
                "preprocessed": score_p
            }
        }
    
    # Clean up temp file
    try:
        image_processed.unlink()
    except:
        pass
    
    return result

if __name__ == "__main__":
    
    IMG_DIR = Path("data/orange_factures/images")
    OUT_DIR = Path("data/orange_factures/results")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    
    all_decisions = []
    stats = {"total": 0, "ok": 0, "error": 0}
    
    print("🧾 Processing Orange Factures\n")
    print("=" * 60)
    
    for img_path in sorted(IMG_DIR.glob("*.jpg")):
        stats["total"] += 1
        print(f"\n📄 {img_path.name}")
        
        try:
            result = select_best_orange(img_path)
            
            # Display results
            chosen = result["chosen"].upper()
            conf = result["avg_conf"]
            lines_count = len(result["lines"])
            total = result["total"]
            score = result["score"]
            
            print(f"   Chosen: {chosen}")
            print(f"   Confidence: {conf:.2%}")
            print(f"   Lines: {lines_count}")
            print(f"   Total detected: {total}")
            print(f"   Score: {score:.3f}")
            print(f"   Scores comparison: Original={result['scores']['original']:.3f}, Preprocessed={result['scores']['preprocessed']:.3f}")
            
            # Save individual decision
            out_file = OUT_DIR / f"{img_path.stem}_decision.json"
            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            
            all_decisions.append(result)
            stats["ok"] += 1
            
        except Exception as e:
            print(f"    Error: {e}")
            stats["error"] += 1
    
    # Save all decisions
    all_decisions_file = OUT_DIR / "all_decisions.json"
    with open(all_decisions_file, "w", encoding="utf-8") as f:
        json.dump(all_decisions, f, ensure_ascii=False, indent=2)
    
    print("\n" + "=" * 60)
    print(" PROCESSING COMPLETE")
    print("=" * 60)
    print(f"Total: {stats['total']} | Success: {stats['ok']} | Errors: {stats['error']}")
    print(f" Results saved in: {OUT_DIR}")
