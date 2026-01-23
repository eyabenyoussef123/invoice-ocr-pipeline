import json
from pathlib import Path
from statistics import mean

# =========================
# Geometry helpers
# =========================
def bbox_left(b): return min(p[0] for p in b)
def bbox_right(b): return max(p[0] for p in b)
def bbox_top(b): return min(p[1] for p in b)
def bbox_bottom(b): return max(p[1] for p in b)

def bbox_union(boxes):
    return [
        min(bbox_left(b) for b in boxes),
        min(bbox_top(b) for b in boxes),
        max(bbox_right(b) for b in boxes),
        max(bbox_bottom(b) for b in boxes)
    ]

# =========================
# Block detection
# =========================
def group_lines_into_blocks(lines, y_gap=40):
    """Group OCR lines into visual blocks using Y distance"""
    lines = sorted(lines, key=lambda l: bbox_top(l["box"]))
    blocks = []
    current = []

    for ln in lines:
        if not current:
            current.append(ln)
            continue

        prev = current[-1]
        gap = bbox_top(ln["box"]) - bbox_bottom(prev["box"])

        if gap < y_gap:
            current.append(ln)
        else:
            blocks.append(current)
            current = [ln]

    if current:
        blocks.append(current)

    return blocks

# =========================
# Main structuring
# =========================
def structure_ocr(final_json_path, out_path):
    data = json.loads(Path(final_json_path).read_text(encoding="utf-8"))
    lines = data["lines"]

    blocks_raw = group_lines_into_blocks(lines)

    blocks = []
    for i, block in enumerate(blocks_raw, start=1):
        block_bbox = bbox_union([l["box"] for l in block])

        block_lines = []
        for j, l in enumerate(block, start=1):
            block_lines.append({
                "line_id": j,
                "text": l["text"],
                "bbox": [
                    bbox_left(l["box"]),
                    bbox_top(l["box"]),
                    bbox_right(l["box"]),
                    bbox_bottom(l["box"])
                ],
                "confidence": l["conf"]
            })

        blocks.append({
            "block_id": i,
            "bbox": block_bbox,
            "lines": block_lines
        })

    structured = {
        "blocks": blocks,
        "meta": {
            "avg_conf": data.get("avg_conf"),
            "lines_count": len(lines),
            "blocks_count": len(blocks),
            "chosen_source": data.get("chosen")
        }
    }

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(structured, indent=2, ensure_ascii=False), encoding="utf-8")

    return structured

# =========================
# CLI
# =========================
if __name__ == "__main__":
    import sys

    inp = sys.argv[1] if len(sys.argv) > 1 else "decision/final_invoice.json"
    out = sys.argv[2] if len(sys.argv) > 2 else "data/final/ocr_structured.json"

    structure_ocr(inp, out)
    print("✅ Structured OCR JSON saved →", out)
