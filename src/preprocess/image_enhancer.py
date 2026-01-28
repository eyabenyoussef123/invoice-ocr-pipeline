import cv2
import numpy as np
from pathlib import Path
import logging
from typing import Tuple, Optional
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


# =========================
# DATA STRUCTURES
# =========================

@dataclass
class QualityReport:
    """Image quality assessment results"""
    is_acceptable: bool
    brightness: float
    message: str


# =========================
# IMAGE LOADING
# =========================

def load_image(image_path: Path) -> np.ndarray:
    """Load image with basic validation"""
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")
    
    logger.info(f"✓ Loaded: {image_path.name} ({image.shape[1]}x{image.shape[0]})")
    return image


# =========================
# SIMPLE QUALITY CHECK
# =========================

def check_brightness(gray: np.ndarray) -> Tuple[float, bool, str]:
    """
    Simple brightness check - very permissive.
    Almost all images pass - only reject if completely black/white.
    
    Returns:
        (brightness_value, is_ok, message)
    """
    mean_brightness = gray.mean()
    
    # VERY permissive: only reject extremes
    if mean_brightness < 5:
        return mean_brightness, False, "Image completely black"
    elif mean_brightness > 254:
        return mean_brightness, False, "Image completely white"
    else:
        return mean_brightness, True, "OK"


def assess_image_quality(image: np.ndarray) -> QualityReport:
    """
    Simple quality check - only brightness.
    
    If brightness is OK, image is ready for OCR.
    Otherwise, ask user to rescan.
    
    Args:
        image: Input image (BGR)
        
    Returns:
        QualityReport with brightness and acceptance status
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    
    brightness, is_ok, message = check_brightness(gray)
    
    return QualityReport(
        is_acceptable=is_ok,
        brightness=round(brightness, 1),
        message=message
    )


# =========================
# SMART PIPELINE
# =========================

def smart_preprocess(
    image_path: Path,
    output_path: Optional[Path] = None
) -> Tuple[Optional[np.ndarray], QualityReport]:
    """
    Simplified preprocessing pipeline.
    
    Strategy:
    1. Load image
    2. Check brightness only
    3. If OK → Return image ready for OCR
    4. If NOT OK → Return None and ask to rescan
    
    No image enhancement, no preprocessing - just quality check.
    
    Args:
        image_path: Input image
        output_path: Where to save (if processing succeeds)
        
    Returns:
        Tuple of (image or None, quality_report)
    """
    logger.info("=" * 60)
    logger.info("🚀 QUALITY CHECK")
    logger.info("=" * 60)
    
    # Load image
    image = load_image(image_path)
    
    # Check brightness
    logger.info("\n📊 Checking brightness...")
    quality = assess_image_quality(image)
    
    logger.info(f"  Brightness: {quality.brightness}")
    
    # Decision point
    if not quality.is_acceptable:
        logger.info("\n" + "=" * 60)
        logger.info("QUALITY CHECK FAILED")
        logger.info("=" * 60)
        logger.info(f"\n⚠️  {quality.message}")
        logger.info("\n📸 Please rescan the document or use a better image")
        logger.info("=" * 60)
        
        return None, quality
    
    # Quality OK → Return original image for OCR
    logger.info("\n✅ Brightness OK")
    
    if output_path:
        cv2.imwrite(str(output_path), image)
        logger.info(f"💾 Saved to: {output_path}")
    
    logger.info("\n" + "=" * 60)
    logger.info("✅ READY FOR OCR")
    logger.info("=" * 60)
    
    return image, quality


# =========================
# BATCH PROCESSING
# =========================

def process_batch(input_dir: Path, output_dir: Path):
    """
    Process multiple invoices with brightness check.
    
    Generates report:
    - How many passed brightness check
    - How many need rescan
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = {
        'passed': [],
        'failed': [],
        'total': 0
    }
    
    png_files = list(input_dir.glob("*.png"))
    jpg_files = list(input_dir.glob("*.jpg"))
    all_files = png_files + jpg_files
    
    for img_path in all_files:
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing: {img_path.name}")
        
        processed, quality = smart_preprocess(
            img_path,
            output_dir / f"{img_path.stem}_ready.png"
        )
        
        results['total'] += 1
        
        if processed is not None:
            results['passed'].append({
                'file': img_path.name,
                'brightness': quality.brightness
            })
        else:
            results['failed'].append({
                'file': img_path.name,
                'brightness': quality.brightness,
                'message': quality.message
            })
    
    # Summary report
    logger.info("\n" + "=" * 60)
    logger.info(" BATCH PROCESSING SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total images:     {results['total']}")
    logger.info(f" Passed:        {len(results['passed'])} ({len(results['passed'])/results['total']*100:.1f}%)")
    logger.info(f" Need rescan:   {len(results['failed'])} ({len(results['failed'])/results['total']*100:.1f}%)")
    
    if results['failed']:
        logger.info(f"\n Images requiring rescan:")
        for fail in results['failed']:
            logger.info(f"  • {fail['file']}: {fail['message']}")
    
    return results


# =========================
# CLI USAGE
# =========================

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage:")
        print("  Single image: python image_enhancer.py <image.png>")
        print("  Batch:        python image_enhancer.py <input_dir> <output_dir>")
        sys.exit(1)
    
    input_path = Path(sys.argv[1])
    
    if input_path.is_dir():
        # Batch mode
        output_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("data/processed")
        process_batch(input_path, output_dir)
    else:
        # Single image mode
        output_path = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("data/processed") / f"{input_path.stem}_ready.png"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        processed, quality = smart_preprocess(input_path, output_path)
        
        if processed is None:
            sys.exit(1)  # Failed quality check