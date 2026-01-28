#!/usr/bin/env python3
from pathlib import Path
import sys
import json
from datetime import datetime
import argparse

sys.path.insert(0, 'src')

from preprocess.image_enhancer import smart_preprocess
from ocr.ocr_engine import run_ocr_on_image, find_total_in_text

import logging
logging.getLogger('paddleocr').setLevel(logging.CRITICAL)
logging.getLogger('paddlecore').setLevel(logging.CRITICAL)
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def process_images(input_dir: Path, output_dir: Path):
    """
    Process ANY batch of images from ANY source.
    
    Args:
        input_dir: Folder containing images (png, jpg, jpeg, etc.)
        output_dir: Where to save results
    """
    
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    
    # Validate input
    if not input_dir.exists():
        logger.error(f' Input folder not found: {input_dir}')
        return False
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find ALL image types
    image_patterns = ['*.png', '*.jpg', '*.jpeg', '*.JPG', '*.PNG', '*.JPEG']
    all_images = []
    for pattern in image_patterns:
        all_images.extend(sorted(input_dir.glob(pattern)))
    
    # Remove duplicates and sort
    all_images = sorted(set(all_images))
    
    if not all_images:
        logger.error(f'❌ No images found in: {input_dir}')
        return False
    
    logger.info('\n' + '='*70)
    logger.info('🚀 GENERIC PIPELINE - BATCH PROCESSING')
    logger.info('='*70)
    logger.info(f'📁 Input:  {input_dir}')
    logger.info(f'📁 Output: {output_dir}')
    logger.info(f'📸 Found {len(all_images)} images to process\n')
    
    results = {
        'processed': [],
        'failed': [],
        'total': len(all_images),
        'input_dir': str(input_dir),
        'output_dir': str(output_dir),
        'timestamp': datetime.now().isoformat()
    }
    
    for idx, img_path in enumerate(all_images, start=1):
        logger.info(f'[{idx}/{len(all_images)}] {img_path.name}')
        logger.info('-' * 70)
        
        try:
            # STEP 1: Quality check
            enhanced_path = output_dir / f"{img_path.stem}_enhanced.png"
            processed, quality = smart_preprocess(img_path, enhanced_path)
            
            if processed is None:
                logger.error(f'  ❌ Quality check FAILED: {quality.message}')
                results['failed'].append({
                    'file': img_path.name,
                    'reason': quality.message,
                    'brightness': quality.brightness
                })
                continue
            
            logger.info(f'   Quality OK (brightness: {quality.brightness})')
            
            # STEP 2: OCR
            extracted, conf, text = run_ocr_on_image(enhanced_path)
            logger.info(f' OCR: {len(extracted)} lines detected (conf: {conf:.3f})')
            
            # STEP 3: Extract total
            total = find_total_in_text(text)
            logger.info(f'   Total: {total}')
            
            # STEP 4: Save individual JSON
            result_file = output_dir / f"{img_path.stem}_result.json"
            with open(result_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'file': img_path.name,
                    'status': 'SUCCESS',
                    'timestamp': datetime.now().isoformat(),
                    'quality': {
                        'brightness': quality.brightness,
                        'is_acceptable': quality.is_acceptable
                    },
                    'ocr': {
                        'lines_extracted': len(extracted),
                        'avg_confidence': round(conf, 3),
                        'lines': [{'text': e['text'], 'conf': round(e['conf'], 3)} for e in extracted]
                    },
                    'extraction': {
                        'detected_total': total,
                        'full_text': text
                    }
                }, f, ensure_ascii=False, indent=2)
            
            results['processed'].append({
                'file': img_path.name,
                'brightness': quality.brightness,
                'lines': len(extracted),
                'confidence': round(conf, 3),
                'total': total
            })
            
        except Exception as e:
            logger.error(f'   ERROR: {str(e)}')
            results['failed'].append({
                'file': img_path.name,
                'reason': str(e)
            })
    
    # SUMMARY
    logger.info('\n' + '='*70)
    logger.info('📊 BATCH SUMMARY')
    logger.info('='*70)
    logger.info(f'Total:      {results["total"]}')
    logger.info(f'✅ Success: {len(results["processed"])} ({len(results["processed"])/results["total"]*100:.1f}%)')
    logger.info(f'Failed:  {len(results["failed"])} ({len(results["failed"])/results["total"]*100:.1f}%)')
    
    if results['processed']:
        avg_conf = sum(r['confidence'] for r in results['processed']) / len(results['processed'])
        logger.info(f'✓ Avg confidence: {avg_conf:.3f}')
    
    if results['failed']:
        logger.info('\n Failed files:')
        for f in results['failed']:
            logger.info(f'   • {f["file"]}: {f["reason"]}')
    
    # Save batch summary
    summary_file = output_dir / 'batch_summary.json'
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    logger.info(f'\n💾 Summary: {summary_file}')
    logger.info('='*70)
    
    return len(results['failed']) == 0


def main():
    parser = argparse.ArgumentParser(
        description='Generic Pipeline: Process ANY batch of images'
    )
    parser.add_argument('input_dir', help='Input folder with images')
    parser.add_argument('output_dir', help='Output folder for results')
    
    args = parser.parse_args()
    
    success = process_images(Path(args.input_dir), Path(args.output_dir))
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
