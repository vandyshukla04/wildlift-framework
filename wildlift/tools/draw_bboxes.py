#!/usr/bin/env python3
"""
Script to draw bounding boxes on images based on JSON annotation files.
Reads JSON files with format: {image_name}_results.json
Draws bounding boxes and saves annotated images.
"""

import json
import os
import argparse
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont


def draw_bbox_on_image(image_path, json_path, output_path, box_color='red', box_width=3, class_filter=None):
    """
    Draw bounding boxes on an image based on JSON annotations.
    
    Args:
        image_path: Path to the input image
        json_path: Path to the JSON annotation file
        output_path: Path where the annotated image will be saved
        box_color: Color of the bounding box (default: 'red')
        box_width: Width of the bounding box lines (default: 3)
        class_filter: Class name to filter (case-insensitive, None = draw all classes)
    """
    # Load the JSON annotation
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Load the image
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)
    
    # Try to load a font for labels, fall back to default if not available
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
    except:
        font = ImageFont.load_default()
    
    # Draw each annotation
    total_annotations = len(data.get('annotations', []))
    drawn_annotations = 0
    
    for annotation in data.get('annotations', []):
        bbox = annotation['bbox']
        class_name = annotation.get('class_name', 'object')
        
        # Apply class filter if specified (case-insensitive matching)
        if class_filter is not None:
            if class_name.lower() != class_filter.lower():
                continue  # Skip this annotation
        
        # Handle score as either list or scalar (type-safe extraction)
        raw_score = annotation.get('score', 0)
        if isinstance(raw_score, list):
            score = raw_score[0] if len(raw_score) > 0 else 0.0
        else:
            score = float(raw_score) if raw_score else 0.0
        
        # bbox format is [x1, y1, x2, y2] in xyxy format
        x1, y1, x2, y2 = bbox
        
        # Draw the bounding box
        draw.rectangle([x1, y1, x2, y2], outline=box_color, width=box_width)
        
        # Draw label with class name and confidence score
        label = f"{class_name}: {score:.2f}"
        
        # Draw label background
        bbox_text = draw.textbbox((x1, y1 - 25), label, font=font)
        draw.rectangle(bbox_text, fill=box_color)
        
        # Draw label text
        draw.text((x1, y1 - 25), label, fill='white', font=font)
        
        drawn_annotations += 1
    
    # Save the annotated image
    image.save(output_path)
    
    # Provide informative feedback
    if class_filter:
        print(f"Saved annotated image: {output_path} ({drawn_annotations}/{total_annotations} annotations - filtered for '{class_filter}')")
    else:
        print(f"Saved annotated image: {output_path} ({drawn_annotations} annotations)")


def process_directory(image_dir, json_dir, output_dir, box_color='red', box_width=3, class_filter=None):
    """
    Process all images in a directory and draw bounding boxes.
    
    Args:
        image_dir: Directory containing input images
        json_dir: Directory containing JSON annotation files
        output_dir: Directory where annotated images will be saved
        box_color: Color of the bounding box (default: 'red')
        box_width: Width of the bounding box lines (default: 3)
        class_filter: Class name to filter (case-insensitive, None = draw all classes)
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    image_files = [f for f in os.listdir(image_dir) 
                   if Path(f).suffix.lower() in image_extensions]
    
    if not image_files:
        print(f"No image files found in {image_dir}")
        return
    
    print(f"Found {len(image_files)} images to process")
    
    processed = 0
    skipped = 0
    
    for image_file in image_files:
        # Construct paths
        image_path = os.path.join(image_dir, image_file)
        image_stem = Path(image_file).stem
        json_file = f"{image_stem}_results.json"
        json_path = os.path.join(json_dir, json_file)
        output_path = os.path.join(output_dir, image_file)
        
        # Check if JSON file exists
        if not os.path.exists(json_path):
            print(f"Warning: JSON file not found for {image_file} (expected: {json_file})")
            skipped += 1
            continue
        
        try:
            draw_bbox_on_image(image_path, json_path, output_path, box_color, box_width, class_filter)
            processed += 1
        except Exception as e:
            print(f"Error processing {image_file}: {str(e)}")
            skipped += 1
    
    print(f"\nProcessing complete!")
    print(f"Successfully processed: {processed}")
    print(f"Skipped: {skipped}")


def main():
    parser = argparse.ArgumentParser(
        description='Draw bounding boxes on images based on JSON annotations',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all images in a directory (draw all classes)
  python draw_bboxes.py --image-dir ./images --json-dir ./labels --output-dir ./annotated
  
  # Process all images but only draw 'rhino' class bounding boxes
  python draw_bboxes.py --image-dir ./images --json-dir ./labels --output-dir ./annotated --class-name rhino
  
  # Process a single image (all classes)
  python draw_bboxes.py --image ./images/1200.jpg --json ./labels/1200_results.json --output ./annotated/1200.jpg
  
  # Process a single image (filter for 'ground' class only)
  python draw_bboxes.py --image ./images/1200.jpg --json ./labels/1200_results.json --output ./annotated/1200.jpg --class-name ground
  
  # Change box color and width
  python draw_bboxes.py --image-dir ./images --json-dir ./labels --output-dir ./annotated --color blue --width 5
  
  # Combine filtering with custom styling
  python draw_bboxes.py --image-dir ./images --json-dir ./labels --output-dir ./annotated --class-name rhino --color green --width 4
        """
    )
    
    # Add arguments for directory processing
    parser.add_argument('--image-dir', type=str, help='Directory containing input images')
    parser.add_argument('--json-dir', type=str, help='Directory containing JSON annotation files')
    parser.add_argument('--output-dir', type=str, help='Directory to save annotated images')
    
    # Add arguments for single file processing
    parser.add_argument('--image', type=str, help='Single input image file')
    parser.add_argument('--json', type=str, help='Single JSON annotation file')
    parser.add_argument('--output', type=str, help='Output path for single annotated image')
    
    # Styling options
    parser.add_argument('--color', type=str, default='red', 
                       help='Bounding box color (default: red)')
    parser.add_argument('--width', type=int, default=3, 
                       help='Bounding box line width (default: 3)')
    
    # Filtering options
    parser.add_argument('--class-name', type=str, default=None,
                       help='Filter annotations by class name (case-insensitive, default: draw all classes)')
    
    args = parser.parse_args()
    
    # Check if processing directory or single file
    if args.image_dir and args.json_dir and args.output_dir:
        process_directory(args.image_dir, args.json_dir, args.output_dir, 
                         args.color, args.width, args.class_name)
    elif args.image and args.json and args.output:
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        draw_bbox_on_image(args.image, args.json, args.output, 
                          args.color, args.width, args.class_name)
    else:
        parser.print_help()
        print("\nError: Please provide either:")
        print("  1. --image-dir, --json-dir, and --output-dir for batch processing")
        print("  2. --image, --json, and --output for single file processing")


if __name__ == "__main__":
    main()