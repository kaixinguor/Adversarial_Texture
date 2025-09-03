#!/usr/bin/env python3
"""
Convert PASCAL format annotations to YOLO format for COCO person dataset
"""

import os
import re
import glob
from pathlib import Path

def parse_pascal_annotation(annotation_file):
    """
    Parse PASCAL format annotation file and extract bounding box information
    
    Args:
        annotation_file (str): Path to the PASCAL annotation file
        
    Returns:
        list: List of bounding boxes in format [(class_id, x_center, y_center, width, height), ...]
    """
    # Try different encodings
    encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
    content = None
    
    for encoding in encodings:
        try:
            with open(annotation_file, 'r', encoding=encoding) as f:
                content = f.read()
            break
        except UnicodeDecodeError:
            continue
    
    if content is None:
        raise ValueError(f"Could not read {annotation_file} with any of the attempted encodings")
    
    # Extract image dimensions
    size_match = re.search(r'Image size \(X x Y x C\) : (\d+) x (\d+) x (\d+)', content)
    if not size_match:
        raise ValueError(f"Could not find image size in {annotation_file}")
    
    img_width = int(size_match.group(1))
    img_height = int(size_match.group(2))
    
    # Extract bounding boxes
    bbox_pattern = r'Bounding box for object \d+ "PASperson" \(Xmin, Ymin\) - \(Xmax, Ymax\) : \((\d+), (\d+)\) - \((\d+), (\d+)\)'
    bbox_matches = re.findall(bbox_pattern, content)
    
    yolo_boxes = []
    for bbox_match in bbox_matches:
        xmin, ymin, xmax, ymax = map(int, bbox_match)
        
        # Convert to YOLO format (normalized coordinates)
        x_center = (xmin + xmax) / 2.0 / img_width
        y_center = (ymin + ymax) / 2.0 / img_height
        width = (xmax - xmin) / img_width
        height = (ymax - ymin) / img_height
        
        # Class ID for person (assuming 0 for person class)
        class_id = 0
        
        yolo_boxes.append((class_id, x_center, y_center, width, height))
    
    return yolo_boxes

def convert_annotations_to_yolo(input_dir, output_dir):
    """
    Convert all PASCAL annotations in input_dir to YOLO format and save to output_dir
    
    Args:
        input_dir (str): Directory containing PASCAL annotation files
        output_dir (str): Directory to save YOLO format annotation files
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all annotation files
    annotation_files = glob.glob(os.path.join(input_dir, "*.txt"))
    
    print(f"Found {len(annotation_files)} annotation files")
    
    converted_count = 0
    error_count = 0
    
    for annotation_file in annotation_files:
        try:
            # Parse the annotation file
            yolo_boxes = parse_pascal_annotation(annotation_file)
            
            if not yolo_boxes:
                print(f"Warning: No bounding boxes found in {annotation_file}")
                continue
            
            # Create output filename (same name as input but in output directory)
            filename = os.path.basename(annotation_file)
            output_file = os.path.join(output_dir, filename)
            
            # Write YOLO format annotations
            with open(output_file, 'w') as f:
                for class_id, x_center, y_center, width, height in yolo_boxes:
                    f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
            
            converted_count += 1
            print(f"Converted {filename}: {len(yolo_boxes)} objects")
            
        except Exception as e:
            error_count += 1
            print(f"Error converting {annotation_file}: {str(e)}")
    
    print(f"\nConversion completed!")
    print(f"Successfully converted: {converted_count} files")
    print(f"Errors: {error_count} files")
    print(f"Output directory: {output_dir}")

def main():
    """Main function"""
    # Define paths
    input_dir = "data/INRIAPerson/Train/annotations"
    output_dir = "temp/INRIAPerson/Train/yolo_labels"
    
    print("Converting PASCAL annotations to YOLO format...")
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print("-" * 50)
    
    # Check if input directory exists
    if not os.path.exists(input_dir):
        print(f"Error: Input directory {input_dir} does not exist!")
        return
    
    # Convert annotations
    convert_annotations_to_yolo(input_dir, output_dir)

if __name__ == "__main__":
    main()
