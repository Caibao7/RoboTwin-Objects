#!/usr/bin/env python3
"""
Script to calculate real sizes for RobotWin objects.
Reads info.json files from UUID directories and calculates real size = dimension * scale * 100 (cm).

Usage:
    python calculate_real_sizes.py [--robotwin-dir PATH] [--output OUTPUT.json]
"""

import json
import os
import sys
import argparse
import re
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple


def is_valid_uuid_dir(directory: Path) -> bool:
    """
    Check if directory name looks like a UUID.
    
    Args:
        directory: Path to check
        
    Returns:
        bool: True if valid UUID directory
    """
    # Basic UUID pattern check (8-4-4-4-12 hex digits)
    name = directory.name
    if len(name) != 36 or name.count('-') != 4:
        return False
    
    # More strict UUID pattern
    uuid_pattern = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.IGNORECASE)
    return bool(uuid_pattern.match(name))


def parse_dimension(dim_str: str) -> Optional[List[float]]:
    """
    Parse dimension string like "1.900209*0.364783*1.251406" into list of floats.
    
    Args:
        dim_str: Dimension string
        
    Returns:
        List of [length, width, height] or None if parsing fails
    """
    if not dim_str:
        return None
        
    try:
        # Split by '*' and convert to floats
        parts = dim_str.strip().split('*')
        if len(parts) != 3:
            return None
            
        dimensions = [float(part.strip()) for part in parts]
        return dimensions
    except (ValueError, AttributeError):
        return None


def parse_scale(scale_data) -> Optional[List[float]]:
    """
    Parse scale data which can be in different formats:
    - List format: [x, y, z]
    - Object format: {"x": val, "y": val, "z": val}
    - Single value: float (uniform scaling)
    
    Args:
        scale_data: Scale data from info.json
        
    Returns:
        List of [x_scale, y_scale, z_scale] or None if parsing fails
    """
    if scale_data is None:
        return None
    
    try:
        # List format: [x, y, z]
        if isinstance(scale_data, list) and len(scale_data) == 3:
            return [float(x) for x in scale_data]
        
        # Object format: {"x": val, "y": val, "z": val}
        elif isinstance(scale_data, dict):
            if all(k in scale_data for k in ['x', 'y', 'z']):
                return [float(scale_data['x']), float(scale_data['y']), float(scale_data['z'])]
        
        # Single value (uniform scaling)
        elif isinstance(scale_data, (int, float)):
            val = float(scale_data)
            return [val, val, val]
        
        return None
    except (ValueError, TypeError):
        return None


def calculate_real_size(dimensions: List[float], scales: List[float]) -> List[float]:
    """
    Calculate real size in cm: real_size = dimension * scale * 100
    
    Args:
        dimensions: [length, width, height] in model units
        scales: [x_scale, y_scale, z_scale] 
        
    Returns:
        [real_length, real_width, real_height] in cm
    """
    if len(dimensions) != 3 or len(scales) != 3:
        raise ValueError("Dimensions and scales must have 3 values each")
    
    real_sizes = []
    for dim, scale in zip(dimensions, scales):
        real_size = dim * scale * 100  # Convert to cm
        real_sizes.append(round(real_size, 2))  # Round to 2 decimal places
    
    return real_sizes


def process_object_info(uuid: str, info_path: Path) -> Optional[Dict[str, Any]]:
    """
    Process a single object's info.json file.
    
    Args:
        uuid: Object UUID
        info_path: Path to info.json file
        
    Returns:
        Dict with processed object data or None if processing fails
    """
    try:
        # Read info.json
        with open(info_path, 'r', encoding='utf-8') as f:
            info_data = json.load(f)
        
        # Extract required fields
        object_name = info_data.get("object_name", "")
        category = info_data.get("category", "")
        dimension_str = info_data.get("dimension", "")
        scale_data = info_data.get("scale")
        
        # Parse dimensions
        dimensions = parse_dimension(dimension_str)
        if dimensions is None:
            return {"error": f"Failed to parse dimensions: '{dimension_str}'"}
        
        # Parse scale
        scales = parse_scale(scale_data)
        if scales is None:
            return {"error": f"Failed to parse scale: {scale_data}"}
        
        # Calculate real size in cm
        real_sizes = calculate_real_size(dimensions, scales)
        
        # Format real size as string (similar to dimension format)
        real_size_str = f"{real_sizes[0]}*{real_sizes[1]}*{real_sizes[2]}"
        
        return {
            "object_name": object_name,
            "category": category,
            "dimension": dimension_str,
            "scale": scales,  # Normalized to list format
            "real_size": real_size_str
        }
        
    except json.JSONDecodeError as e:
        return {"error": f"JSON decode error: {e}"}
    except Exception as e:
        return {"error": f"Processing error: {e}"}


def process_robotwin_objects(robotwin_dir: Path) -> Dict[str, Any]:
    """
    Process all UUID directories in robotwin_objects folder.
    
    Args:
        robotwin_dir: Path to robotwin_objects directory
        
    Returns:
        Dict with processing results
    """
    if not robotwin_dir.exists():
        raise FileNotFoundError(f"Directory not found: {robotwin_dir}")
    
    results = {}
    stats = {
        'total_dirs': 0,
        'uuid_dirs': 0,
        'info_json_found': 0,
        'processed_success': 0,
        'processing_errors': 0,
        'error_details': []
    }
    
    print(f"Processing directory: {robotwin_dir}")
    print("-" * 60)
    
    # Process all subdirectories
    for item in robotwin_dir.iterdir():
        if not item.is_dir():
            continue
            
        stats['total_dirs'] += 1
        
        # Check if it's a valid UUID directory
        if not is_valid_uuid_dir(item):
            continue
            
        stats['uuid_dirs'] += 1
        uuid = item.name
        
        # Check if info.json exists
        info_path = item / "info.json"
        if not info_path.exists():
            continue
            
        stats['info_json_found'] += 1
        
        # Process the object
        result = process_object_info(uuid, info_path)
        
        if result and "error" not in result:
            results[uuid] = result
            stats['processed_success'] += 1
            print(f"✓ {uuid}: {result['object_name']} -> {result['real_size']} cm")
        else:
            stats['processing_errors'] += 1
            error_msg = result.get("error", "Unknown error") if result else "No result"
            stats['error_details'].append(f"{uuid}: {error_msg}")
            print(f"✗ {uuid}: {error_msg}")
    
    return {
        'results': results,
        'stats': stats
    }


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Calculate real sizes for RobotWin objects')
    
    parser.add_argument(
        '--robotwin-dir',
        type=str,
        default='D:/codefield/VLA/objaverse/robotwin_objects/robotwin_objects',
        help='Path to robotwin_objects directory (default: D:/codefield/VLA/objaverse/robotwin_objects/robotwin_objects)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='robotwin_real_sizes.json',
        help='Output JSON file name (default: robotwin_real_sizes.json)'
    )
    
    args = parser.parse_args()
    robotwin_dir = Path(args.robotwin_dir)
    output_path = Path(args.output)
    
    print("=" * 70)
    print("RobotWin Objects Real Size Calculator")
    print("=" * 70)
    print(f"Input directory: {robotwin_dir}")
    print(f"Output file: {output_path}")
    print("=" * 70)
    
    try:
        # Process all objects
        processing_result = process_robotwin_objects(robotwin_dir)
        results = processing_result['results']
        stats = processing_result['stats']
        
        # Save results to JSON file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        # Print summary
        print("\n" + "=" * 60)
        print("PROCESSING SUMMARY:")
        print(f"Total directories scanned: {stats['total_dirs']}")
        print(f"UUID directories found: {stats['uuid_dirs']}")
        print(f"Objects with info.json: {stats['info_json_found']}")
        print(f"Successfully processed: {stats['processed_success']}")
        print(f"Processing errors: {stats['processing_errors']}")
        
        if stats['error_details']:
            print(f"\nError details (first 5):")
            for error in stats['error_details'][:5]:
                print(f"  - {error}")
            if len(stats['error_details']) > 5:
                print(f"  ... and {len(stats['error_details']) - 5} more errors")
        
        print(f"\nResults saved to: {output_path}")
        print(f"Total objects in output: {len(results)}")
        
        return 0
        
    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())