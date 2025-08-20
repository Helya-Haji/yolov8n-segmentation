import os
import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

dataset_path = r""

def inspect_annotations(dataset_path, num_samples=300):
    """Inspect annotation quality by visualizing images with their masks"""
    
    train_images_dir = os.path.join(dataset_path, 'train', 'images')
    train_labels_dir = os.path.join(dataset_path, 'train', 'labels')
    
    if not os.path.exists(train_images_dir) or not os.path.exists(train_labels_dir):
        print("❌ Train directories not found!")
        return
    
    # Get image files
    image_files = [f for f in os.listdir(train_images_dir) 
                   if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
    
    print(f"🔍 Inspecting {min(num_samples, len(image_files))} training samples...")
    
    issues_found = []
    
    for i, img_file in enumerate(image_files[:num_samples]):
        print(f"\n📋 Sample {i+1}: {img_file}")
        
        # Load image
        img_path = os.path.join(train_images_dir, img_file)
        image = cv2.imread(img_path)
        if image is None:
            print(f"  ❌ Cannot load image: {img_file}")
            issues_found.append(f"Cannot load image: {img_file}")
            continue
        
        h, w = image.shape[:2]
        print(f"  📐 Image size: {w}x{h}")
        
        # Check corresponding label file
        label_file = os.path.splitext(img_file)[0] + '.txt'
        label_path = os.path.join(train_labels_dir, label_file)
        
        if not os.path.exists(label_path):
            print(f"  ❌ Missing label file: {label_file}")
            issues_found.append(f"Missing label: {label_file}")
            continue
        
        # Read label file
        try:
            with open(label_path, 'r') as f:
                lines = f.readlines()
            
            if not lines:
                print(f"  ❌ Empty label file: {label_file}")
                issues_found.append(f"Empty label: {label_file}")
                continue
            
            print(f"  📝 Label lines: {len(lines)}")
            
            # Parse segmentation annotations
            for line_idx, line in enumerate(lines):
                parts = line.strip().split()
                if len(parts) < 6:  # class_id + at least 2 points (4 coordinates)
                    print(f"    ❌ Line {line_idx+1}: Too few coordinates ({len(parts)})")
                    issues_found.append(f"{img_file} line {line_idx+1}: Too few coordinates")
                    continue
                
                class_id = int(parts[0])
                coords = [float(x) for x in parts[1:]]
                
                if len(coords) % 2 != 0:
                    print(f"    ❌ Line {line_idx+1}: Odd number of coordinates")
                    issues_found.append(f"{img_file} line {line_idx+1}: Odd coordinates")
                    continue
                
                # Check if coordinates are normalized (0-1)
                if any(c < 0 or c > 1 for c in coords):
                    print(f"    ❌ Line {line_idx+1}: Coordinates not normalized (0-1)")
                    issues_found.append(f"{img_file} line {line_idx+1}: Not normalized")
                
                # Check polygon area (rough estimate)
                points = np.array(coords).reshape(-1, 2)
                points_pixel = points * np.array([w, h])
                area = cv2.contourArea(points_pixel.astype(np.int32))
                area_ratio = area / (w * h)
                
                print(f"    📊 Polygon {line_idx+1}: {len(points)} points, area={area_ratio:.3%}")
                
                if area_ratio < 0.001:  # Very small area
                    print(f"    ⚠️ Very small annotation area: {area_ratio:.3%}")
                    issues_found.append(f"{img_file}: Very small area {area_ratio:.3%}")
                
                if area_ratio > 0.8:  # Very large area
                    print(f"    ⚠️ Very large annotation area: {area_ratio:.3%}")
                    issues_found.append(f"{img_file}: Very large area {area_ratio:.3%}")
        
        except Exception as e:
            print(f"  ❌ Error reading label: {e}")
            issues_found.append(f"Error reading {label_file}: {e}")
    
    # Summary
    print(f"\n📊 INSPECTION SUMMARY:")
    print(f"  Total samples checked: {min(num_samples, len(image_files))}")
    print(f"  Issues found: {len(issues_found)}")
    
    if issues_found:
        print(f"\n❌ ISSUES FOUND:")
        for issue in issues_found[:10]:  # Show first 10 issues
            print(f"  - {issue}")
        if len(issues_found) > 10:
            print(f"  ... and {len(issues_found) - 10} more issues")
    else:
        print(f"✅ No obvious issues found in the samples checked")
    
    return len(issues_found) == 0

def visualize_annotations(dataset_path, num_samples=5):
    """Visualize images with their annotations overlaid"""
    
    train_images_dir = os.path.join(dataset_path, 'train', 'images')
    train_labels_dir = os.path.join(dataset_path, 'train', 'labels')
    
    image_files = [f for f in os.listdir(train_images_dir) 
                   if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
    
    fig, axes = plt.subplots(2, min(num_samples, 3), figsize=(15, 10))
    if num_samples == 1:
        axes = axes.reshape(2, 1)
    
    for i, img_file in enumerate(image_files[:num_samples]):
        if i >= 3:  # Limit to 3 for display
            break
            
        # Load image
        img_path = os.path.join(train_images_dir, img_file)
        image = cv2.imread(img_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]
        
        # Load label
        label_file = os.path.splitext(img_file)[0] + '.txt'
        label_path = os.path.join(train_labels_dir, label_file)
        
        # Create mask
        mask = np.zeros((h, w), dtype=np.uint8)
        
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                lines = f.readlines()
            
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 6:
                    coords = [float(x) for x in parts[1:]]
                    if len(coords) % 2 == 0:
                        points = np.array(coords).reshape(-1, 2)
                        points_pixel = (points * np.array([w, h])).astype(np.int32)
                        cv2.fillPoly(mask, [points_pixel], 255)
        
        # Display original image
        axes[0, i].imshow(image_rgb)
        axes[0, i].set_title(f'Original: {img_file}')
        axes[0, i].axis('off')
        
        # Display image with mask overlay
        overlay = image_rgb.copy()
        overlay[mask > 0] = [255, 0, 0]  # Red overlay for mask
        blended = cv2.addWeighted(image_rgb, 0.7, overlay, 0.3, 0)
        
        axes[1, i].imshow(blended)
        axes[1, i].set_title(f'With Annotations')
        axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.savefig('annotation_inspection.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("💾 Visualization saved as 'annotation_inspection.png'")

if __name__ == "__main__":
    print("🔍 DATASET QUALITY INSPECTION")
    print("="*50)
    
    # Step 1: Inspect annotations
    is_clean = inspect_annotations(dataset_path, num_samples=20)
    
    # Step 2: Visualize some samples
    print(f"\n🎨 Creating visualizations...")
    visualize_annotations(dataset_path, num_samples=5)
    
    if not is_clean:
        print(f"\n⚠️ RECOMMENDATION: Clean your dataset before training!")
        print(f"   - Fix or remove images with incorrect annotations")
        print(f"   - Ensure all femur bones are properly segmented")
        print(f"   - Check for missing or empty label files")
    else:
        print(f"\n✅ Dataset looks good for training!")
