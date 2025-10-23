import os
from PIL import Image
from pathlib import Path

def resize_imagenet(input_dir, output_dir, target_size=(224, 224), log_interval=500):
    """
    Resize ImageNet100 images from 500x334 to 224x224.
    
    Args:
        input_dir: Path to input directory containing train/val folders
        output_dir: Path to output directory (will mirror input structure)
        target_size: Target image size (width, height), default (224, 224)
        log_interval: Print progress every N images, default 500
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # Process both train and val directories
    for split in ['train', 'val']:
        split_input = input_path / split
        split_output = output_path / split
        
        if not split_input.exists():
            print(f"Warning: {split_input} does not exist, skipping...")
            continue
        
        # Get all class directories
        class_dirs = [d for d in split_input.iterdir() if d.is_dir()]
        print(f"\nProcessing {split} set: {len(class_dirs)} classes")
        
        total_images = 0
        processed_images = 0
        
        for class_dir in class_dirs:
            class_name = class_dir.name
            output_class_dir = split_output / class_name
            output_class_dir.mkdir(parents=True, exist_ok=True)
            
            # Get all JPEG images in the class directory
            images = list(class_dir.glob('*.JPEG')) + list(class_dir.glob('*.jpg'))
            total_images += len(images)
            
            for img_path in images:
                try:
                    # Open and resize image
                    with Image.open(img_path) as img:
                        # Convert to RGB if necessary (handles grayscale, RGBA, etc.)
                        if img.mode != 'RGB':
                            img = img.convert('RGB')
                        
                        # Resize using high-quality Lanczos resampling
                        img_resized = img.resize(target_size, Image.LANCZOS)
                        
                        # Save to output directory
                        output_img_path = output_class_dir / img_path.name
                        img_resized.save(output_img_path, 'JPEG', quality=95)
                    
                    processed_images += 1
                    
                    # Log progress every log_interval images
                    if processed_images % log_interval == 0:
                        print(f"  [{split}] Processed {processed_images}/{total_images} images "
                              f"({100*processed_images/total_images:.1f}%)")
                
                except Exception as e:
                    print(f"\nError processing {img_path}: {e}")
        
        print(f"  [{split}] Complete: {processed_images}/{total_images} images processed")
    
    print(f"\nResizing complete! Output saved to {output_dir}")

INPUT_DIR = "./ImageNet100"  
OUTPUT_DIR = "./ImageNet100_224"
resize_imagenet(INPUT_DIR, OUTPUT_DIR, target_size=(224, 224))