import numpy as np
import cv2
import os
import random
from PIL import Image, ImageDraw, ImageFilter
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import argparse
from pathlib import Path
from datetime import datetime

class NoiseImageGeneratorForDetector:
    """
    Image generator specifically designed for the FIXED Noise Detection System
    Creates test images matching the exact noise models used in training
    """
    
    def __init__(self, output_dir="detector_test_images", image_size=(128, 128)):
        self.output_dir = output_dir
        self.image_size = image_size
        self.noise_types = ["gaussian", "salt_pepper", "speckle", "detector_striping", "detector_banding"]
        
        # Noise parameters matching the detector's training
        self.noise_configs = {
            'gaussian': [
                {'std': 5, 'mean': 0},
                {'std': 12, 'mean': 0},
                {'std': 20, 'mean': 0},
                {'std': 30, 'mean': 0},
                {'std': 8, 'mean': 2},
                {'std': 15, 'mean': -1},
                {'std': 25, 'mean': 3}
            ],
            'salt_pepper': [
                {'density': 0.01, 'salt_ratio': 0.5},
                {'density': 0.03, 'salt_ratio': 0.5},
                {'density': 0.07, 'salt_ratio': 0.5},
                {'density': 0.12, 'salt_ratio': 0.5},
                {'density': 0.02, 'salt_ratio': 0.6},
                {'density': 0.05, 'salt_ratio': 0.4},
                {'density': 0.08, 'salt_ratio': 0.7}
            ],
            'speckle': [
                {'var': 0.08},
                {'var': 0.2},
                {'var': 0.35},
                {'var': 0.5},
                {'var': 0.15},
                {'var': 0.25},
                {'var': 0.4}
            ],
            'detector_striping': [
                {'amp': 15, 'period': 6, 'direction': 'vertical'},
                {'amp': 30, 'period': 10, 'direction': 'horizontal'},
                {'amp': 45, 'period': 14, 'direction': 'vertical'},
                {'amp': 60, 'period': 20, 'direction': 'horizontal'},
                {'amp': 25, 'period': 8, 'direction': 'vertical'},
                {'amp': 40, 'period': 12, 'direction': 'horizontal'},
                {'amp': 50, 'period': 16, 'direction': 'vertical'}
            ],
            'detector_banding': [
                {'intensity': 10, 'num_bands': 4, 'direction': 'vertical'},
                {'intensity': 20, 'num_bands': 6, 'direction': 'horizontal'},
                {'intensity': 15, 'num_bands': 8, 'direction': 'vertical'},
                {'intensity': 25, 'num_bands': 5, 'direction': 'horizontal'},
                {'intensity': 12, 'num_bands': 7, 'direction': 'vertical'},
                {'intensity': 18, 'num_bands': 9, 'direction': 'horizontal'},
                {'intensity': 22, 'num_bands': 6, 'direction': 'vertical'}
            ]
        }
        
        self._create_directories()
    
    def _create_directories(self):
        """Create the required directory structure"""
        base_path = Path(self.output_dir)
        base_path.mkdir(exist_ok=True)
        
        for noise_type in self.noise_types:
            noise_path = base_path / noise_type
            noise_path.mkdir(exist_ok=True)
            print(f"Created directory: {noise_path}")
        
        # Create additional test directories
        (base_path / "mixed_test").mkdir(exist_ok=True)
        (base_path / "clean_images").mkdir(exist_ok=True)
        print(f"Created test directories in: {base_path}")
    
    def generate_base_image(self, image_id, pattern_type=None):
        """Generate base images matching detector training patterns"""
        # Use image_id for reproducible randomness
        np.random.seed(image_id % 10000)
        random.seed(image_id % 10000)
        
        if pattern_type is None:
            pattern_type = random.choice([
                'uniform_low', 'uniform_mid', 'uniform_high',
                'gradient_horizontal', 'gradient_vertical', 'gradient_diagonal', 'gradient_radial',
                'checkerboard', 'circles', 'rectangles', 'texture',
                'mnist_like', 'natural_like', 'synthetic_complex'
            ])
        
        img = np.zeros(self.image_size, dtype=np.uint8)
        
        if pattern_type == 'uniform_low':
            img.fill(64)
        elif pattern_type == 'uniform_mid':
            img.fill(128)
        elif pattern_type == 'uniform_high':
            img.fill(192)
            
        elif pattern_type == 'gradient_horizontal':
            for i in range(self.image_size[1]):
                intensity = int(255 * i / self.image_size[1])
                img[:, i] = intensity
                
        elif pattern_type == 'gradient_vertical':
            for i in range(self.image_size[0]):
                intensity = int(255 * i / self.image_size[0])
                img[i, :] = intensity
                
        elif pattern_type == 'gradient_diagonal':
            for i in range(self.image_size[0]):
                for j in range(self.image_size[1]):
                    intensity = int(255 * (i + j) / (self.image_size[0] + self.image_size[1]))
                    img[i, j] = intensity
                    
        elif pattern_type == 'gradient_radial':
            center_x, center_y = self.image_size[1]//2, self.image_size[0]//2
            max_dist = np.sqrt(center_x**2 + center_y**2)
            for i in range(self.image_size[0]):
                for j in range(self.image_size[1]):
                    dist = np.sqrt((j - center_x)**2 + (i - center_y)**2)
                    intensity = int(255 * dist / max_dist)
                    img[i, j] = min(intensity, 255)
                    
        elif pattern_type == 'checkerboard':
            square_size = random.randint(8, 32)
            for i in range(0, self.image_size[0], square_size):
                for j in range(0, self.image_size[1], square_size):
                    if (i // square_size + j // square_size) % 2 == 0:
                        img[i:i+square_size, j:j+square_size] = random.randint(180, 255)
                    else:
                        img[i:i+square_size, j:j+square_size] = random.randint(0, 75)
                        
        elif pattern_type == 'circles':
            img.fill(random.randint(50, 100))
            num_circles = random.randint(3, 8)
            for _ in range(num_circles):
                center_x = random.randint(0, self.image_size[1])
                center_y = random.randint(0, self.image_size[0])
                radius = random.randint(15, 50)
                intensity = random.randint(150, 255)
                
                y, x = np.ogrid[:self.image_size[0], :self.image_size[1]]
                mask = (x - center_x)**2 + (y - center_y)**2 <= radius**2
                img[mask] = intensity
                
        elif pattern_type == 'rectangles':
            img.fill(random.randint(30, 80))
            num_rects = random.randint(4, 10)
            for _ in range(num_rects):
                x1 = random.randint(0, self.image_size[1]//2)
                y1 = random.randint(0, self.image_size[0]//2)
                x2 = random.randint(x1, self.image_size[1])
                y2 = random.randint(y1, self.image_size[0])
                intensity = random.randint(120, 255)
                img[y1:y2, x1:x2] = intensity
                
        elif pattern_type == 'texture':
            # Create textured pattern
            base_level = random.randint(80, 150)
            noise = np.random.normal(0, 20, self.image_size)
            img = np.clip(base_level + noise, 0, 255).astype(np.uint8)
            
        elif pattern_type == 'mnist_like':
            # Create MNIST-like patterns
            img.fill(0)
            # Simple digit-like patterns
            center_x, center_y = self.image_size[1]//2, self.image_size[0]//2
            thickness = random.randint(8, 16)
            
            # Draw simple shapes
            shape_type = random.choice(['circle', 'line', 'rectangle'])
            if shape_type == 'circle':
                cv2.circle(img, (center_x, center_y), random.randint(20, 40), 255, thickness)
            elif shape_type == 'line':
                pt1 = (random.randint(10, 30), random.randint(10, 30))
                pt2 = (random.randint(90, 110), random.randint(90, 110))
                cv2.line(img, pt1, pt2, 255, thickness)
            else:  # rectangle
                pt1 = (random.randint(10, 30), random.randint(10, 30))
                pt2 = (random.randint(90, 110), random.randint(90, 110))
                cv2.rectangle(img, pt1, pt2, 255, thickness)
                
        elif pattern_type == 'natural_like':
            # Create natural image-like patterns
            # Multi-scale noise to simulate natural textures
            img_float = np.zeros(self.image_size, dtype=np.float64)
            
            for scale in [1, 2, 4, 8]:
                noise_level = np.random.normal(0, 30/scale, 
                    (self.image_size[0]//scale, self.image_size[1]//scale))
                noise_resized = cv2.resize(noise_level, self.image_size[::-1])
                img_float += noise_resized
            
            img_float += random.randint(80, 150)
            img = np.clip(img_float, 0, 255).astype(np.uint8)
            
        else:  # synthetic_complex
            # Complex synthetic pattern
            x = np.linspace(0, 4*np.pi, self.image_size[1])
            y = np.linspace(0, 4*np.pi, self.image_size[0])
            X, Y = np.meshgrid(x, y)
            
            pattern = 128 + 64 * (np.sin(X) * np.cos(Y) + 0.5 * np.sin(2*X + Y))
            img = np.clip(pattern, 0, 255).astype(np.uint8)
        
        # Reset random seeds
        np.random.seed(None)
        random.seed(None)
        
        return img
    
    def add_gaussian_noise(self, image, params):
        """Add Gaussian noise matching detector's exact model"""
        img_float = image.astype(np.float64)
        noise = np.random.normal(params['mean'], params['std'], img_float.shape)
        noisy = img_float + noise
        return np.clip(noisy, 0, 255).astype(np.uint8)
    
    def add_salt_pepper_noise(self, image, params):
        """Add salt & pepper noise matching detector's exact model"""
        noisy = image.copy()
        total_pixels = image.size
        
        # Generate random locations for noise
        num_noisy_pixels = int(params['density'] * total_pixels)
        corrupted_indices = np.random.choice(total_pixels, num_noisy_pixels, replace=False)
        corrupted_coords = np.unravel_index(corrupted_indices, image.shape)
        
        # Split between salt and pepper based on ratio
        num_salt = int(num_noisy_pixels * params['salt_ratio'])
        
        # Salt (white) pixels
        salt_coords = (corrupted_coords[0][:num_salt], corrupted_coords[1][:num_salt])
        noisy[salt_coords] = 255
        
        # Pepper (black) pixels
        pepper_coords = (corrupted_coords[0][num_salt:], corrupted_coords[1][num_salt:])
        noisy[pepper_coords] = 0
        
        return noisy
    
    def add_speckle_noise(self, image, params):
        """Add speckle noise matching detector's exact model"""
        img_float = image.astype(np.float64)
        # Speckle: I_out = I_in * (1 + n) where n ~ N(0, variance)
        multiplicative_noise = np.random.normal(0, np.sqrt(params['var']), img_float.shape)
        noisy = img_float * (1 + multiplicative_noise)
        return np.clip(noisy, 0, 255).astype(np.uint8)
    
    def add_detector_striping(self, image, params):
        """Add detector striping matching detector's exact model"""
        img_float = image.astype(np.float64)
        h, w = img_float.shape
        
        if params['direction'] == 'horizontal':
            # Horizontal stripes
            stripes = params['amp'] * np.sin(2 * np.pi * np.arange(h) / params['period'])
            stripes = stripes[:, np.newaxis]
        else:
            # Vertical stripes
            stripes = params['amp'] * np.sin(2 * np.pi * np.arange(w) / params['period'])
            stripes = stripes[np.newaxis, :]
        
        noisy = img_float + stripes
        return np.clip(noisy, 0, 255).astype(np.uint8)
    
    def add_detector_banding(self, image, params):
        """Add detector banding matching detector's exact model"""
        img_float = image.astype(np.float64)
        h, w = img_float.shape
        
        if params['direction'] == 'vertical':
            # Vertical banding
            band_width = w // params['num_bands']
            for i in range(params['num_bands']):
                start_col = i * band_width
                end_col = min((i + 1) * band_width, w)
                band_offset = random.uniform(-params['intensity'], params['intensity'])
                img_float[:, start_col:end_col] += band_offset
        else:
            # Horizontal banding
            band_height = h // params['num_bands']
            for i in range(params['num_bands']):
                start_row = i * band_height
                end_row = min((i + 1) * band_height, h)
                band_offset = random.uniform(-params['intensity'], params['intensity'])
                img_float[start_row:end_row, :] += band_offset
        
        return np.clip(img_float, 0, 255).astype(np.uint8)
    
    def generate_test_dataset(self, images_per_type=40, base_patterns=None):
        """Generate comprehensive test dataset"""
        print(f"🎯 Generating test dataset for FIXED Noise Detection System")
        print(f"📊 Images per noise type: {images_per_type}")
        print(f"🔧 Noise models: Matching detector training exactly")
        print(f"👤 User: vkhare2909")
        print(f"📅 Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        noise_functions = {
            'gaussian': self.add_gaussian_noise,
            'salt_pepper': self.add_salt_pepper_noise,
            'speckle': self.add_speckle_noise,
            'detector_striping': self.add_detector_striping,
            'detector_banding': self.add_detector_banding
        }
        
        total_generated = 0
        
        # Generate clean reference images first
        print(f"\n📸 Generating clean reference images...")
        for i in range(20):  # Generate 20 clean images
            base_img = self.generate_base_image(i)
            filename = f"clean_reference_{i+1:03d}.png"
            filepath = os.path.join(self.output_dir, "clean_images", filename)
            cv2.imwrite(filepath, base_img)
        
        print(f"   ✅ Generated 20 clean reference images")
        
        # Generate noisy images for each type
        for noise_type in self.noise_types:
            print(f"\n🔄 Generating {noise_type} images...")
            
            # Map detector_striping and detector_banding to actual functions
            if noise_type == 'detector_striping':
                actual_function = self.add_detector_striping
                configs = self.noise_configs['detector_striping']
            elif noise_type == 'detector_banding':
                actual_function = self.add_detector_banding
                configs = self.noise_configs['detector_banding']
            else:
                actual_function = noise_functions[noise_type]
                configs = self.noise_configs[noise_type]
            
            for i in range(images_per_type):
                # Generate base image with variety
                base_img = self.generate_base_image(i * 1000 + hash(noise_type) % 1000)
                
                # Select noise configuration
                config = configs[i % len(configs)]
                
                # Add noise
                noisy_img = actual_function(base_img, config)
                
                # Save image
                filename = f"{noise_type}_test_{i+1:03d}.png"
                filepath = os.path.join(self.output_dir, noise_type, filename)
                cv2.imwrite(filepath, noisy_img)
                
                total_generated += 1
                
                if (i + 1) % 10 == 0:
                    print(f"   Generated {i+1}/{images_per_type} {noise_type} images")
        
        # Generate mixed test cases
        print(f"\n🔀 Generating mixed test cases...")
        for i in range(20):
            base_img = self.generate_base_image(i + 50000)
            
            # Randomly select two noise types and combine them
            noise_types = random.sample(list(self.noise_configs.keys()), 2)
            
            # Apply first noise
            config1 = random.choice(self.noise_configs[noise_types[0]])
            if noise_types[0] == 'detector_striping':
                noisy_img = self.add_detector_striping(base_img, config1)
            elif noise_types[0] == 'detector_banding':
                noisy_img = self.add_detector_banding(base_img, config1)
            else:
                noisy_img = noise_functions[noise_types[0]](base_img, config1)
            
            # Apply second noise (with reduced intensity)
            config2 = random.choice(self.noise_configs[noise_types[1]])
            if noise_types[1] in config2:
                for key in config2:
                    if isinstance(config2[key], (int, float)):
                        config2[key] *= 0.5  # Reduce intensity for mixed case
            
            if noise_types[1] == 'detector_striping':
                final_img = self.add_detector_striping(noisy_img, config2)
            elif noise_types[1] == 'detector_banding':
                final_img = self.add_detector_banding(noisy_img, config2)
            else:
                final_img = noise_functions[noise_types[1]](noisy_img, config2)
            
            filename = f"mixed_{noise_types[0]}_{noise_types[1]}_{i+1:03d}.png"
            filepath = os.path.join(self.output_dir, "mixed_test", filename)
            cv2.imwrite(filepath, final_img)
        
        print(f"   ✅ Generated 20 mixed test cases")
        
        # Create summary report
        self.create_generation_report(images_per_type, total_generated)
        
        print(f"\n🎉 Test dataset generation complete!")
        print(f"📊 Total images generated: {total_generated + 40}")  # +40 for clean and mixed
        print(f"📁 Saved to: {self.output_dir}")
        print(f"🔧 Ready for FIXED Noise Detection System testing")
    
    def create_generation_report(self, images_per_type, total_generated):
        """Create a detailed generation report"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        report_data = {
            'generation_info': {
                'timestamp': datetime.now().isoformat(),
                'user': 'vkhare2909',
                'system': 'NoiseImageGeneratorForDetector',
                'detector_target': 'FIXED Noise Detection System',
                'version': '1.0'
            },
            'dataset_structure': {
                'images_per_noise_type': images_per_type,
                'total_noisy_images': total_generated,
                'clean_reference_images': 20,
                'mixed_test_cases': 20,
                'total_images': total_generated + 40
            },
            'noise_configurations': self.noise_configs,
            'directory_structure': {
                'base': self.output_dir,
                'subdirectories': self.noise_types + ['clean_images', 'mixed_test']
            }
        }
        
        # Save JSON report
        import json
        report_path = os.path.join(self.output_dir, f"generation_report_{timestamp}.json")
        with open(report_path, 'w') as f:
            json.dump(report_data, f, indent=2, default=str)
        
        # Save text summary
        summary_path = os.path.join(self.output_dir, f"generation_summary_{timestamp}.txt")
        with open(summary_path, 'w') as f:
            f.write("NOISE IMAGE GENERATOR REPORT\n")
            f.write("=" * 50 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"User: vkhare2909\n")
            f.write(f"Target System: FIXED Noise Detection System\n\n")
            
            f.write("DATASET SUMMARY:\n")
            f.write("-" * 20 + "\n")
            f.write(f"Images per noise type: {images_per_type}\n")
            f.write(f"Total noisy images: {total_generated}\n")
            f.write(f"Clean reference images: 20\n")
            f.write(f"Mixed test cases: 20\n")
            f.write(f"Total images: {total_generated + 40}\n\n")
            
            f.write("NOISE TYPES AND CONFIGURATIONS:\n")
            f.write("-" * 35 + "\n")
            for noise_type, configs in self.noise_configs.items():
                f.write(f"\n{noise_type.upper()}:\n")
                for i, config in enumerate(configs):
                    f.write(f"  Config {i+1}: {config}\n")
            
            f.write(f"\nDIRECTORY STRUCTURE:\n")
            f.write("-" * 20 + "\n")
            f.write(f"{self.output_dir}/\n")
            for subdir in self.noise_types + ['clean_images', 'mixed_test']:
                f.write(f"├── {subdir}/\n")
        
        print(f"📄 Generation report saved: {report_path}")
    
    def generate_specific_noise_samples(self, noise_type, count=10, save_params=True):
        """Generate specific samples for testing individual noise types"""
        print(f"🎯 Generating {count} samples of {noise_type} noise")
        
        if noise_type not in self.noise_configs:
            print(f"❌ Unknown noise type: {noise_type}")
            return
        
        # Create specific directory
        specific_dir = os.path.join(self.output_dir, f"specific_{noise_type}")
        os.makedirs(specific_dir, exist_ok=True)
        
        configs = self.noise_configs[noise_type]
        
        for i in range(count):
            # Generate base image
            base_img = self.generate_base_image(i + 10000)
            
            # Select configuration
            config = configs[i % len(configs)]
            
            # Add noise based on type
            if noise_type == 'gaussian':
                noisy_img = self.add_gaussian_noise(base_img, config)
            elif noise_type == 'salt_pepper':
                noisy_img = self.add_salt_pepper_noise(base_img, config)
            elif noise_type == 'speckle':
                noisy_img = self.add_speckle_noise(base_img, config)
            elif noise_type == 'detector_striping':
                noisy_img = self.add_detector_striping(base_img, config)
            elif noise_type == 'detector_banding':
                noisy_img = self.add_detector_banding(base_img, config)
            
            # Save image
            filename = f"{noise_type}_specific_{i+1:03d}.png"
            filepath = os.path.join(specific_dir, filename)
            cv2.imwrite(filepath, noisy_img)
            
            # Save parameters if requested
            if save_params:
                param_file = os.path.join(specific_dir, f"{noise_type}_specific_{i+1:03d}_params.txt")
                with open(param_file, 'w') as f:
                    f.write(f"Noise Type: {noise_type}\n")
                    f.write(f"Parameters: {config}\n")
                    f.write(f"Base Image ID: {i + 10000}\n")
                    f.write(f"Generated: {datetime.now().isoformat()}\n")
        
        print(f"✅ Generated {count} {noise_type} samples in {specific_dir}")

def main():
    """Main function for noise image generator"""
    parser = argparse.ArgumentParser(
        description='Noise Image Generator for FIXED Noise Detection System'
    )
    parser.add_argument('--output', default='detector_test_images', 
                       help='Output directory name')
    parser.add_argument('--count', type=int, default=40,
                       help='Number of images per noise type')
    parser.add_argument('--size', nargs=2, type=int, default=[128, 128],
                       help='Image size (height width)')
    parser.add_argument('--specific', choices=['gaussian', 'salt_pepper', 'speckle', 
                                              'detector_striping', 'detector_banding'],
                       help='Generate samples for specific noise type only')
    parser.add_argument('--specific-count', type=int, default=10,
                       help='Number of samples for specific noise type')
    
    args = parser.parse_args()
    
    print("🎯 NOISE IMAGE GENERATOR FOR FIXED DETECTION SYSTEM")
    print("=" * 60)
    print(f"📁 Output Directory: {args.output}")
    print(f"📏 Image Size: {args.size[0]}x{args.size[1]}")
    print(f"👤 User: vkhare2909")
    print(f"📅 Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🔧 Target: FIXED Noise Detection System")
    print()
    
    generator = NoiseImageGeneratorForDetector(
        output_dir=args.output, 
        image_size=tuple(args.size)
    )
    
    if args.specific:
        # Generate specific noise type samples
        generator.generate_specific_noise_samples(
            args.specific, 
            args.specific_count, 
            save_params=True
        )
    else:
        # Generate full test dataset
        generator.generate_test_dataset(images_per_type=args.count)

if __name__ == "__main__":
    main()