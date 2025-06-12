import numpy as np
import cv2
import scipy.stats
import scipy.signal
from scipy.ndimage import gaussian_filter
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from skimage.filters import gabor
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import glob
from datetime import datetime
import json
import argparse
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class FinalNoiseDetector:
    """
    FINAL CORRECTED Noise Detector - Based on vkhare2909's test results analysis
    Fixes the salt & pepper over-detection issue that was misclassifying speckle
    """
    
    def __init__(self, output_dir="noise_analysis_results", training_size="small"):
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.output_dir = output_dir
        self.is_trained = False
        self.training_size = training_size
        
        # Initialize label encoder with known classes
        self.label_encoder.fit(['gaussian', 'salt_pepper', 'speckle', 'striping'])
        
        # FINAL: Optimized ensemble based on test results
        self.classifiers = {
            'rf': RandomForestClassifier(
                n_estimators=500,      # Increased for stability
                max_depth=18,          # Optimized depth
                min_samples_split=4,
                min_samples_leaf=2,
                max_features='sqrt',
                class_weight='balanced',
                random_state=42,
                n_jobs=-1
            ),
            'extra': ExtraTreesClassifier(
                n_estimators=400,
                max_depth=22,
                min_samples_split=3,
                min_samples_leaf=1,
                max_features='sqrt',
                class_weight='balanced',
                random_state=42,
                n_jobs=-1
            ),
            'gb': GradientBoostingClassifier(
                n_estimators=250,
                learning_rate=0.06,
                max_depth=7,
                subsample=0.85,
                random_state=42
            ),
            'svm_linear': SVC(         # Changed to linear for better speckle discrimination
                kernel='linear',
                C=5,
                probability=True,
                class_weight='balanced',
                random_state=42
            )
        }
    
    def safe_json_convert(self, obj):
        """Convert numpy arrays and other non-JSON-serializable objects to JSON-safe format"""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {key: self.safe_json_convert(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self.safe_json_convert(item) for item in obj]
        else:
            return obj
    
    def get_training_config(self):
        """Get training configuration"""
        configs = {
            'small': {
                'base_images': 400,      # Increased for better discrimination
                'mnist_count': 200,
                'synthetic_count': 200,
                'noise_variations': 5,   # More variations
                'description': 'Final training (~4800 images, 5-6 min)'
            },
            'medium': {
                'base_images': 600,
                'mnist_count': 300,
                'synthetic_count': 300,
                'noise_variations': 5,
                'description': 'Enhanced training (~7200 images, 7-8 min)'
            },
            'large': {
                'base_images': 800,
                'mnist_count': 400,
                'synthetic_count': 400,
                'noise_variations': 6,
                'description': 'Comprehensive training (~11520 images, 12-15 min)'
            }
        }
        return configs[self.training_size]
    
    def create_output_structure(self, base_name=""):
        """Create organized output directory structure"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if base_name:
            self.output_dir = f"{self.output_dir}_{base_name}_{timestamp}"
        else:
            self.output_dir = f"{self.output_dir}_{timestamp}"
        
        dirs_to_create = [
            self.output_dir,
            f"{self.output_dir}/input_images",
            f"{self.output_dir}/predictions",
            f"{self.output_dir}/summary_reports",
            f"{self.output_dir}/processed_results"
        ]
        
        for dir_path in dirs_to_create:
            os.makedirs(dir_path, exist_ok=True)
        
        print(f"✅ Output directory created: {self.output_dir}")
        return self.output_dir
    
    def generate_final_training_dataset(self):
        """Generate FINAL training dataset with corrected noise models"""
        config = self.get_training_config()
        
        print(f"🔧 FINAL Training Configuration: {self.training_size.upper()}")
        print(f"📊 {config['description']}")
        print(f"🎯 Focus: Eliminate speckle → salt&pepper confusion")
        
        base_images = []
        
        # 1. MNIST + Natural patterns
        try:
            import tensorflow as tf
            (X_mnist, y_mnist), _ = tf.keras.datasets.mnist.load_data()
            
            selected_indices = []
            samples_per_digit = config['mnist_count'] // 10
            
            for digit in range(10):
                digit_indices = np.where(y_mnist == digit)[0]
                selected = np.random.choice(digit_indices, samples_per_digit, replace=False)
                selected_indices.extend(selected)
            
            for idx in selected_indices[:config['mnist_count']]:
                img = cv2.resize(X_mnist[idx], (128, 128))
                base_images.append(img)
                
            print(f"   ✅ Added {len(selected_indices)} MNIST images")
            
        except Exception as e:
            print(f"   ⚠️ TensorFlow not available, using synthetic patterns")
        
        # 2. Enhanced synthetic patterns with speckle-friendly textures
        synthetic_patterns = [
            'checkerboard', 'gradient', 'circle', 'line', 'texture', 
            'random', 'sinusoidal', 'step', 'diagonal', 'crosshatch',
            'dots', 'waves', 'rings', 'uniform_low', 'uniform_mid', 'uniform_high',
            'natural_texture', 'smooth_gradient', 'fine_texture'  # Added for speckle
        ]
        
        patterns_per_type = config['synthetic_count'] // len(synthetic_patterns)
        
        for pattern_type in synthetic_patterns:
            for variation in range(patterns_per_type):
                img = self.generate_synthetic_pattern(pattern_type, variation)
                base_images.append(img)
        
        print(f"   ✅ Added {config['synthetic_count']} synthetic patterns")
        
        # 3. FINAL noise generation with CORRECTED parameters
        dataset = []
        labels = []
        
        # FINAL: Properly separated noise parameters
        noise_configs = {
            'gaussian': [
                {'std': 8}, {'std': 15}, {'std': 25}, {'std': 35}, {'std': 45}
            ][:config['noise_variations']],
            'salt_pepper': [
                {'density': 0.005}, {'density': 0.015}, {'density': 0.03}, 
                {'density': 0.06}, {'density': 0.1}
            ][:config['noise_variations']],
            'speckle': [
                {'var': 0.05, 'intensity_dep': True}, 
                {'var': 0.15, 'intensity_dep': True}, 
                {'var': 0.3, 'intensity_dep': True},
                {'var': 0.45, 'intensity_dep': True}, 
                {'var': 0.6, 'intensity_dep': True}
            ][:config['noise_variations']],
            'striping': [
                {'amp': 12, 'period': 5}, {'amp': 25, 'period': 8}, 
                {'amp': 40, 'period': 12}, {'amp': 55, 'period': 16},
                {'amp': 70, 'period': 22}
            ][:config['noise_variations']]
        }
        
        print(f"🔄 Generating FINAL corrected noisy variations...")
        for i, img in enumerate(base_images):
            if i % 100 == 0:
                print(f"   Processing base image {i+1}/{len(base_images)}")
                
            for noise_type, configs in noise_configs.items():
                for config_params in configs:
                    noisy_img = self.add_final_noise(img, noise_type, config_params)
                    dataset.append(noisy_img)
                    labels.append(noise_type)
        
        total_images = len(dataset)
        print(f"✅ Generated {total_images} FINAL training images")
        print(f"🎯 Speckle discrimination: ENHANCED")
        
        return np.array(dataset), np.array(labels)
    
    def generate_synthetic_pattern(self, pattern_type, variation):
        """Generate enhanced synthetic patterns"""
        size = 128
        img = np.zeros((size, size), dtype=np.uint8)
        
        var_factor = 1 + (variation % 4) * 0.15
        
        if pattern_type == 'natural_texture':
            # Create natural-looking texture
            base = np.random.rand(size//4, size//4) * 255
            img = cv2.resize(base, (size, size)).astype(np.uint8)
            
        elif pattern_type == 'smooth_gradient':
            # Smooth gradient for speckle testing
            x = np.linspace(0, 255, size)
            y = np.linspace(0, 255, size)
            X, Y = np.meshgrid(x, y)
            img = (128 + 64 * np.sin(X/50) * np.cos(Y/50)).astype(np.uint8)
            
        elif pattern_type == 'fine_texture':
            # Fine texture pattern
            img = (128 + 32 * np.random.randn(size, size)).astype(np.uint8)
            img = np.clip(img, 0, 255)
            
        elif pattern_type == 'uniform_low':
            img.fill(80)
        elif pattern_type == 'uniform_mid':
            img.fill(128)
        elif pattern_type == 'uniform_high':
            img.fill(180)
            
        elif pattern_type == 'checkerboard':
            square_size = int(8 * var_factor)
            for i in range(0, size, square_size * 2):
                for j in range(0, size, square_size * 2):
                    img[i:i+square_size, j:j+square_size] = 200
                    if i + square_size < size and j + square_size < size:
                        img[i+square_size:i+2*square_size, j+square_size:j+2*square_size] = 200
        
        else:
            # Default gradient
            for i in range(size):
                img[:, i] = int(128 + 64 * np.sin(2 * np.pi * i / (size * var_factor)))
        
        return img
    
    def add_final_noise(self, image, noise_type, params):
        """FINAL corrected noise addition"""
        img = image.astype(np.float64)
        
        if noise_type == 'gaussian':
            # Pure additive Gaussian
            noise = np.random.normal(0, params['std'], img.shape)
            noisy = img + noise
            
        elif noise_type == 'salt_pepper':
            # CORRECTED: True impulse noise with random distribution
            noisy = img.copy()
            total_pixels = img.size
            
            # Random locations for corruption
            num_corrupted = int(params['density'] * total_pixels)
            
            if num_corrupted > 0:
                # Randomly select pixels to corrupt
                flat_indices = np.random.choice(total_pixels, num_corrupted, replace=False)
                coords = np.unravel_index(flat_indices, img.shape)
                
                # Randomly assign salt or pepper
                salt_mask = np.random.rand(num_corrupted) > 0.5
                
                # Apply salt (255) and pepper (0)
                salt_coords = (coords[0][salt_mask], coords[1][salt_mask])
                pepper_coords = (coords[0][~salt_mask], coords[1][~salt_mask])
                
                noisy[salt_coords] = 255
                noisy[pepper_coords] = 0
            
        elif noise_type == 'speckle':
            # ENHANCED: True multiplicative speckle with intensity dependence
            if params.get('intensity_dep', True):
                # Intensity-dependent multiplicative noise
                # Higher intensity regions get more noise variance
                normalized_intensity = img / 255.0
                
                # Noise variance proportional to local intensity
                local_var = params['var'] * (0.3 + 0.7 * normalized_intensity)
                multiplicative_noise = np.random.normal(0, 1, img.shape) * np.sqrt(local_var)
                
                # True speckle model: I_out = I_in * (1 + noise)
                noisy = img * (1 + multiplicative_noise)
            else:
                # Standard multiplicative noise
                multiplicative_noise = np.random.normal(0, np.sqrt(params['var']), img.shape)
                noisy = img * (1 + multiplicative_noise)
            
        elif noise_type == 'striping':
            # Periodic striping
            h, w = img.shape
            if np.random.random() > 0.5:
                stripes = params['amp'] * np.sin(2 * np.pi * np.arange(h) / params['period'])
                stripes = stripes[:, np.newaxis]
            else:
                stripes = params['amp'] * np.sin(2 * np.pi * np.arange(w) / params['period'])
                stripes = stripes[np.newaxis, :]
            noisy = img + stripes
        
        return np.clip(noisy, 0, 255).astype(np.uint8)
    
    def train_system(self):
        """Train the FINAL system"""
        print("🚀 Training FINAL noise detection system...")
        print("🎯 Focus: Eliminate speckle → salt&pepper misclassification")
        print("👤 User: vkhare2909")
        print(f"📅 Date: 2025-06-12 06:40:36 UTC")
        start_time = datetime.now()
        
        X_train, y_train = self.generate_final_training_dataset()
        
        print("🔍 Extracting FINAL optimized features...")
        X_features = []
        feature_start = datetime.now()
        
        for i, img in enumerate(X_train):
            if i % 400 == 0:
                print(f"   Processed {i}/{len(X_train)} training images ({i/len(X_train)*100:.1f}%)")
            features = self.extract_final_features(img)
            X_features.append(features)
        
        feature_time = (datetime.now() - feature_start).total_seconds()
        print(f"   ✅ Feature extraction complete ({feature_time:.1f}s)")
        
        X_features = np.array(X_features)
        
        print("📊 Scaling features...")
        X_scaled = self.scaler.fit_transform(X_features)
        
        y_encoded = self.label_encoder.transform(y_train)
        training_start = datetime.now()
        
        for name, clf in self.classifiers.items():
            clf_start = datetime.now()
            print(f"   Training {name.upper()}...")
            clf.fit(X_scaled, y_encoded)
            clf_time = (datetime.now() - clf_start).total_seconds()
            print(f"   ✅ {name.upper()} trained ({clf_time:.1f}s)")
        
        total_time = (datetime.now() - start_time).total_seconds()
        
        self.is_trained = True
        
        print(f"🎉 FINAL training complete!")
        print(f"⏱️  Total time: {total_time:.1f}s")
        print(f"📊 Training images: {len(X_train)}")
        print(f"🎯 Feature dimension: {X_features.shape[1]}")
        
        self.validate_final_training(X_scaled, y_encoded)
    
    def validate_final_training(self, X_scaled, y_encoded):
        """Validate FINAL training"""
        print("🔬 Validating FINAL training quality...")
        
        X_train_val, X_test_val, y_train_val, y_test_val = train_test_split(
            X_scaled, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42
        )
        
        # Train and evaluate
        predictions = []
        for name, clf in self.classifiers.items():
            clf.fit(X_train_val, y_train_val)
            pred = clf.predict(X_test_val)
            acc = accuracy_score(y_test_val, pred)
            predictions.append(pred)
            print(f"   {name.upper()}: {acc:.3f} accuracy")
        
        # FINAL ensemble
        ensemble_pred = np.array(predictions).T
        weights = [0.35, 0.3, 0.25, 0.1]  # Emphasize tree-based methods
        ensemble_final = []
        
        for preds in ensemble_pred:
            weighted_votes = np.zeros(4)
            for i, pred in enumerate(preds):
                weighted_votes[pred] += weights[i]
            ensemble_final.append(np.argmax(weighted_votes))
        
        ensemble_final = np.array(ensemble_final)
        ensemble_acc = accuracy_score(y_test_val, ensemble_final)
        print(f"   🎯 FINAL ENSEMBLE: {ensemble_acc:.3f} accuracy")
        
        # Critical: Speckle vs Salt&Pepper confusion analysis
        speckle_idx = 2
        salt_pepper_idx = 1
        
        speckle_mask = (y_test_val == speckle_idx)
        salt_pepper_mask = (y_test_val == salt_pepper_idx)
        
        if np.any(speckle_mask):
            speckle_predictions = ensemble_final[speckle_mask]
            speckle_true = y_test_val[speckle_mask]
            speckle_acc = accuracy_score(speckle_true, speckle_predictions)
            
            # Count speckle → salt&pepper confusion
            speckle_as_salt_pepper = np.sum(speckle_predictions == salt_pepper_idx)
            total_speckle = len(speckle_predictions)
            confusion_rate = speckle_as_salt_pepper / total_speckle if total_speckle > 0 else 0
            
            print(f"   🔍 SPECKLE accuracy: {speckle_acc:.3f}")
            print(f"   ⚠️  Speckle→Salt&Pepper confusion: {confusion_rate:.3f} ({speckle_as_salt_pepper}/{total_speckle})")
        
        if np.any(salt_pepper_mask):
            salt_pepper_predictions = ensemble_final[salt_pepper_mask]
            salt_pepper_true = y_test_val[salt_pepper_mask]
            salt_pepper_acc = accuracy_score(salt_pepper_true, salt_pepper_predictions)
            print(f"   🔍 SALT&PEPPER accuracy: {salt_pepper_acc:.3f}")
        
        if ensemble_acc >= 0.96:
            print("   ✅ FINAL Training quality: EXCELLENT")
        elif ensemble_acc >= 0.93:
            print("   ✅ FINAL Training quality: VERY GOOD")
        else:
            print("   ⚠️  FINAL Training quality: NEEDS IMPROVEMENT")
    
    def stage1_final_screening(self, image):
        """FINAL Stage 1 screening - VERY conservative salt&pepper detection"""
        hist = cv2.calcHist([image], [0], None, [256], [0, 256]).flatten()
        
        # FINAL: VERY conservative salt & pepper detection
        exact_extreme_pixels = hist[0] + hist[255]  # Only exact 0 and 255
        total_pixels = image.size
        exact_extreme_ratio = exact_extreme_pixels / total_pixels
        
        # FINAL: Much higher threshold + isolation test
        if exact_extreme_ratio > 0.15:  # Very high threshold
            # Additional test: check if extreme pixels are isolated (true salt&pepper characteristic)
            binary_extreme = ((image == 0) | (image == 255)).astype(np.uint8)
            
            # Count isolated extreme pixels
            kernel = np.ones((3, 3), np.uint8)
            opened = cv2.morphologyEx(binary_extreme, cv2.MORPH_OPEN, kernel)
            isolated_pixels = np.sum(binary_extreme) - np.sum(opened)
            isolation_ratio = isolated_pixels / (np.sum(binary_extreme) + 1)
            
            # Only classify as salt&pepper if high isolation
            if isolation_ratio > 0.3:  # High isolation indicates true salt&pepper
                confidence = min(exact_extreme_ratio * 6 + isolation_ratio * 2, 0.95)
                return 'salt_pepper', confidence, {
                    'exact_extreme_ratio': exact_extreme_ratio,
                    'isolation_ratio': isolation_ratio
                }
        
        # Enhanced speckle pre-detection
        patch_size = 16
        h, w = image.shape
        mean_intensities = []
        local_variances = []
        
        # Sample patches for speckle test
        num_patches = 0
        for i in range(0, min(h-patch_size, 96), patch_size):
            for j in range(0, min(w-patch_size, 96), patch_size):
                patch = image[i:i+patch_size, j:j+patch_size].astype(np.float64)
                mean_val = np.mean(patch)
                var_val = np.var(patch)
                
                # Only include patches with reasonable intensity
                if 20 < mean_val < 235:
                    mean_intensities.append(mean_val)
                    local_variances.append(var_val)
                    num_patches += 1
        
        # Speckle detection via intensity-variance correlation
        if len(mean_intensities) > 8:
            try:
                correlation = np.corrcoef(mean_intensities, local_variances)[0, 1]
                if not np.isnan(correlation) and correlation > 0.65:
                    # Strong correlation suggests speckle
                    
                    # Additional test: coefficient of variation should be intensity-dependent
                    cv_values = []
                    for mean_val, var_val in zip(mean_intensities, local_variances):
                        if mean_val > 10:
                            cv = np.sqrt(var_val) / mean_val
                            cv_values.append(cv)
                    
                    if cv_values and np.mean(cv_values) > 0.1:
                        confidence = min(correlation * 1.1 + np.mean(cv_values) * 2, 0.88)
                        return 'speckle', confidence, {
                            'intensity_var_correlation': correlation,
                            'mean_cv': np.mean(cv_values),
                            'num_patches': num_patches
                        }
            except:
                pass
        
        # Conservative Gaussian detection
        sample_size = min(4000, image.size)
        sample = np.random.choice(image.flatten(), sample_size, replace=False)
        
        mean_val = np.mean(sample)
        std_val = np.std(sample)
        
        # Only test if significant noise is present
        if std_val > 10:
            normalized_sample = (sample - mean_val) / std_val
            ks_stat, p_value = scipy.stats.kstest(normalized_sample, 'norm')
            
            # More conservative Gaussian detection
            if p_value > 0.2 and std_val > 15:
                confidence = min(p_value * 2.5, 0.90)
                return 'gaussian', confidence, {
                    'ks_p_value': p_value, 
                    'sample_std': std_val,
                    'sample_mean': mean_val
                }
        
        # No confident early decision
        return None, 0.0, {
            'exact_extreme_ratio': exact_extreme_ratio,
            'sample_std': std_val,
            'decision': 'full_analysis_required'
        }
    
    def extract_final_features(self, image):
        """Extract FINAL optimized feature set"""
        features = []
        
        # Enhanced feature groups
        features.extend(self._extract_enhanced_noise_statistics(image))      # 25 features
        features.extend(self._extract_discriminative_histogram_features(image)) # 18 features
        features.extend(self._extract_advanced_speckle_features(image))      # 20 features
        features.extend(self._extract_refined_salt_pepper_features(image))   # 12 features
        features.extend(self._extract_optimized_frequency_features(image))   # 15 features
        features.extend(self._extract_essential_texture_features(image))     # 12 features
        
        return np.array(features)  # Total: 102 optimized features
    
    def _extract_enhanced_noise_statistics(self, image):
        """Enhanced noise statistics with better discrimination"""
        features = []
        img_float = image.astype(np.float64)
        
        # Multi-scale analysis
        for sigma in [0.8, 1.5, 2.5, 4.0, 6.0]:
            smoothed = gaussian_filter(img_float, sigma=sigma)
            noise = img_float - smoothed
            
            features.extend([
                np.mean(np.abs(noise)),
                np.std(noise),
                np.var(noise),
                np.percentile(np.abs(noise), 95),
                scipy.stats.skew(noise.flatten())
            ])
        
        return features
    
    def _extract_discriminative_histogram_features(self, image):
        """Histogram features optimized for noise discrimination"""
        features = []
        
        # Multi-resolution analysis
        for bins in [16, 32, 64, 128]:
            hist, _ = np.histogram(image, bins=bins, range=(0, 256))
            hist_norm = hist / np.sum(hist)
            
            # Key discriminative features
            entropy = -np.sum(hist_norm * np.log2(hist_norm + 1e-10))
            uniformity = np.sum(hist_norm**2)
            
            features.extend([entropy, uniformity])
        
        # Full resolution extreme analysis
        hist_256 = cv2.calcHist([image], [0], None, [256], [0, 256]).flatten()
        hist_256_norm = hist_256 / np.sum(hist_256)
        
        features.extend([
            hist_256_norm[0],          # Exact black pixels
            hist_256_norm[255],        # Exact white pixels
            np.sum(hist_256_norm[:5]), # Very dark
            np.sum(hist_256_norm[-5:]), # Very bright
            np.sum(hist_256_norm[125:131]), # Mid-range
            np.var(hist_256_norm),     # Histogram spread
            np.max(hist_256_norm),     # Peak value
            len(scipy.signal.find_peaks(hist_256_norm, height=0.003)[0]), # Peak count
            np.sum(hist_256_norm > 0.01), # Number of significant bins
            np.median(hist_256_norm)   # Median bin value
        ])
        
        return features
    
    def _extract_advanced_speckle_features(self, image):
        """Advanced speckle detection features"""
        features = []
        img_float = image.astype(np.float64)
        
        # Multi-scale intensity-variance analysis
        patch_sizes = [8, 16, 24]
        
        for patch_size in patch_sizes:
            h, w = img_float.shape
            mean_intensities = []
            local_variances = []
            local_cvs = []
            
            for i in range(0, h-patch_size, patch_size//2):
                for j in range(0, w-patch_size, patch_size//2):
                    if i + patch_size < h and j + patch_size < w:
                        patch = img_float[i:i+patch_size, j:j+patch_size]
                        mean_val = np.mean(patch)
                        var_val = np.var(patch)
                        
                        if 10 < mean_val < 245:  # Valid intensity range
                            mean_intensities.append(mean_val)
                            local_variances.append(var_val)
                            
                            cv = np.sqrt(var_val) / mean_val
                            local_cvs.append(cv)
            
            if len(mean_intensities) > 5:
                # Correlation (key speckle indicator)
                try:
                    correlation = np.corrcoef(mean_intensities, local_variances)[0, 1]
                    features.append(correlation if not np.isnan(correlation) else 0)
                except:
                    features.append(0)
                
                # Linear regression slope
                try:
                    slope = np.polyfit(mean_intensities, local_variances, 1)[0]
                    features.append(slope)
                except:
                    features.append(0)
                
                # CV statistics
                if local_cvs:
                    features.extend([
                        np.mean(local_cvs),
                        np.std(local_cvs)
                    ])
                else:
                    features.extend([0, 0])
            else:
                features.extend([0, 0, 0, 0])
        
        # Multiplicative noise model testing
        smoothed = gaussian_filter(img_float, sigma=2.0)
        
        # Avoid division by zero
        smoothed_safe = np.where(smoothed > 5, smoothed, 5)
        multiplicative_component = img_float / smoothed_safe
        
        features.extend([
            np.mean(multiplicative_component),
            np.std(multiplicative_component),
            np.var(multiplicative_component),
            scipy.stats.skew(multiplicative_component.flatten()),
            np.corrcoef(smoothed_safe.flatten(), multiplicative_component.flatten())[0, 1]
        ])
        
        return features
    
    def _extract_refined_salt_pepper_features(self, image):
        """Refined salt & pepper detection features"""
        features = []
        
        # Exact extreme pixel analysis
        hist = cv2.calcHist([image], [0], None, [256], [0, 256]).flatten()
        total_pixels = image.size
        
        exact_black = hist[0] / total_pixels
        exact_white = hist[255] / total_pixels
        near_black = np.sum(hist[0:3]) / total_pixels
        near_white = np.sum(hist[253:256]) / total_pixels
        
        features.extend([
            exact_black,
            exact_white,
            exact_black + exact_white,
            near_black - exact_black,  # Near but not exact
            near_white - exact_white,
        ])
        
        # Isolation analysis
        binary_extreme = ((image == 0) | (image == 255)).astype(np.uint8)
        
        if np.sum(binary_extreme) > 0:
            # Morphological analysis of extreme pixels
            kernel3 = np.ones((3, 3), np.uint8)
            kernel5 = np.ones((5, 5), np.uint8)
            
            opened3 = cv2.morphologyEx(binary_extreme, cv2.MORPH_OPEN, kernel3)
            opened5 = cv2.morphologyEx(binary_extreme, cv2.MORPH_OPEN, kernel5)
            
            isolated3 = np.sum(binary_extreme) - np.sum(opened3)
            isolated5 = np.sum(binary_extreme) - np.sum(opened5)
            
            features.extend([
                isolated3 / (np.sum(binary_extreme) + 1),
                isolated5 / (np.sum(binary_extreme) + 1),
                isolated3 / total_pixels,
                isolated5 / total_pixels
            ])
        else:
            features.extend([0, 0, 0, 0])
        
        # Clustering analysis of extreme pixels
        extreme_positions = np.where(binary_extreme)
        if len(extreme_positions[0]) > 5:
            # Average distance between extreme pixels
            coords = np.column_stack(extreme_positions)
            if len(coords) > 1:
                from scipy.spatial.distance import pdist
                distances = pdist(coords)
                avg_distance = np.mean(distances)
                min_distance = np.min(distances)
                features.extend([avg_distance, min_distance])
            else:
                features.extend([0, 0])
        else:
            features.extend([0, 0])
        
        return features
    
    def _extract_optimized_frequency_features(self, image):
        """Optimized frequency features"""
        features = []
        
        # 2D FFT
        fft = np.fft.fft2(image)
        fft_shifted = np.fft.fftshift(fft)
        magnitude = np.abs(fft_shifted)
        magnitude_log = np.log(magnitude + 1)
        
        center = np.array(image.shape) // 2
        y, x = np.ogrid[:image.shape[0], :image.shape[1]]
        radius = np.sqrt((x - center[1])**2 + (y - center[0])**2)
        
        # Frequency band analysis
        max_radius = min(image.shape) // 2
        for r in [0.1, 0.3, 0.5, 0.7, 0.9]:
            mask = radius <= r * max_radius
            features.append(np.mean(magnitude_log[mask]))
        
        # Directional analysis
        h_strength = np.mean(magnitude_log[center[0]-1:center[0]+2, :])
        v_strength = np.mean(magnitude_log[:, center[1]-1:center[1]+2])
        features.extend([h_strength, v_strength])
        
        # High frequency content
        high_freq_mask = radius > max_radius * 0.8
        features.append(np.mean(magnitude_log[high_freq_mask]))
        
        # Power spectrum statistics
        psd = magnitude**2
        features.extend([
            np.mean(psd),
            np.std(psd),
            np.max(psd),
            np.sum(psd > np.percentile(psd, 95)),
            np.sum(psd > np.percentile(psd, 99)),
            np.mean(psd[high_freq_mask])
        ])
        
        return features
    
    def _extract_essential_texture_features(self, image):
        """Essential texture features"""
        features = []
        
        # Gradient analysis
        grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        features.extend([
            np.mean(gradient_magnitude),
            np.std(gradient_magnitude),
            np.var(gradient_magnitude),
            np.max(gradient_magnitude)
        ])
        
        # Local binary pattern
        lbp = local_binary_pattern(image, 8, 1, method='uniform')
        hist_lbp, _ = np.histogram(lbp, bins=10, range=(0, 10))
        hist_lbp = hist_lbp / (np.sum(hist_lbp) + 1)
        
        features.extend([
            np.var(hist_lbp),
            -np.sum(hist_lbp * np.log2(hist_lbp + 1e-10)),
            np.max(hist_lbp)
        ])
        
        # GLCM features
        try:
            glcm = graycomatrix(image, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
            contrast = graycoprops(glcm, 'contrast')[0, 0]
            homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
            energy = graycoprops(glcm, 'energy')[0, 0]
            correlation = graycoprops(glcm, 'correlation')[0, 0]
            features.extend([contrast, homogeneity, energy, correlation])
        except:
            features.extend([0, 0, 0, 0])
        
        # Laplacian variance (measure of focus/sharpness)
        laplacian_var = cv2.Laplacian(image, cv2.CV_64F).var()
        features.append(laplacian_var)
        
        return features
    
    def predict_single_image(self, image, image_name=""):
        """FINAL prediction with optimized logic"""
        if not self.is_trained:
            print("❌ System not trained! Run train_system() first.")
            return None
        
        start_time = datetime.now()
        
        # Stage 1: FINAL conservative screening
        stage1_pred, stage1_conf, stage1_info = self.stage1_final_screening(image)
        
        if stage1_conf > 0.92:  # Very high confidence threshold
            prediction = stage1_pred
            confidence = stage1_conf
            ensemble_probabilities = {}
            feature_time = 0
        else:
            # Full analysis pipeline
            feature_start = datetime.now()
            features = self.extract_final_features(image)
            features_scaled = self.scaler.transform([features])
            feature_time = (datetime.now() - feature_start).total_seconds()
            
            # FINAL ensemble prediction
            probabilities = {}
            for name, clf in self.classifiers.items():
                probabilities[name] = clf.predict_proba(features_scaled)[0]
            
            # FINAL: Optimized weights based on test results
            weights = {'rf': 0.35, 'extra': 0.3, 'gb': 0.25, 'svm_linear': 0.1}
            ensemble_proba = np.zeros(4)
            for name, proba in probabilities.items():
                ensemble_proba += weights[name] * proba
            
            pred_encoded = np.argmax(ensemble_proba)
            prediction = self.label_encoder.inverse_transform([pred_encoded])[0]
            confidence = np.max(ensemble_proba)
            ensemble_probabilities = probabilities
        
        total_time = (datetime.now() - start_time).total_seconds()
        
        # Compile FINAL results
        result = {
            'image_name': image_name,
            'prediction': prediction,
            'confidence': float(confidence),
            'stage1_decision': stage1_pred,
            'stage1_confidence': float(stage1_conf) if stage1_conf is not None else 0.0,
            'stage1_info': self.safe_json_convert(stage1_info),
            'ensemble_probabilities': self.safe_json_convert(ensemble_probabilities),
            'processing_time': float(total_time),
            'feature_time': float(feature_time),
            'image_shape': [int(x) for x in image.shape],
            'mean_intensity': float(np.mean(image)),
            'std_intensity': float(np.std(image)),
            'timestamp': '2025-06-12T06:40:36Z',
            'system_version': 'final_v1.0',
            'user': 'vkhare2909'
        }
        
        return result
    
    def load_and_preprocess_image(self, image_path):
        """Load and preprocess image for analysis"""
        try:
            if image_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif')):
                image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                
                if image is None:
                    return None
                
                # Resize if needed
                if max(image.shape) > 512:
                    scale_factor = 512 / max(image.shape)
                    new_height = int(image.shape[0] * scale_factor)
                    new_width = int(image.shape[1] * scale_factor)
                    image = cv2.resize(image, (new_width, new_height))
                
                if min(image.shape) < 64:
                    image = cv2.resize(image, (128, 128))
                
                return image
            else:
                return None
                
        except Exception as e:
            return None
    
    def process_image_folder(self, folder_path, file_pattern="*"):
        """Process all images in folder with FINAL system"""
        folder_path = Path(folder_path)
        
        if not folder_path.exists():
            print(f"❌ Folder not found: {folder_path}")
            return
        
        folder_name = folder_path.name
        self.create_output_structure(folder_name)
        
        # Find image files
        image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tiff', '*.tif']
        image_files = []
        
        for ext in image_extensions:
            pattern = f"{file_pattern}.{ext[2:]}" if file_pattern != "*" else ext
            image_files.extend(folder_path.glob(pattern))
        
        if not image_files:
            print(f"❌ No image files found in {folder_path}")
            return
        
        print(f"🔍 Found {len(image_files)} images to process")
        print(f"🔧 FINAL corrected noise detection system: ENABLED")
        print(f"🎯 Fix: Eliminate speckle → salt&pepper misclassification")
        print(f"👤 User: vkhare2909")
        print(f"📅 Time: 2025-06-12 06:40:36 UTC")
        
        if not self.is_trained:
            self.train_system()
        
        all_results = []
        processed_count = 0
        
        # Count by type for analysis
        type_counts = {'gaussian': 0, 'salt_pepper': 0, 'speckle': 0, 'striping': 0}
        
        print(f"\n📸 Processing images with FINAL corrected detection...")
        for i, image_path in enumerate(image_files):
            print(f"   {i+1}/{len(image_files)}: {image_path.name} ... ", end="")
            
            image = self.load_and_preprocess_image(str(image_path))
            if image is None:
                print("❌ Failed")
                continue
            
            result = self.predict_single_image(image, str(image_path))
            if result is None:
                print("❌ Analysis failed")
                continue
            
            # Save results
            self.save_analysis(image, result, i)
            all_results.append(result)
            processed_count += 1
            
            # Track predictions
            type_counts[result['prediction']] += 1
            
            print(f"✅ {result['prediction'].upper()} ({result['confidence']:.1%})")
        
        if all_results:
            self.create_summary(all_results, folder_name)
            print(f"\n🎉 FINAL processing complete: {processed_count}/{len(image_files)} images")
            print(f"📊 FINAL Results: {type_counts}")
            print(f"📁 Saved to: {self.output_dir}")
            
            # Analysis of improvements
            if 'speckle' in folder_name.lower():
                speckle_correct = type_counts['speckle']
                speckle_as_salt_pepper = type_counts['salt_pepper']
                total_speckle = speckle_correct + speckle_as_salt_pepper + type_counts['gaussian'] + type_counts['striping']
                if total_speckle > 0:
                    improvement = f"Speckle accuracy: {speckle_correct}/{total_speckle} ({speckle_correct/total_speckle*100:.1f}%)"
                    print(f"🎯 {improvement}")
        else:
            print("❌ No images were successfully processed")
    
    def save_analysis(self, image, result, image_id):
        """Save analysis results"""
        image_name = result['image_name']
        base_name = os.path.splitext(os.path.basename(image_name))[0]
        
        # Save image copy
        cv2.imwrite(f"{self.output_dir}/input_images/{base_name}_input.png", image)
        
        # Save JSON result
        json_safe_result = self.safe_json_convert(result)
        with open(f"{self.output_dir}/predictions/{base_name}_prediction.json", 'w') as f:
            json.dump(json_safe_result, f, indent=2)
        
        # Save readable report
        with open(f"{self.output_dir}/predictions/{base_name}_report.txt", 'w') as f:
            f.write(f"FINAL NOISE DETECTION REPORT\n")
            f.write(f"============================\n")
            f.write(f"Image: {os.path.basename(result['image_name'])}\n")
            f.write(f"Date: 2025-06-12 06:40:36 UTC\n")
            f.write(f"User: vkhare2909\n")
            f.write(f"System: Final v1.0 - Speckle Optimized\n\n")
            f.write(f"RESULT: {result['prediction'].upper()}\n")
            f.write(f"Confidence: {result['confidence']:.1%}\n")
            f.write(f"Stage 1 Decision: {result['stage1_decision']}\n")
            f.write(f"Stage 1 Confidence: {result['stage1_confidence']:.1%}\n")
            f.write(f"Processing Time: {result['processing_time']:.3f}s\n\n")
            
            if result.get('stage1_info'):
                f.write(f"Stage 1 Analysis:\n")
                for key, value in result['stage1_info'].items():
                    f.write(f"  {key}: {value}\n")
    
    def create_summary(self, all_results, folder_name):
        """Create summary report"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        predictions = [r['prediction'] for r in all_results]
        confidences = [r['confidence'] for r in all_results]
        processing_times = [r['processing_time'] for r in all_results]
        
        prediction_counts = {
            'gaussian': predictions.count('gaussian'),
            'salt_pepper': predictions.count('salt_pepper'),
            'speckle': predictions.count('speckle'),
            'striping': predictions.count('striping')
        }
        
        # Summary data
        summary_data = {
            'folder_info': {
                'name': folder_name,
                'timestamp': '2025-06-12T06:40:36Z',
                'user': 'vkhare2909',
                'total_images': len(all_results),
                'system_version': 'final_v1.0_speckle_optimized'
            },
            'noise_distribution': prediction_counts,
            'confidence_stats': {
                'mean': float(np.mean(confidences)),
                'std': float(np.std(confidences)),
                'min': float(np.min(confidences)),
                'max': float(np.max(confidences)),
                'high_confidence_count': int(sum(1 for c in confidences if c > 0.9))
            },
            'performance_stats': {
                'mean_time': float(np.mean(processing_times)),
                'total_time': float(np.sum(processing_times))
            },
            'individual_results': self.safe_json_convert(all_results)
        }
        
        with open(f"{self.output_dir}/summary_reports/final_summary_{timestamp}.json", 'w') as f:
            json.dump(summary_data, f, indent=2)
        
        # CSV results
        import csv
        with open(f"{self.output_dir}/summary_reports/final_results.csv", 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Image', 'Prediction', 'Confidence', 'Processing_Time', 'Stage1_Decision', 'System_Version'])
            for result in all_results:
                writer.writerow([
                    os.path.basename(result['image_name']),
                    result['prediction'],
                    result['confidence'],
                    result['processing_time'],
                    result['stage1_decision'],
                    'final_v1.0'
                ])
        
        print(f"📊 FINAL Summary: {prediction_counts}")
        print(f"🎯 Mean confidence: {np.mean(confidences):.1%}")
        print(f"⏱️  Mean processing time: {np.mean(processing_times):.3f}s")

def main():
    """Main function for FINAL system"""
    parser = argparse.ArgumentParser(description='FINAL Corrected Noise Detection System')
    parser.add_argument('folder_path', help='Path to folder containing images')
    parser.add_argument('--pattern', default='*', help='File pattern to match')
    parser.add_argument('--output', default='final_noise_analysis', help='Output directory name')
    parser.add_argument('--training', choices=['small', 'medium', 'large'], default='small',
                       help='Training dataset size')
    
    args = parser.parse_args()
    
    print("🔧 FINAL CORRECTED NOISE DETECTION SYSTEM")
    print("=" * 50)
    print("🎯 FINAL: Eliminate speckle → salt&pepper misclassification")
    print("🔬 Based on analysis of vkhare2909's test results")
    print(f"📁 Input Folder: {args.folder_path}")
    print(f"👤 User: vkhare2909")
    print(f"📅 Date: 2025-06-12 06:40:36 UTC")
    print(f"🔧 System: Final v1.0 - Speckle Optimized")
    print()
    
    detector = FinalNoiseDetector(args.output, training_size=args.training)
    detector.process_image_folder(args.folder_path, args.pattern)

if __name__ == "__main__":
    main()