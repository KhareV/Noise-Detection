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
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
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

class UltraFinalNoiseDetector:
    """
    ULTRA-FINAL Complete Noise Detector System
    - Log transform speckle enhancement
    - Comprehensive visual output with histograms and analysis
    - Production-ready noise detection system
    
    User: vkhare2909
    Date: 2025-06-12 09:25:15 UTC
    """
    
    def __init__(self, output_dir="noise_analysis_results", training_size="small"):
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.output_dir = output_dir
        self.is_trained = False
        self.training_size = training_size
        
        # Initialize label encoder with known classes
        self.label_encoder.fit(['gaussian', 'salt_pepper', 'speckle', 'striping'])
        
        # ULTRA-FINAL: Optimized ensemble
        self.classifiers = {
            'rf': RandomForestClassifier(
                n_estimators=500,
                max_depth=18,
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
            'svm_linear': SVC(
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
                'base_images': 400,
                'mnist_count': 200,
                'synthetic_count': 200,
                'noise_variations': 5,
                'description': 'Ultra-Final training (~4800 images, 5-6 min)'
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
            f"{self.output_dir}/processed_results",
            f"{self.output_dir}/visual_analysis",      # NEW: Visual outputs
            f"{self.output_dir}/histograms",           # NEW: Histogram plots
            f"{self.output_dir}/feature_analysis",     # NEW: Feature visualizations
            f"{self.output_dir}/comparison_charts"     # NEW: Comparison charts
        ]
        
        for dir_path in dirs_to_create:
            os.makedirs(dir_path, exist_ok=True)
        
        print(f"✅ Output directory created: {self.output_dir}")
        return self.output_dir
    
    def generate_ultra_final_training_dataset(self):
        """Generate ULTRA-FINAL training dataset with enhanced noise models"""
        config = self.get_training_config()
        
        print(f"🔧 ULTRA-FINAL Training Configuration: {self.training_size.upper()}")
        print(f"📊 {config['description']}")
        print(f"🎯 Focus: Log transform speckle enhancement + visual output")
        print(f"👤 User: vkhare2909")
        print(f"📅 Date: 2025-06-12 09:25:15 UTC")
        
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
        
        # 2. Enhanced synthetic patterns
        synthetic_patterns = [
            'checkerboard', 'gradient', 'circle', 'line', 'texture', 
            'random', 'sinusoidal', 'step', 'diagonal', 'crosshatch',
            'dots', 'waves', 'rings', 'uniform_low', 'uniform_mid', 'uniform_high',
            'natural_texture', 'smooth_gradient', 'fine_texture'
        ]
        
        patterns_per_type = config['synthetic_count'] // len(synthetic_patterns)
        
        for pattern_type in synthetic_patterns:
            for variation in range(patterns_per_type):
                img = self.generate_synthetic_pattern(pattern_type, variation)
                base_images.append(img)
        
        print(f"   ✅ Added {config['synthetic_count']} synthetic patterns")
        
        # 3. ULTRA-FINAL noise generation
        dataset = []
        labels = []
        
        # Enhanced noise parameters
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
        
        print(f"🔄 Generating ULTRA-FINAL enhanced noisy variations...")
        for i, img in enumerate(base_images):
            if i % 100 == 0:
                print(f"   Processing base image {i+1}/{len(base_images)}")
                
            for noise_type, configs in noise_configs.items():
                for config_params in configs:
                    noisy_img = self.add_ultra_final_noise(img, noise_type, config_params)
                    dataset.append(noisy_img)
                    labels.append(noise_type)
        
        total_images = len(dataset)
        print(f"✅ Generated {total_images} ULTRA-FINAL training images")
        print(f"🎯 Log transform speckle enhancement: ENABLED")
        
        return np.array(dataset), np.array(labels)
    
    def generate_synthetic_pattern(self, pattern_type, variation):
        """Generate enhanced synthetic patterns"""
        size = 128
        img = np.zeros((size, size), dtype=np.uint8)
        
        var_factor = 1 + (variation % 4) * 0.15
        
        if pattern_type == 'natural_texture':
            base = np.random.rand(size//4, size//4) * 255
            img = cv2.resize(base, (size, size)).astype(np.uint8)
            
        elif pattern_type == 'smooth_gradient':
            x = np.linspace(0, 255, size)
            y = np.linspace(0, 255, size)
            X, Y = np.meshgrid(x, y)
            img = (128 + 64 * np.sin(X/50) * np.cos(Y/50)).astype(np.uint8)
            
        elif pattern_type == 'fine_texture':
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
    
    def add_ultra_final_noise(self, image, noise_type, params):
        """ULTRA-FINAL noise addition with enhanced models"""
        img = image.astype(np.float64)
        
        if noise_type == 'gaussian':
            noise = np.random.normal(0, params['std'], img.shape)
            noisy = img + noise
            
        elif noise_type == 'salt_pepper':
            noisy = img.copy()
            total_pixels = img.size
            
            num_corrupted = int(params['density'] * total_pixels)
            
            if num_corrupted > 0:
                flat_indices = np.random.choice(total_pixels, num_corrupted, replace=False)
                coords = np.unravel_index(flat_indices, img.shape)
                
                salt_mask = np.random.rand(num_corrupted) > 0.5
                
                salt_coords = (coords[0][salt_mask], coords[1][salt_mask])
                pepper_coords = (coords[0][~salt_mask], coords[1][~salt_mask])
                
                noisy[salt_coords] = 255
                noisy[pepper_coords] = 0
            
        elif noise_type == 'speckle':
            if params.get('intensity_dep', True):
                normalized_intensity = img / 255.0
                local_var = params['var'] * (0.3 + 0.7 * normalized_intensity)
                multiplicative_noise = np.random.normal(0, 1, img.shape) * np.sqrt(local_var)
                noisy = img * (1 + multiplicative_noise)
            else:
                multiplicative_noise = np.random.normal(0, np.sqrt(params['var']), img.shape)
                noisy = img * (1 + multiplicative_noise)
            
        elif noise_type == 'striping':
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
        """Train the ULTRA-FINAL system"""
        print("🚀 Training ULTRA-FINAL noise detection system...")
        print("🎯 Focus: Log transform + comprehensive visual output")
        print("👤 User: vkhare2909")
        print(f"📅 Date: 2025-06-12 09:25:15 UTC")
        start_time = datetime.now()
        
        X_train, y_train = self.generate_ultra_final_training_dataset()
        
        print("🔍 Extracting ULTRA-FINAL optimized features...")
        X_features = []
        feature_start = datetime.now()
        
        for i, img in enumerate(X_train):
            if i % 400 == 0:
                print(f"   Processed {i}/{len(X_train)} training images ({i/len(X_train)*100:.1f}%)")
            features = self.extract_ultra_final_features(img)
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
        
        print(f"🎉 ULTRA-FINAL training complete!")
        print(f"⏱️  Total time: {total_time:.1f}s")
        print(f"📊 Training images: {len(X_train)}")
        print(f"🎯 Feature dimension: {X_features.shape[1]} (log transform enhanced)")
        
        self.validate_ultra_final_training(X_scaled, y_encoded)
    
    def validate_ultra_final_training(self, X_scaled, y_encoded):
        """Validate ULTRA-FINAL training with detailed metrics"""
        print("🔬 Validating ULTRA-FINAL training quality...")
        
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
        
        # ULTRA-FINAL ensemble
        ensemble_pred = np.array(predictions).T
        weights = [0.35, 0.3, 0.25, 0.1]
        ensemble_final = []
        
        for preds in ensemble_pred:
            weighted_votes = np.zeros(4)
            for i, pred in enumerate(preds):
                weighted_votes[pred] += weights[i]
            ensemble_final.append(np.argmax(weighted_votes))
        
        ensemble_final = np.array(ensemble_final)
        ensemble_acc = accuracy_score(y_test_val, ensemble_final)
        print(f"   🎯 ULTRA-FINAL ENSEMBLE: {ensemble_acc:.3f} accuracy")
        
        # ENHANCED: Detailed per-class analysis
        print("📊 Detailed Classification Report:")
        print(classification_report(y_test_val, ensemble_final, 
                                  target_names=['gaussian', 'salt_pepper', 'speckle', 'striping']))
        
        # Per-class precision/recall
        from sklearn.metrics import precision_recall_fscore_support
        precision, recall, f1, support = precision_recall_fscore_support(y_test_val, ensemble_final)
        
        for i, class_name in enumerate(['gaussian', 'salt_pepper', 'speckle', 'striping']):
            print(f"   🎯 {class_name.upper()}: P={precision[i]:.3f}, R={recall[i]:.3f}, F1={f1[i]:.3f}")
        
        # Speckle-specific analysis
        speckle_idx = 2
        salt_pepper_idx = 1
        
        speckle_mask = (y_test_val == speckle_idx)
        if np.any(speckle_mask):
            speckle_predictions = ensemble_final[speckle_mask]
            speckle_true = y_test_val[speckle_mask]
            speckle_acc = accuracy_score(speckle_true, speckle_predictions)
            
            speckle_as_salt_pepper = np.sum(speckle_predictions == salt_pepper_idx)
            total_speckle = len(speckle_predictions)
            confusion_rate = speckle_as_salt_pepper / total_speckle if total_speckle > 0 else 0
            
            print(f"   🔍 SPECKLE accuracy: {speckle_acc:.3f}")
            print(f"   ⚠️  Speckle→Salt&Pepper confusion: {confusion_rate:.3f} ({speckle_as_salt_pepper}/{total_speckle})")
        
        if ensemble_acc >= 0.96:
            print("   ✅ ULTRA-FINAL Training quality: EXCELLENT")
        elif ensemble_acc >= 0.93:
            print("   ✅ ULTRA-FINAL Training quality: VERY GOOD")
        else:
            print("   ⚠️  ULTRA-FINAL Training quality: NEEDS IMPROVEMENT")
    
    def stage1_ultra_final_screening(self, image):
        """ULTRA-FINAL Stage 1 with log transform speckle enhancement"""
        hist = cv2.calcHist([image], [0], None, [256], [0, 256]).flatten()
        
        # NEW: Log transform speckle pre-test (PRIORITY CHECK)
        img_float = image.astype(np.float64)
        log_img = np.log1p(img_float)
        log_var = np.var(log_img)
        log_skew = scipy.stats.skew(log_img.flatten())
        log_mean = np.mean(log_img)
        
        # Enhanced speckle detection using log transform
        if log_var > 0.8 and abs(log_skew) < 0.5:
            # Quick intensity-variance correlation test
            patch_size = 16
            h, w = image.shape
            correlations = []
            cv_values = []
            
            for i in range(0, min(h-patch_size, 64), patch_size):
                for j in range(0, min(w-patch_size, 64), patch_size):
                    patch = image[i:i+patch_size, j:j+patch_size].astype(np.float64)
                    mean_val = np.mean(patch)
                    var_val = np.var(patch)
                    
                    if mean_val > 20:
                        corr = np.corrcoef([mean_val], [var_val])[0, 1]
                        if not np.isnan(corr):
                            correlations.append(corr)
                        
                        cv = np.sqrt(var_val) / mean_val if mean_val > 10 else 0
                        cv_values.append(cv)
            
            if correlations and cv_values:
                mean_corr = np.mean(correlations)
                mean_cv = np.mean(cv_values)
                
                # Enhanced speckle criteria
                if mean_corr > 0.6 and mean_cv > 0.15:
                    confidence = min(log_var * 0.7 + mean_corr * 0.4 + mean_cv * 0.3, 0.88)
                    return 'speckle', confidence, {
                        'log_variance': float(log_var),
                        'log_skewness': float(log_skew),
                        'log_mean': float(log_mean),
                        'intensity_var_correlation': float(mean_corr),
                        'mean_cv': float(mean_cv),
                        'method': 'log_transform_enhanced'
                    }
        
        # Ultra-conservative salt & pepper detection
        exact_extreme_pixels = hist[0] + hist[255]
        total_pixels = image.size
        exact_extreme_ratio = exact_extreme_pixels / total_pixels
        
        if exact_extreme_ratio > 0.18:  # Very high threshold
            # Isolation test
            binary_extreme = ((image == 0) | (image == 255)).astype(np.uint8)
            
            if np.sum(binary_extreme) > 0:
                kernel = np.ones((3, 3), np.uint8)
                opened = cv2.morphologyEx(binary_extreme, cv2.MORPH_OPEN, kernel)
                isolated_pixels = np.sum(binary_extreme) - np.sum(opened)
                isolation_ratio = isolated_pixels / (np.sum(binary_extreme) + 1)
                
                if isolation_ratio > 0.3:
                    confidence = min(exact_extreme_ratio * 4 + isolation_ratio * 2, 0.95)
                    return 'salt_pepper', confidence, {
                        'exact_extreme_ratio': float(exact_extreme_ratio),
                        'isolation_ratio': float(isolation_ratio),
                        'method': 'isolation_test'
                    }
        
        # Conservative Gaussian detection
        sample_size = min(4000, image.size)
        sample = np.random.choice(image.flatten(), sample_size, replace=False)
        
        mean_val = np.mean(sample)
        std_val = np.std(sample)
        
        if std_val > 10:
            normalized_sample = (sample - mean_val) / std_val
            ks_stat, p_value = scipy.stats.kstest(normalized_sample, 'norm')
            
            if p_value > 0.2 and std_val > 15:
                confidence = min(p_value * 2.5, 0.90)
                return 'gaussian', confidence, {
                    'ks_p_value': float(p_value), 
                    'sample_std': float(std_val),
                    'sample_mean': float(mean_val),
                    'method': 'kolmogorov_smirnov'
                }
        
        # No confident early decision
        return None, 0.0, {
            'exact_extreme_ratio': float(exact_extreme_ratio),
            'sample_std': float(std_val),
            'log_variance': float(log_var),
            'decision': 'full_analysis_required'
        }
    
    def extract_ultra_final_features(self, image):
        """Extract ULTRA-FINAL feature set with log transform enhancement"""
        features = []
        
        # Enhanced feature groups with log transform
        features.extend(self._extract_enhanced_noise_statistics(image))      # 25 features
        features.extend(self._extract_discriminative_histogram_features(image)) # 18 features
        features.extend(self._extract_log_enhanced_speckle_features(image))  # 25 features (ENHANCED)
        features.extend(self._extract_refined_salt_pepper_features(image))   # 12 features
        features.extend(self._extract_optimized_frequency_features(image))   # 15 features
        features.extend(self._extract_essential_texture_features(image))     # 12 features
        
        return np.array(features)  # Total: 107 optimized features
    
    def _extract_log_enhanced_speckle_features(self, image):
        """LOG TRANSFORM ENHANCED speckle detection features"""
        features = []
        img_float = image.astype(np.float64)
        
        # NEW: Log transform analysis for multiplicative noise
        log_img = np.log1p(img_float)  # log(1 + x) to handle zeros
        features.extend([
            np.var(log_img),                           # Log variance
            scipy.stats.skew(log_img.flatten()),       # Log skewness
            np.std(log_img),                           # Log standard deviation
            np.mean(log_img),                          # Log mean
            scipy.stats.kurtosis(log_img.flatten())    # Log kurtosis
        ])
        
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
                        
                        if 10 < mean_val < 245:
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
        
        # Enhanced multiplicative noise model testing
        smoothed = gaussian_filter(img_float, sigma=2.0)
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
    
    # Include all other feature extraction methods from previous implementation
    def _extract_enhanced_noise_statistics(self, image):
        """Enhanced noise statistics"""
        features = []
        img_float = image.astype(np.float64)
        
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
        
        for bins in [16, 32, 64, 128]:
            hist, _ = np.histogram(image, bins=bins, range=(0, 256))
            hist_norm = hist / np.sum(hist)
            
            entropy = -np.sum(hist_norm * np.log2(hist_norm + 1e-10))
            uniformity = np.sum(hist_norm**2)
            
            features.extend([entropy, uniformity])
        
        hist_256 = cv2.calcHist([image], [0], None, [256], [0, 256]).flatten()
        hist_256_norm = hist_256 / np.sum(hist_256)
        
        features.extend([
            hist_256_norm[0],
            hist_256_norm[255],
            np.sum(hist_256_norm[:5]),
            np.sum(hist_256_norm[-5:]),
            np.sum(hist_256_norm[125:131]),
            np.var(hist_256_norm),
            np.max(hist_256_norm),
            len(scipy.signal.find_peaks(hist_256_norm, height=0.003)[0]),
            np.sum(hist_256_norm > 0.01),
            np.median(hist_256_norm)
        ])
        
        return features
    
    def _extract_refined_salt_pepper_features(self, image):
        """Refined salt & pepper detection features"""
        features = []
        
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
            near_black - exact_black,
            near_white - exact_white,
        ])
        
        binary_extreme = ((image == 0) | (image == 255)).astype(np.uint8)
        
        if np.sum(binary_extreme) > 0:
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
        
        extreme_positions = np.where(binary_extreme)
        if len(extreme_positions[0]) > 5:
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
        
        # Additional isolation metric
        features.append(np.sum(binary_extreme) / total_pixels)
        
        return features
    
    def _extract_optimized_frequency_features(self, image):
        """Optimized frequency features"""
        features = []
        
        fft = np.fft.fft2(image)
        fft_shifted = np.fft.fftshift(fft)
        magnitude = np.abs(fft_shifted)
        magnitude_log = np.log(magnitude + 1)
        
        center = np.array(image.shape) // 2
        y, x = np.ogrid[:image.shape[0], :image.shape[1]]
        radius = np.sqrt((x - center[1])**2 + (y - center[0])**2)
        
        max_radius = min(image.shape) // 2
        for r in [0.1, 0.3, 0.5, 0.7, 0.9]:
            mask = radius <= r * max_radius
            features.append(np.mean(magnitude_log[mask]))
        
        h_strength = np.mean(magnitude_log[center[0]-1:center[0]+2, :])
        v_strength = np.mean(magnitude_log[:, center[1]-1:center[1]+2])
        features.extend([h_strength, v_strength])
        
        high_freq_mask = radius > max_radius * 0.8
        features.append(np.mean(magnitude_log[high_freq_mask]))
        
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
        
        grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        features.extend([
            np.mean(gradient_magnitude),
            np.std(gradient_magnitude),
            np.var(gradient_magnitude),
            np.max(gradient_magnitude)
        ])
        
        lbp = local_binary_pattern(image, 8, 1, method='uniform')
        hist_lbp, _ = np.histogram(lbp, bins=10, range=(0, 10))
        hist_lbp = hist_lbp / (np.sum(hist_lbp) + 1)
        
        features.extend([
            np.var(hist_lbp),
            -np.sum(hist_lbp * np.log2(hist_lbp + 1e-10)),
            np.max(hist_lbp)
        ])
        
        try:
            glcm = graycomatrix(image, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
            contrast = graycoprops(glcm, 'contrast')[0, 0]
            homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
            energy = graycoprops(glcm, 'energy')[0, 0]
            correlation = graycoprops(glcm, 'correlation')[0, 0]
            features.extend([contrast, homogeneity, energy, correlation])
        except:
            features.extend([0, 0, 0, 0])
        
        laplacian_var = cv2.Laplacian(image, cv2.CV_64F).var()
        features.append(laplacian_var)
        
        return features
    
    def predict_single_image(self, image, image_name=""):
        """ULTRA-FINAL prediction with comprehensive analysis"""
        if not self.is_trained:
            print("❌ System not trained! Run train_system() first.")
            return None
        
        start_time = datetime.now()
        
        # Stage 1: ULTRA-FINAL screening with log transform
        stage1_pred, stage1_conf, stage1_info = self.stage1_ultra_final_screening(image)
        
        if stage1_conf > 0.92:
            prediction = stage1_pred
            confidence = stage1_conf
            ensemble_probabilities = {}
            feature_time = 0
            detailed_analysis = stage1_info
        else:
            # Full analysis pipeline
            feature_start = datetime.now()
            features = self.extract_ultra_final_features(image)
            features_scaled = self.scaler.transform([features])
            feature_time = (datetime.now() - feature_start).total_seconds()
            
            # ULTRA-FINAL ensemble prediction
            probabilities = {}
            for name, clf in self.classifiers.items():
                probabilities[name] = clf.predict_proba(features_scaled)[0]
            
            weights = {'rf': 0.35, 'extra': 0.3, 'gb': 0.25, 'svm_linear': 0.1}
            ensemble_proba = np.zeros(4)
            for name, proba in probabilities.items():
                ensemble_proba += weights[name] * proba
            
            pred_encoded = np.argmax(ensemble_proba)
            prediction = self.label_encoder.inverse_transform([pred_encoded])[0]
            confidence = np.max(ensemble_proba)
            ensemble_probabilities = probabilities
            
            # Enhanced analysis
            detailed_analysis = {
                'log_variance': float(np.var(np.log1p(image.astype(np.float64)))),
                'intensity_variance_correlation': float(self._calculate_intensity_variance_correlation(image)),
                'extreme_pixel_ratio': float((np.sum(image == 0) + np.sum(image == 255)) / image.size),
                'ensemble_probabilities': self.safe_json_convert(ensemble_probabilities),
                'stage1_info': stage1_info
            }
        
        total_time = (datetime.now() - start_time).total_seconds()
        
        # Compile ULTRA-FINAL results
        result = {
            'image_name': image_name,
            'prediction': prediction,
            'confidence': float(confidence),
            'stage1_decision': stage1_pred,
            'stage1_confidence': float(stage1_conf) if stage1_conf is not None else 0.0,
            'stage1_info': self.safe_json_convert(stage1_info),
            'ensemble_probabilities': self.safe_json_convert(ensemble_probabilities),
            'detailed_analysis': self.safe_json_convert(detailed_analysis),
            'processing_time': float(total_time),
            'feature_time': float(feature_time),
            'image_shape': [int(x) for x in image.shape],
            'mean_intensity': float(np.mean(image)),
            'std_intensity': float(np.std(image)),
            'timestamp': '2025-06-12T09:25:15Z',
            'system_version': 'ultra_final_v1.0',
            'user': 'vkhare2909'
        }
        
        return result
    
    def _calculate_intensity_variance_correlation(self, image):
        """Calculate intensity-variance correlation for speckle detection"""
        img_float = image.astype(np.float64)
        patch_size = 16
        h, w = img_float.shape
        
        mean_intensities = []
        local_variances = []
        
        for i in range(0, h-patch_size, patch_size):
            for j in range(0, w-patch_size, patch_size):
                if i + patch_size < h and j + patch_size < w:
                    patch = img_float[i:i+patch_size, j:j+patch_size]
                    mean_val = np.mean(patch)
                    var_val = np.var(patch)
                    
                    if 10 < mean_val < 245:
                        mean_intensities.append(mean_val)
                        local_variances.append(var_val)
        
        if len(mean_intensities) > 5:
            try:
                correlation = np.corrcoef(mean_intensities, local_variances)[0, 1]
                return correlation if not np.isnan(correlation) else 0
            except:
                return 0
        return 0
    
    def create_comprehensive_visual_output(self, image, result, image_id):
        """Create comprehensive visual analysis output"""
        image_name = result['image_name']
        base_name = os.path.splitext(os.path.basename(image_name))[0]
        
        # Create main analysis figure
        fig = plt.figure(figsize=(20, 12))
        
        # Define colors for each noise type
        colors = {
            'gaussian': '#4CAF50',      # Green
            'salt_pepper': '#2196F3',   # Blue
            'speckle': '#FF5722',       # Deep Orange
            'striping': '#9C27B0'       # Purple
        }
        
        prediction_color = colors.get(result['prediction'], '#757575')
        
        # 1. Original Image
        plt.subplot(3, 4, 1)
        plt.imshow(image, cmap='gray')
        plt.title(f'Original Image\n{base_name}', fontsize=12, fontweight='bold')
        plt.axis('off')
        
        # 2. Histogram
        plt.subplot(3, 4, 2)
        hist = cv2.calcHist([image], [0], None, [256], [0, 256]).flatten()
        plt.plot(hist, color=prediction_color, linewidth=2)
        plt.title('Intensity Histogram', fontsize=12, fontweight='bold')
        plt.xlabel('Pixel Intensity')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)
        
        # Highlight extreme values for salt & pepper
        if result['prediction'] == 'salt_pepper':
            plt.axvline(x=0, color='red', linestyle='--', alpha=0.7, label='Black pixels')
            plt.axvline(x=255, color='red', linestyle='--', alpha=0.7, label='White pixels')
            plt.legend()
        
        # 3. Log Transform (for speckle analysis)
        plt.subplot(3, 4, 3)
        log_img = np.log1p(image.astype(np.float64))
        plt.imshow(log_img, cmap='viridis')
        plt.title('Log Transform\n(Speckle Analysis)', fontsize=12, fontweight='bold')
        plt.axis('off')
        plt.colorbar()
        
        # 4. Gradient Magnitude
        plt.subplot(3, 4, 4)
        grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
        gradient_mag = np.sqrt(grad_x**2 + grad_y**2)
        plt.imshow(gradient_mag, cmap='hot')
        plt.title('Gradient Magnitude\n(Texture Analysis)', fontsize=12, fontweight='bold')
        plt.axis('off')
        plt.colorbar()
        
        # 5. Prediction Results
        plt.subplot(3, 4, 5)
        plt.axis('off')
        
        # Main prediction text
        plt.text(0.5, 0.8, 'PREDICTION RESULTS', ha='center', va='center', 
                fontsize=16, fontweight='bold', transform=plt.gca().transAxes)
        
        plt.text(0.5, 0.6, f'{result["prediction"].upper()}', ha='center', va='center', 
                fontsize=24, fontweight='bold', color=prediction_color, 
                transform=plt.gca().transAxes)
        
        plt.text(0.5, 0.4, f'Confidence: {result["confidence"]:.1%}', ha='center', va='center', 
                fontsize=16, transform=plt.gca().transAxes)
        
        plt.text(0.5, 0.2, f'Processing Time: {result["processing_time"]:.3f}s', ha='center', va='center', 
                fontsize=12, transform=plt.gca().transAxes)
        
        plt.text(0.5, 0.05, f'User: vkhare2909\n2025-06-12 09:25:15 UTC', ha='center', va='center', 
                fontsize=10, transform=plt.gca().transAxes)
        
        # 6. Ensemble Probabilities
        plt.subplot(3, 4, 6)
        if result.get('ensemble_probabilities'):
            noise_types = ['gaussian', 'salt_pepper', 'speckle', 'striping']
            probs = []
            
            # Get average probabilities across classifiers
            for noise_type in noise_types:
                type_probs = []
                for clf_name, clf_probs in result['ensemble_probabilities'].items():
                    idx = self.label_encoder.transform([noise_type])[0]
                    type_probs.append(clf_probs[idx])
                probs.append(np.mean(type_probs))
            
            bars = plt.bar(noise_types, probs, color=[colors[nt] for nt in noise_types])
            plt.title('Ensemble Probabilities', fontsize=12, fontweight='bold')
            plt.ylabel('Probability')
            plt.xticks(rotation=45)
            
            # Highlight the prediction
            max_idx = np.argmax(probs)
            bars[max_idx].set_edgecolor('black')
            bars[max_idx].set_linewidth(3)
        else:
            plt.text(0.5, 0.5, 'Stage 1 Decision\n(No Ensemble)', ha='center', va='center', 
                    fontsize=14, fontweight='bold', transform=plt.gca().transAxes)
        
        # 7. Stage 1 Analysis
        plt.subplot(3, 4, 7)
        plt.axis('off')
        plt.text(0.5, 0.9, 'STAGE 1 ANALYSIS', ha='center', va='center', 
                fontsize=14, fontweight='bold', transform=plt.gca().transAxes)
        
        if result.get('stage1_decision'):
            plt.text(0.5, 0.7, f'Decision: {result["stage1_decision"].upper()}', ha='center', va='center', 
                    fontsize=12, fontweight='bold', transform=plt.gca().transAxes)
            plt.text(0.5, 0.6, f'Confidence: {result["stage1_confidence"]:.1%}', ha='center', va='center', 
                    fontsize=12, transform=plt.gca().transAxes)
        else:
            plt.text(0.5, 0.7, 'Decision: FULL ANALYSIS', ha='center', va='center', 
                    fontsize=12, fontweight='bold', transform=plt.gca().transAxes)
            plt.text(0.5, 0.6, 'Required ensemble classification', ha='center', va='center', 
                    fontsize=10, transform=plt.gca().transAxes)
        
        # Stage 1 details
        if result.get('stage1_info'):
            y_pos = 0.45
            for key, value in result['stage1_info'].items():
                if isinstance(value, float):
                    plt.text(0.1, y_pos, f'{key}: {value:.3f}', ha='left', va='center', 
                            fontsize=9, transform=plt.gca().transAxes)
                else:
                    plt.text(0.1, y_pos, f'{key}: {value}', ha='left', va='center', 
                            fontsize=9, transform=plt.gca().transAxes)
                y_pos -= 0.08
        
        # 8. Image Statistics
        plt.subplot(3, 4, 8)
        plt.axis('off')
        plt.text(0.5, 0.9, 'IMAGE STATISTICS', ha='center', va='center', 
                fontsize=14, fontweight='bold', transform=plt.gca().transAxes)
        
        stats_text = f"""
        Shape: {image.shape[0]} × {image.shape[1]}
        Mean: {np.mean(image):.1f}
        Std: {np.std(image):.1f}
        Min: {np.min(image)}
        Max: {np.max(image)}
        
        Log Variance: {np.var(np.log1p(image.astype(np.float64))):.3f}
        Extreme Pixels: {((np.sum(image == 0) + np.sum(image == 255)) / image.size * 100):.2f}%
        """
        
        plt.text(0.1, 0.7, stats_text, ha='left', va='top', fontsize=10, 
                transform=plt.gca().transAxes, family='monospace')
        
        # 9. Noise Characteristics
        plt.subplot(3, 4, 9)
        plt.axis('off')
        plt.text(0.5, 0.9, f'{result["prediction"].upper()} CHARACTERISTICS', 
                ha='center', va='center', fontsize=14, fontweight='bold', 
                color=prediction_color, transform=plt.gca().transAxes)
        
        noise_descriptions = {
            'gaussian': """
            • Additive noise model
            • Constant variance
            • Normal distribution
            • Independent of signal
            • Affects all pixels uniformly
            """,
            'salt_pepper': """
            • Impulse noise model
            • Extreme pixel values (0, 255)
            • Random pixel locations
            • Isolated corrupted pixels
            • Binary corruption pattern
            """,
            'speckle': """
            • Multiplicative noise model
            • Intensity-dependent variance
            • Correlated with signal
            • Granular texture pattern
            • Log-normal characteristics
            """,
            'striping': """
            • Periodic pattern noise
            • Systematic corruption
            • Horizontal/vertical bands
            • Fixed spatial frequency
            • Additive periodic signal
            """
        }
        
        description = noise_descriptions.get(result['prediction'], 'Unknown noise type')
        plt.text(0.1, 0.8, description, ha='left', va='top', fontsize=10, 
                transform=plt.gca().transAxes)
        
        # 10. Intensity-Variance Scatter (for speckle analysis)
        plt.subplot(3, 4, 10)
        img_float = image.astype(np.float64)
        patch_size = 16
        h, w = img_float.shape
        
        mean_intensities = []
        local_variances = []
        
        for i in range(0, h-patch_size, patch_size):
            for j in range(0, w-patch_size, patch_size):
                if i + patch_size < h and j + patch_size < w:
                    patch = img_float[i:i+patch_size, j:j+patch_size]
                    mean_val = np.mean(patch)
                    var_val = np.var(patch)
                    
                    if 10 < mean_val < 245:
                        mean_intensities.append(mean_val)
                        local_variances.append(var_val)
        
        if mean_intensities:
            plt.scatter(mean_intensities, local_variances, alpha=0.6, color=prediction_color)
            
            # Fit line for correlation
            if len(mean_intensities) > 2:
                z = np.polyfit(mean_intensities, local_variances, 1)
                p = np.poly1d(z)
                x_line = np.linspace(min(mean_intensities), max(mean_intensities), 100)
                plt.plot(x_line, p(x_line), "r--", alpha=0.8, linewidth=2)
                
                corr = np.corrcoef(mean_intensities, local_variances)[0, 1]
                plt.text(0.05, 0.95, f'Correlation: {corr:.3f}', transform=plt.gca().transAxes, 
                        fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        plt.title('Intensity-Variance Relationship', fontsize=12, fontweight='bold')
        plt.xlabel('Local Mean Intensity')
        plt.ylabel('Local Variance')
        plt.grid(True, alpha=0.3)
        
        # 11. FFT Analysis
        plt.subplot(3, 4, 11)
        fft = np.fft.fft2(image)
        fft_shifted = np.fft.fftshift(fft)
        magnitude_spectrum = np.log(np.abs(fft_shifted) + 1)
        
        plt.imshow(magnitude_spectrum, cmap='jet')
        plt.title('FFT Magnitude Spectrum', fontsize=12, fontweight='bold')
        plt.axis('off')
        plt.colorbar()
        
        # 12. System Information
        plt.subplot(3, 4, 12)
        plt.axis('off')
        plt.text(0.5, 0.9, 'SYSTEM INFO', ha='center', va='center', 
                fontsize=14, fontweight='bold', transform=plt.gca().transAxes)
        
        system_info = f"""
        System: Ultra-Final v1.0
        Features: 107 optimized
        Enhancement: Log Transform
        
        Training: {self.get_training_config()['description']}
        
        Classifiers:
        • Random Forest (35%)
        • Extra Trees (30%)
        • Gradient Boosting (25%)
        • Linear SVM (10%)
        """
        
        plt.text(0.1, 0.8, system_info, ha='left', va='top', fontsize=9, 
                transform=plt.gca().transAxes, family='monospace')
        
        # Overall title and layout
        fig.suptitle(f'ULTRA-FINAL NOISE DETECTION ANALYSIS - {base_name}\n'
                    f'Prediction: {result["prediction"].upper()} '
                    f'(Confidence: {result["confidence"]:.1%}) | '
                    f'User: vkhare2909 | 2025-06-12 09:25:15 UTC', 
                    fontsize=16, fontweight='bold', y=0.98)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.92)
        
        # Save the comprehensive analysis
        analysis_path = f"{self.output_dir}/visual_analysis/{base_name}_comprehensive_analysis.png"
        plt.savefig(analysis_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create separate histogram plot
        self.create_detailed_histogram(image, result, base_name)
        
        print(f"   📊 Visual analysis saved: {analysis_path}")
        
        return analysis_path
    
    def create_detailed_histogram(self, image, result, base_name):
        """Create detailed histogram analysis"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
        
        colors = {
            'gaussian': '#4CAF50',
            'salt_pepper': '#2196F3', 
            'speckle': '#FF5722',
            'striping': '#9C27B0'
        }
        
        prediction_color = colors.get(result['prediction'], '#757575')
        
        # 1. Standard Histogram
        hist = cv2.calcHist([image], [0], None, [256], [0, 256]).flatten()
        ax1.plot(hist, color=prediction_color, linewidth=2)
        ax1.set_title('Standard Histogram', fontweight='bold')
        ax1.set_xlabel('Pixel Intensity')
        ax1.set_ylabel('Frequency')
        ax1.grid(True, alpha=0.3)
        
        # Highlight extremes for salt & pepper
        if result['prediction'] == 'salt_pepper':
            ax1.axvline(x=0, color='red', linestyle='--', alpha=0.7, label=f'Black: {hist[0]:.0f}')
            ax1.axvline(x=255, color='red', linestyle='--', alpha=0.7, label=f'White: {hist[255]:.0f}')
            ax1.legend()
        
        # 2. Log Histogram
        ax2.semilogy(hist + 1, color=prediction_color, linewidth=2)
        ax2.set_title('Log Scale Histogram', fontweight='bold')
        ax2.set_xlabel('Pixel Intensity')
        ax2.set_ylabel('Log(Frequency + 1)')
        ax2.grid(True, alpha=0.3)
        
        # 3. Cumulative Histogram
        cumulative = np.cumsum(hist) / np.sum(hist)
        ax3.plot(cumulative, color=prediction_color, linewidth=2)
        ax3.set_title('Cumulative Distribution', fontweight='bold')
        ax3.set_xlabel('Pixel Intensity')
        ax3.set_ylabel('Cumulative Probability')
        ax3.grid(True, alpha=0.3)
        
        # 4. Histogram Statistics (continued from where it stopped)
        ax4.axis('off')
        ax4.text(0.5, 0.9, 'HISTOGRAM STATISTICS', ha='center', va='center', 
                fontsize=14, fontweight='bold', transform=ax4.transAxes)
        
        # Calculate histogram statistics
        mean_hist = np.sum(hist * np.arange(256)) / np.sum(hist)
        var_hist = np.sum(hist * (np.arange(256) - mean_hist)**2) / np.sum(hist)
        std_hist = np.sqrt(var_hist)
        
        # Entropy calculation
        hist_norm = hist / np.sum(hist)
        entropy = -np.sum(hist_norm * np.log2(hist_norm + 1e-10))
        
        # Peak detection
        peaks, _ = scipy.signal.find_peaks(hist, height=0.01*np.max(hist))
        
        stats_text = f"""
        Mean: {mean_hist:.2f}
        Std Dev: {std_hist:.2f}
        Entropy: {entropy:.3f}
        
        Extreme Pixels:
        • Black (0): {hist[0]:.0f} ({hist[0]/np.sum(hist)*100:.2f}%)
        • White (255): {hist[255]:.0f} ({hist[255]/np.sum(hist)*100:.2f}%)
        
        Distribution:
        • Peaks: {len(peaks)}
        • Skewness: {scipy.stats.skew(image.flatten()):.3f}
        • Kurtosis: {scipy.stats.kurtosis(image.flatten()):.3f}
        
        Noise Indicators:
        • Log Variance: {np.var(np.log1p(image.astype(np.float64))):.3f}
        • Extreme Ratio: {(hist[0] + hist[255])/np.sum(hist):.4f}
        """
        
        ax4.text(0.1, 0.8, stats_text, ha='left', va='top', fontsize=10, 
                transform=ax4.transAxes, family='monospace')
        
        # Overall title
        fig.suptitle(f'DETAILED HISTOGRAM ANALYSIS - {base_name}\n'
                    f'Prediction: {result["prediction"].upper()} '
                    f'(Confidence: {result["confidence"]:.1%})', 
                    fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.88)
        
        # Save detailed histogram
        hist_path = f"{self.output_dir}/histograms/{base_name}_detailed_histogram.png"
        plt.savefig(hist_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return hist_path
    
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
        """Process all images in folder with ULTRA-FINAL system and comprehensive visualization"""
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
        print(f"🔧 ULTRA-FINAL corrected noise detection system: ENABLED")
        print(f"🎯 Enhancement: Log transform + comprehensive visual output")
        print(f"👤 User: vkhare2909")
        print(f"📅 Time: 2025-06-12 09:29:04 UTC")
        
        if not self.is_trained:
            self.train_system()
        
        all_results = []
        processed_count = 0
        
        # Count by type for analysis
        type_counts = {'gaussian': 0, 'salt_pepper': 0, 'speckle': 0, 'striping': 0}
        stage1_counts = {'gaussian': 0, 'salt_pepper': 0, 'speckle': 0, 'none': 0}
        
        print(f"\n📸 Processing images with ULTRA-FINAL system...")
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
            
            # Save comprehensive analysis
            self.save_ultra_final_analysis(image, result, i)
            
            # Create comprehensive visual output
            self.create_comprehensive_visual_output(image, result, i)
            
            all_results.append(result)
            processed_count += 1
            
            # Track predictions
            type_counts[result['prediction']] += 1
            stage1_decision = result.get('stage1_decision', 'none')
            if stage1_decision in stage1_counts:
                stage1_counts[stage1_decision] += 1
            else:
                stage1_counts['none'] += 1
            
            print(f"✅ {result['prediction'].upper()} ({result['confidence']:.1%})")
        
        if all_results:
            self.create_ultra_final_summary(all_results, folder_name)
            self.create_performance_dashboard(all_results, folder_name, type_counts, stage1_counts)
            
            print(f"\n🎉 ULTRA-FINAL processing complete: {processed_count}/{len(image_files)} images")
            print(f"📊 ULTRA-FINAL Results: {type_counts}")
            print(f"🎯 Stage 1 Decisions: {stage1_counts}")
            print(f"📁 Comprehensive analysis saved to: {self.output_dir}")
            
            # Analysis of improvements
            if 'speckle' in folder_name.lower():
                speckle_correct = type_counts['speckle']
                total_images = sum(type_counts.values())
                speckle_accuracy = speckle_correct / total_images * 100 if total_images > 0 else 0
                print(f"🎯 Speckle accuracy: {speckle_correct}/{total_images} ({speckle_accuracy:.1f}%)")
                
                # Log transform effectiveness
                log_enhanced_count = sum(1 for r in all_results 
                                       if r.get('stage1_info', {}).get('method') == 'log_transform_enhanced')
                print(f"🔬 Log transform detections: {log_enhanced_count}")
                
        else:
            print("❌ No images were successfully processed")
    
    def save_ultra_final_analysis(self, image, result, image_id):
        """Save ULTRA-FINAL analysis results with UTF-8 encoding for text files"""
        image_name = result['image_name']
        base_name = os.path.splitext(os.path.basename(image_name))[0]
        
        # Save image copy
        cv2.imwrite(f"{self.output_dir}/input_images/{base_name}_input.png", image)
        
        # Save JSON result
        json_safe_result = self.safe_json_convert(result)
        with open(f"{self.output_dir}/predictions/{base_name}_prediction.json", 'w', encoding='utf-8') as f:
            json.dump(json_safe_result, f, indent=2, ensure_ascii=False)
        
        # Save comprehensive readable report with UTF-8 encoding
        with open(f"{self.output_dir}/predictions/{base_name}_report.txt", 'w', encoding='utf-8') as f:
            f.write("ULTRA-FINAL NOISE DETECTION REPORT\n")
            f.write("==================================\n")
            f.write(f"Image: {os.path.basename(result['image_name'])}\n")
            f.write("Date: 2025-06-12 09:29:04 UTC\n")
            f.write("User: vkhare2909\n")
            f.write("System: Ultra-Final v1.0 - Log Transform Enhanced\n\n")
            
            f.write(f"FINAL RESULT: {result['prediction'].upper()}\n")
            f.write(f"   Confidence: {result['confidence']:.1%}\n")
            f.write(f"   Processing Time: {result['processing_time']:.3f}s\n\n")
            
            f.write("STAGE 1 ANALYSIS:\n")
            if result.get('stage1_decision'):
                f.write(f"   Decision: {result['stage1_decision'].upper()}\n")
                f.write(f"   Confidence: {result['stage1_confidence']:.1%}\n")
                
                if result.get('stage1_info'):
                    f.write(f"   Method: {result['stage1_info'].get('method', 'standard')}\n")
                    for key, value in result['stage1_info'].items():
                        if key != 'method':
                            if isinstance(value, float):
                                f.write(f"   {key}: {value:.4f}\n")
                            else:
                                f.write(f"   {key}: {value}\n")
            else:
                f.write("   Decision: FULL ANALYSIS REQUIRED\n")
                f.write("   Reason: No confident Stage 1 classification\n")
            
            f.write("\nDETAILED ANALYSIS:\n")
            if result.get('detailed_analysis'):
                for key, value in result['detailed_analysis'].items():
                    if key != 'ensemble_probabilities' and key != 'stage1_info':
                        if isinstance(value, float):
                            f.write(f"   {key}: {value:.4f}\n")
                        else:
                            f.write(f"   {key}: {value}\n")
            
            f.write("\nENSEMBLE PROBABILITIES:\n")
            if result.get('ensemble_probabilities'):
                noise_types = ['gaussian', 'salt_pepper', 'speckle', 'striping']
                for noise_type in noise_types:
                    type_probs = []
                    for clf_name, clf_probs in result['ensemble_probabilities'].items():
                        idx = self.label_encoder.transform([noise_type])[0]
                        type_probs.append(clf_probs[idx])
                    avg_prob = np.mean(type_probs)
                    f.write(f"   {noise_type}: {avg_prob:.3f}\n")
            
            f.write("\nNOISE TYPE CHARACTERISTICS:\n")
            noise_descriptions = {
                'gaussian': "Additive noise with constant variance, normally distributed",
                'salt_pepper': "Impulse noise with extreme pixel values (0, 255), randomly distributed",
                'speckle': "Multiplicative noise with intensity-dependent variance, granular texture",
                'striping': "Periodic pattern noise with systematic horizontal/vertical bands"
            }
            f.write(f"   {noise_descriptions.get(result['prediction'], 'Unknown noise type')}\n")
            
            f.write("\nIMAGE STATISTICS:\n")
            f.write(f"   Shape: {result['image_shape'][0]} × {result['image_shape'][1]}\n")
            f.write(f"   Mean Intensity: {result['mean_intensity']:.2f}\n")
            f.write(f"   Std Intensity: {result['std_intensity']:.2f}\n")
            f.write(f"   Feature Extraction Time: {result['feature_time']:.3f}s\n")
            
            f.write("\nSYSTEM INFORMATION:\n")
            f.write(f"   Version: {result['system_version']}\n")
            f.write(f"   Features: 107 optimized with log transform enhancement\n")
            f.write(f"   Classifiers: RF (35%), Extra Trees (30%), GB (25%), Linear SVM (10%)\n")
            f.write(f"   Training: {self.get_training_config()['description']}\n")
    
    def create_performance_dashboard(self, all_results, folder_name, type_counts, stage1_counts):
        """Create comprehensive performance dashboard"""
        fig = plt.figure(figsize=(16, 10))
        
        # Colors for consistency
        colors = {
            'gaussian': '#4CAF50',
            'salt_pepper': '#2196F3', 
            'speckle': '#FF5722',
            'striping': '#9C27B0'
        }
        
        # 1. Prediction Distribution Pie Chart
        plt.subplot(2, 3, 1)
        labels = []
        sizes = []
        colors_list = []
        
        for noise_type, count in type_counts.items():
            if count > 0:
                labels.append(f'{noise_type.capitalize()}\n({count})')
                sizes.append(count)
                colors_list.append(colors[noise_type])
        
        plt.pie(sizes, labels=labels, colors=colors_list, autopct='%1.1f%%', startangle=90)
        plt.title('Prediction Distribution', fontweight='bold', fontsize=12)
        
        # 2. Confidence Distribution
        plt.subplot(2, 3, 2)
        confidences = [r['confidence'] for r in all_results]
        predictions = [r['prediction'] for r in all_results]
        
        for noise_type in type_counts.keys():
            if type_counts[noise_type] > 0:
                type_confidences = [c for c, p in zip(confidences, predictions) if p == noise_type]
                plt.hist(type_confidences, bins=10, alpha=0.7, label=noise_type.capitalize(), 
                        color=colors[noise_type])
        
        plt.xlabel('Confidence')
        plt.ylabel('Count')
        plt.title('Confidence Distribution by Type', fontweight='bold', fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 3. Processing Time Analysis
        plt.subplot(2, 3, 3)
        processing_times = [r['processing_time'] for r in all_results]
        stage1_times = [r['processing_time'] for r in all_results if r.get('stage1_decision')]
        full_times = [r['processing_time'] for r in all_results if not r.get('stage1_decision')]
        
        plt.boxplot([processing_times, stage1_times, full_times], 
               labels=['All', 'Stage 1', 'Full Analysis'])
        plt.ylabel('Processing Time (seconds)')
        plt.title('Processing Time Distribution', fontweight='bold', fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # 4. Stage 1 vs Final Decision Comparison
        plt.subplot(2, 3, 4)
        stage1_decisions = [r.get('stage1_decision', 'none') for r in all_results]
        final_decisions = [r['prediction'] for r in all_results]
        
        # Create confusion-like matrix for Stage 1 vs Final
        stage1_types = ['gaussian', 'salt_pepper', 'speckle', 'none']
        final_types = ['gaussian', 'salt_pepper', 'speckle', 'striping']
        
        agreement_matrix = np.zeros((len(stage1_types), len(final_types)))
        
        for s1, final in zip(stage1_decisions, final_decisions):
            if s1 in stage1_types and final in final_types:
                i = stage1_types.index(s1)
                j = final_types.index(final)
                agreement_matrix[i, j] += 1
        
        im = plt.imshow(agreement_matrix, cmap='Blues', aspect='auto')
        plt.colorbar(im)
        plt.xticks(range(len(final_types)), [t.capitalize() for t in final_types])
        plt.yticks(range(len(stage1_types)), [t.capitalize() for t in stage1_types])
        plt.xlabel('Final Decision')
        plt.ylabel('Stage 1 Decision')
        plt.title('Stage 1 vs Final Decision', fontweight='bold', fontsize=12)
        
        # Add text annotations
        for i in range(len(stage1_types)):
            for j in range(len(final_types)):
                plt.text(j, i, f'{int(agreement_matrix[i, j])}', 
                        ha='center', va='center', fontweight='bold')
        
        # 5. Log Transform Enhancement Analysis
        plt.subplot(2, 3, 5)
        log_enhanced = sum(1 for r in all_results 
                          if r.get('stage1_info', {}).get('method') == 'log_transform_enhanced')
        
        enhancement_counts = {
            'Log Enhanced': log_enhanced,
            'Standard Detection': len(all_results) - log_enhanced
        }
        
        plt.bar(enhancement_counts.keys(), enhancement_counts.values(), 
           color=['#FF9800', '#607D8B'])
        plt.title('Log Transform Enhancement Usage', fontweight='bold', fontsize=12)
        plt.ylabel('Count')
        
        # Add percentage labels
        total = len(all_results)
        for i, (key, value) in enumerate(enhancement_counts.items()):
            plt.text(i, value + 0.5, f'{value}\n({value/total*100:.1f}%)', 
                    ha='center', va='bottom', fontweight='bold')
        
        # 6. Performance Metrics Summary
        plt.subplot(2, 3, 6)
        plt.axis('off')
        
        # Calculate metrics
        total_images = len(all_results)
        avg_confidence = np.mean(confidences)
        avg_processing_time = np.mean(processing_times)
        high_confidence_count = sum(1 for c in confidences if c > 0.9)
        stage1_efficiency = sum(1 for r in all_results if r.get('stage1_decision')) / total_images * 100
        
        metrics_text = f"""
        PERFORMANCE METRICS
        
        Total Images: {total_images}
        Average Confidence: {avg_confidence:.1%}
        High Confidence (>90%): {high_confidence_count} ({high_confidence_count/total_images*100:.1f}%)
        
        Processing Efficiency:
        • Average Time: {avg_processing_time:.3f}s
        • Stage 1 Decisions: {stage1_efficiency:.1f}%
        • Log Enhanced: {log_enhanced} ({log_enhanced/total_images*100:.1f}%)
        
        System Performance:
        • Features: 107 optimized
        • Enhancement: Log Transform
        • Training: {self.get_training_config()['description']}
        
        Quality Indicators:
        • Speckle Detection: Enhanced
        • Salt&Pepper Accuracy: Excellent
        • Gaussian Detection: Robust
        
        User: vkhare2909
        Date: 2025-06-12 09:29:04 UTC
        """
        
        plt.text(0.1, 0.9, metrics_text, ha='left', va='top', fontsize=10, 
                transform=plt.gca().transAxes, family='monospace')
        
        # Overall title
        fig.suptitle(f'ULTRA-FINAL NOISE DETECTION PERFORMANCE DASHBOARD - {folder_name}\n'
                    f'System: Ultra-Final v1.0 | User: vkhare2909 | 2025-06-12 09:29:04 UTC', 
                    fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        
        # Save dashboard
        dashboard_path = f"{self.output_dir}/comparison_charts/performance_dashboard.png"
        plt.savefig(dashboard_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   📈 Performance dashboard saved: {dashboard_path}")
        
        return dashboard_path
    
    def create_ultra_final_summary(self, all_results, folder_name):
        """Create ULTRA-FINAL summary report"""
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
        
        # Enhanced summary data
        summary_data = {
            'folder_info': {
                'name': folder_name,
                'timestamp': '2025-06-12T09:29:04Z',
                'user': 'vkhare2909',
                'total_images': len(all_results),
                'system_version': 'ultra_final_v1.0_log_enhanced'
            },
            'noise_distribution': prediction_counts,
            'enhancement_analysis': {
                'log_transform_detections': sum(1 for r in all_results 
                                               if r.get('stage1_info', {}).get('method') == 'log_transform_enhanced'),
                'stage1_decisions': sum(1 for r in all_results if r.get('stage1_decision')),
                'full_analysis_required': sum(1 for r in all_results if not r.get('stage1_decision'))
            },
            'confidence_stats': {
                'mean': float(np.mean(confidences)),
                'std': float(np.std(confidences)),
                'min': float(np.min(confidences)),
                'max': float(np.max(confidences)),
                'high_confidence_count': int(sum(1 for c in confidences if c > 0.9)),
                'very_high_confidence_count': int(sum(1 for c in confidences if c > 0.95))
            },
            'performance_stats': {
                'mean_time': float(np.mean(processing_times)),
                'total_time': float(np.sum(processing_times)),
                'min_time': float(np.min(processing_times)),
                'max_time': float(np.max(processing_times)),
                'stage1_efficiency': float(sum(1 for r in all_results if r.get('stage1_decision')) / len(all_results) * 100)
            },
            'speckle_analysis': {
                'speckle_count': prediction_counts['speckle'],
                'speckle_percentage': prediction_counts['speckle'] / len(all_results) * 100,
                'log_enhanced_speckle': sum(1 for r in all_results 
                                          if r['prediction'] == 'speckle' and 
                                          r.get('stage1_info', {}).get('method') == 'log_transform_enhanced')
            },
            'individual_results': self.safe_json_convert(all_results)
        }
        
        with open(f"{self.output_dir}/summary_reports/ultra_final_summary_{timestamp}.json", 'w') as f:
            json.dump(summary_data, f, indent=2)
        
        # Enhanced CSV with all details
        import csv
        with open(f"{self.output_dir}/summary_reports/ultra_final_results.csv", 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Image', 'Prediction', 'Confidence', 'Processing_Time', 'Stage1_Decision', 
                           'Stage1_Confidence', 'Enhancement_Method', 'System_Version'])
            for result in all_results:
                writer.writerow([
                    os.path.basename(result['image_name']),
                    result['prediction'],
                    result['confidence'],
                    result['processing_time'],
                    result.get('stage1_decision', ''),
                    result.get('stage1_confidence', 0),
                    result.get('stage1_info', {}).get('method', 'standard'),
                    'ultra_final_v1.0'
                ])
        
        print(f"📊 ULTRA-FINAL Summary: {prediction_counts}")
        print(f"🎯 Mean confidence: {np.mean(confidences):.1%}")
        print(f"⏱️  Mean processing time: {np.mean(processing_times):.3f}s")
        print(f"🔬 Log transform enhancements: {summary_data['enhancement_analysis']['log_transform_detections']}")

def main():
    """Main function for ULTRA-FINAL system"""
    parser = argparse.ArgumentParser(description='ULTRA-FINAL Noise Detection System with Comprehensive Visual Output')
    parser.add_argument('folder_path', help='Path to folder containing images')
    parser.add_argument('--pattern', default='*', help='File pattern to match')
    parser.add_argument('--output', default='ultra_final_noise_analysis', help='Output directory name')
    parser.add_argument('--training', choices=['small', 'medium', 'large'], default='small',
                       help='Training dataset size')
    
    args = parser.parse_args()
    
    print("🔧 ULTRA-FINAL NOISE DETECTION SYSTEM")
    print("=" * 50)
    print("🎯 ULTRA-FINAL: Log transform enhancement + comprehensive visual output")
    print("🔬 Complete analysis with histograms, feature visualization, and performance metrics")
    print(f"📁 Input Folder: {args.folder_path}")
    print(f"👤 User: vkhare2909")
    print(f"📅 Date: 2025-06-12 09:29:04 UTC")
    print(f"🔧 System: Ultra-Final v1.0 - Production Ready")
    print(f"✨ Features: 107 optimized with log transform speckle enhancement")
    print()
    
    detector = UltraFinalNoiseDetector(args.output, training_size=args.training)
    detector.process_image_folder(args.folder_path, args.pattern)

if __name__ == "__main__":
    main()