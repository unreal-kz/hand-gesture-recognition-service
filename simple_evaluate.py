#!/usr/bin/env python3
"""
Simple Performance Evaluation Script
Tests the hand gesture recognition service and calculates basic metrics
"""

import os
import requests
import base64
import json
from datetime import datetime

class SimpleEvaluator:
    def __init__(self, port=8000):
        self.dataset_dir = "hand_gesture_dataset"
        self.service_url = f"http://localhost:{port}"
        self.results = []
        self.true_labels = []
        self.predicted_labels = []
        
        # Define gesture mapping
        self.gesture_mapping = {
            'close': 'Close',
            'one': 'One', 
            'two': 'Two',
            'three': 'Three',
            'four': 'Four',
            'open': 'Open'
        }
        
    def test_service_health(self):
        """Test if the service is running"""
        try:
            print(f"🔍 Testing service at: {self.service_url}")
            response = requests.get(f"{self.service_url}/health", timeout=10)
            if response.status_code == 200:
                health_data = response.json()
                print(f"✅ Service is healthy: {health_data['status']}")
                return True
            else:
                print(f"❌ Service health check failed: {response.status_code}")
                return False
        except requests.exceptions.ConnectionError:
            print(f"❌ Cannot connect to service at {self.service_url}")
            print("   Make sure the service is running with: python app.py")
            return False
        except Exception as e:
            print(f"❌ Error testing service: {e}")
            return False
    
    def find_images(self):
        """Find all images in the dataset"""
        images = []
        
        if not os.path.exists(self.dataset_dir):
            print(f"❌ Dataset directory not found: {self.dataset_dir}")
            return images
            
        for gesture_dir in os.listdir(self.dataset_dir):
            gesture_path = os.path.join(self.dataset_dir, gesture_dir)
            if os.path.isdir(gesture_path) and gesture_dir in self.gesture_mapping:
                for filename in os.listdir(gesture_path):
                    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                        images.append({
                            'path': os.path.join(gesture_path, filename),
                            'true_gesture': gesture_dir,
                            'filename': filename
                        })
        
        return images
    
    def test_image(self, image_info):
        """Test a single image with the service"""
        print(f"🔍 Testing: {image_info['true_gesture']}/{image_info['filename']}")
        
        try:
            # Convert image to base64
            with open(image_info['path'], 'rb') as image_file:
                encoded_string = base64.b64encode(image_file.read()).decode()
            
            # Test with service
            response = requests.post(
                f"{self.service_url}/detect-fingers",
                json={"image_base64": encoded_string},
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                
                # Extract prediction
                predicted_gesture = result['detected_gesture']
                confidence = result['confidence']
                processing_time = result['processing_time_ms']
                
                print(f"✅ Success!")
                print(f"   🎯 True: {image_info['true_gesture'].title()}")
                print(f"   🔮 Predicted: {predicted_gesture}")
                print(f"   📊 Confidence: {confidence:.3f}")
                print(f"   ⏱️  Time: {processing_time}ms")
                
                # Check if prediction is correct
                prediction_correct = predicted_gesture.lower() == image_info['true_gesture'].lower()
                
                if prediction_correct:
                    print(f"   🎉 CORRECT!")
                else:
                    print(f"   ❌ WRONG")
                
                # Store for evaluation
                self.true_labels.append(image_info['true_gesture'].title())
                self.predicted_labels.append(predicted_gesture)
                
                # Store detailed result
                test_result = {
                    'image': image_info['filename'],
                    'true_gesture': image_info['true_gesture'].title(),
                    'predicted_gesture': predicted_gesture,
                    'confidence': confidence,
                    'processing_time_ms': processing_time,
                    'correct': prediction_correct,
                    'success': True,
                    'error': None
                }
                
                return True
                
            else:
                print(f"❌ Error: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            print(f"❌ Failed: {e}")
            return False
    
    def calculate_basic_metrics(self):
        """Calculate basic performance metrics"""
        if not self.true_labels or not self.predicted_labels:
            print("❌ No predictions available for evaluation")
            return None
            
        total = len(self.true_labels)
        correct = sum(1 for i, true in enumerate(self.true_labels) 
                     if self.predicted_labels[i].lower() == true.lower())
        
        accuracy = correct / total if total > 0 else 0
        
        # Calculate per-gesture metrics
        gesture_stats = {}
        for gesture in self.gesture_mapping.values():
            gesture_stats[gesture] = {'total': 0, 'correct': 0, 'precision': 0, 'recall': 0}
        
        # Count true positives, false positives, false negatives
        for i, true in enumerate(self.true_labels):
            predicted = self.predicted_labels[i]
            gesture_stats[true]['total'] += 1
            if predicted.lower() == true.lower():
                gesture_stats[true]['correct'] += 1
        
        # Calculate precision and recall for each gesture
        for gesture, stats in gesture_stats.items():
            if stats['total'] > 0:
                stats['recall'] = stats['correct'] / stats['total']
            
            # Count false positives (predicted as this gesture but actually different)
            false_positives = sum(1 for pred in self.predicted_labels 
                                if pred.lower() == gesture.lower() and 
                                pred.lower() != gesture.lower())
            
            total_predicted = stats['correct'] + false_positives
            if total_predicted > 0:
                stats['precision'] = stats['correct'] / total_predicted
        
        # Calculate macro F1 score
        f1_scores = []
        for gesture, stats in gesture_stats.items():
            if stats['precision'] + stats['recall'] > 0:
                f1 = 2 * (stats['precision'] * stats['recall']) / (stats['precision'] + stats['recall'])
                f1_scores.append(f1)
        
        macro_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0
        
        return {
            'accuracy': accuracy,
            'macro_f1': macro_f1,
            'gesture_stats': gesture_stats,
            'total_tests': total,
            'correct_predictions': correct
        }
    
    def print_report(self, metrics):
        """Print performance report"""
        if not metrics:
            return
            
        print("\n" + "="*80)
        print("📊 PERFORMANCE EVALUATION REPORT")
        print("="*80)
        
        print(f"\n🎯 OVERALL PERFORMANCE:")
        print(f"   Total Tests:     {metrics['total_tests']}")
        print(f"   Correct:         {metrics['correct_predictions']}")
        print(f"   Accuracy:        {metrics['accuracy']:.4f}")
        print(f"   Macro F1 Score:  {metrics['macro_f1']:.4f}")
        
        print(f"\n📋 PER-GESTURE PERFORMANCE:")
        print(f"{'Gesture':<12} {'Total':<8} {'Correct':<8} {'Precision':<10} {'Recall':<10} {'F1-Score':<10}")
        print("-" * 70)
        
        for gesture, stats in metrics['gesture_stats'].items():
            if stats['total'] > 0:
                f1 = 2 * (stats['precision'] * stats['recall']) / (stats['precision'] + stats['recall']) if (stats['precision'] + stats['recall']) > 0 else 0
                print(f"{gesture:<12} {stats['total']:<8} {stats['correct']:<8} {stats['precision']:<10.4f} {stats['recall']:<10.4f} {f1:<10.4f}")
        
        # Performance interpretation
        print(f"\n💡 PERFORMANCE INTERPRETATION:")
        f1_score_val = metrics['macro_f1']
        if f1_score_val >= 0.9:
            print("   🎉 EXCELLENT: Your service is performing exceptionally well!")
        elif f1_score_val >= 0.8:
            print("   🚀 VERY GOOD: Your service is performing very well!")
        elif f1_score_val >= 0.7:
            print("   👍 GOOD: Your service is performing well with room for improvement.")
        elif f1_score_val >= 0.6:
            print("   ⚠️  FAIR: Your service needs some improvements.")
        else:
            print("   ❌ POOR: Your service needs significant improvements.")
    
    def run_evaluation(self):
        """Run the complete evaluation"""
        print("🚀 Simple Performance Evaluation - Hand Gesture Recognition Service")
        print("="*80)
        
        # Check service health
        if not self.test_service_health():
            print(f"\n💡 Try these solutions:")
            print(f"   1. Start the service: python app.py")
            print(f"   2. Check if port 8000 is free: lsof -i :8000")
            print(f"   3. Try a different port by modifying the script")
            return
        
        # Find images
        images = self.find_images()
        if not images:
            return
            
        print(f"\n📸 Found {len(images)} images to evaluate")
        
        # Test each image
        successful_tests = 0
        for i, image_info in enumerate(images, 1):
            print(f"\n[{i}/{len(images)}]", end=" ")
            if self.test_image(image_info):
                successful_tests += 1
        
        print(f"\n✅ Successfully tested {successful_tests}/{len(images)} images")
        
        # Calculate metrics
        if self.true_labels and self.predicted_labels:
            print(f"\n🔬 Calculating performance metrics...")
            metrics = self.calculate_basic_metrics()
            
            if metrics:
                self.print_report(metrics)
                
                # Save results
                self.save_results(metrics)
            else:
                print("❌ Failed to calculate metrics")
        else:
            print("❌ No successful predictions to evaluate")
    
    def save_results(self, metrics):
        """Save evaluation results to file"""
        results_file = f"simple_evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        save_data = {
            'evaluation_date': datetime.now().isoformat(),
            'service_url': self.service_url,
            'performance_metrics': metrics,
            'detailed_results': self.results
        }
        
        with open(results_file, 'w') as f:
            json.dump(save_data, f, indent=2)
        
        print(f"\n💾 Evaluation results saved to: {results_file}")

def main():
    """Main function"""
    try:
        # Try different ports if 8000 doesn't work
        ports_to_try = [8000, 8001, 8002]
        
        for port in ports_to_try:
            print(f"\n🔌 Trying port {port}...")
            evaluator = SimpleEvaluator(port=port)
            
            # Test if service is running on this port
            if evaluator.test_service_health():
                print(f"✅ Service found on port {port}")
                evaluator.run_evaluation()
                break
            else:
                print(f"❌ No service on port {port}")
        else:
            print("\n❌ Could not find service on any port")
            print("   Please start the service with: python app.py")
            
    except KeyboardInterrupt:
        print("\n\n👋 Evaluation interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
