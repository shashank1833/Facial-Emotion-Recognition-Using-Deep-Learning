"""
Image Inference for Emotion Recognition

Single image emotion prediction using Hybrid CNN (no temporal LSTM).
"""

import os
import sys
import argparse
import cv2
import numpy as np

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from inference_utils import InferenceBase, save_prediction_report


class ImageEmotionInference(InferenceBase):
    """
    Emotion inference from single images.
    
    Uses Hybrid CNN architecture (global + zone features) without temporal modeling.
    """
    
    def __init__(self, model_path: str, config_path: str = 'configs/config.yaml'):
        """
        Initialize image inference.
        
        Args:
            model_path: Path to trained model checkpoint
            config_path: Path to configuration file
        """
        super().__init__(model_path, config_path)
        
        # Set detector to static image mode for better accuracy
        self.detector.static_image_mode = True
    
    def predict_from_path(self, image_path: str, visualize: bool = True) -> dict:
        """
        Predict emotion from image file.
        
        Args:
            image_path: Path to image file
            visualize: Whether to create visualization
            
        Returns:
            Dictionary with prediction results
        """
        # Load image
        print(f"\nLoading image: {image_path}")
        frame = cv2.imread(image_path)
        
        if frame is None:
            raise ValueError(f"Cannot read image from {image_path}")
        
        print(f"Image size: {frame.shape[1]}x{frame.shape[0]}")
        
        # Process frame
        print("Detecting face and extracting features...")

        # FER-style pre-cropped images (48x48, grayscale-like)
        if frame.shape[0] <= 64 and frame.shape[1] <= 64:
            print("✓ Detected pre-cropped FER-style image, skipping face detection")
            frame_data = self.process_single_frame(frame, skip_detection=True)
        else:
            frame_data = self.process_single_frame(frame)
        
        if frame_data is None:
            raise ValueError("No face detected in image")
        
        print("✓ Face detected successfully")
        
        # Predict emotion (CNN only, no LSTM)
        print("Predicting emotion...")
        emotion, confidence, probabilities = self.predict_cnn_only(frame_data)
        
        print(f"✓ Prediction complete: {emotion} ({confidence*100:.1f}%)")
        
        # Prepare results
        results = {
            'emotion': emotion,
            'confidence': confidence,
            'probabilities': probabilities,
            'emotions': self.emotions,
            'frame': frame,
            'frame_data': frame_data
        }
        
        # Create visualization if requested
        if visualize:
            results['visualization'] = self.visualize_prediction(
                frame, emotion, confidence, probabilities,
                frame_data['landmarks']
            )
        
        return results
    
    def predict_from_array(self, image: np.ndarray, visualize: bool = True) -> dict:
        """
        Predict emotion from numpy array.
        
        Args:
            image: Image as numpy array (BGR)
            visualize: Whether to create visualization
            
        Returns:
            Dictionary with prediction results
        """
        print("Processing image array...")
        
        # Process frame
        frame_data = self.process_single_frame(image)
        
        if frame_data is None:
            raise ValueError("No face detected in image")
        
        # Predict emotion
        emotion, confidence, probabilities = self.predict_cnn_only(frame_data)
        
        # Prepare results
        results = {
            'emotion': emotion,
            'confidence': confidence,
            'probabilities': probabilities,
            'emotions': self.emotions,
            'frame': image,
            'frame_data': frame_data
        }
        
        if visualize:
            results['visualization'] = self.visualize_prediction(
                image, emotion, confidence, probabilities,
                frame_data['landmarks']
            )
        
        return results
    
    def process_and_save(self, 
                        image_path: str,
                        output_dir: str = 'outputs/image_inference',
                        show_result: bool = True):
        """
        Process image and save results.
        
        Args:
            image_path: Path to input image
            output_dir: Directory to save outputs
            show_result: Whether to display result window
        """
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Get base filename
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        
        # Predict
        results = self.predict_from_path(image_path, visualize=True)
        
        # Save visualization
        vis_path = os.path.join(output_dir, f"{base_name}_prediction.jpg")
        cv2.imwrite(vis_path, results['visualization'])
        print(f"✓ Visualization saved: {vis_path}")
        
        # Save report
        report_path = os.path.join(output_dir, f"{base_name}_report.txt")
        save_prediction_report(
            report_path,
            {
                'emotion': results['emotion'],
                'confidence': results['confidence'],
                'probabilities': results['probabilities'],
                'emotions': self.emotions
            },
            {
                'Source': image_path,
                'Type': 'Single Image',
                'Resolution': f"{results['frame'].shape[1]}x{results['frame'].shape[0]}"
            }
        )
        
        # Display result
        if show_result:
            print("\nDisplaying result (press any key to close)...")
            cv2.imshow('Emotion Recognition - Image', results['visualization'])
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        return results


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(
        description='Emotion Recognition from Single Image',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python inference/image_inference.py --model checkpoints/best_model.pth --image photo.jpg
  
  # Save outputs without display
  python inference/image_inference.py --model checkpoints/best_model.pth --image photo.jpg --no-display
  
  # Custom output directory
  python inference/image_inference.py --model checkpoints/best_model.pth --image photo.jpg --output results/
        """
    )
    
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--image', type=str, required=True,
                       help='Path to input image')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--output', type=str, default='outputs/image_inference',
                       help='Output directory for results')
    parser.add_argument('--no-display', action='store_true',
                       help='Do not display result window')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.model):
        print(f"✗ Error: Model not found at {args.model}")
        return
    
    if not os.path.exists(args.image):
        print(f"✗ Error: Image not found at {args.image}")
        return
    
    # Print header
    print("\n" + "=" * 60)
    print("  EMOTION RECOGNITION - SINGLE IMAGE")
    print("=" * 60)
    
    try:
        # Create inference object
        inference = ImageEmotionInference(
            model_path=args.model,
            config_path=args.config
        )
        
        # Process and save
        results = inference.process_and_save(
            image_path=args.image,
            output_dir=args.output,
            show_result=not args.no_display
        )
        
        # Print summary
        print("\n" + "=" * 60)
        print("  PREDICTION SUMMARY")
        print("=" * 60)
        print(f"  Emotion:     {results['emotion']}")
        print(f"  Confidence:  {results['confidence']*100:.2f}%")
        print("\n  Probability Distribution:")
        for emotion, prob in zip(results['emotions'], results['probabilities']):
            bar = "█" * int(prob * 30)
            print(f"    {emotion:10s} {bar:30s} {prob*100:5.2f}%")
        print("=" * 60 + "\n")
        
        # Cleanup
        inference.close()
        
        print("✓ Image inference completed successfully")
        
    except Exception as e:
        print(f"\n✗ Error during inference: {str(e)}")
        import traceback
        traceback.print_exc()
        return


if __name__ == "__main__":
    main()
