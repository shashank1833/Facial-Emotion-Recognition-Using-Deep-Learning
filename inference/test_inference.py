"""
Inference Testing and Demonstration Script

Tests all inference modes: image, video, and webcam.
"""

import os
import sys
import argparse
import cv2
import numpy as np

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def create_test_image(output_path='test_data/test_image.jpg'):
    """Create a synthetic test image with face-like features."""
    print("Creating synthetic test image...")
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Create blank canvas
    img = np.ones((480, 640, 3), dtype=np.uint8) * 200
    
    # Draw face outline
    cv2.circle(img, (320, 240), 120, (150, 150, 150), -1)
    
    # Draw eyes
    cv2.circle(img, (280, 210), 20, (50, 50, 50), -1)
    cv2.circle(img, (360, 210), 20, (50, 50, 50), -1)
    
    # Draw nose
    pts = np.array([[320, 230], [310, 270], [330, 270]], np.int32)
    cv2.fillPoly(img, [pts], (120, 120, 120))
    
    # Draw mouth (smiling)
    cv2.ellipse(img, (320, 290), (40, 20), 0, 0, 180, (80, 80, 80), 3)
    
    # Add text
    cv2.putText(img, "Test Image", (230, 50),
               cv2.FONT_HERSHEY_SIMPLEX, 1.5, (50, 50, 50), 3)
    
    cv2.imwrite(output_path, img)
    print(f"✓ Test image created: {output_path}")
    
    return output_path


def create_test_video(output_path='test_data/test_video.mp4', duration=5, fps=10):
    """Create a synthetic test video."""
    print("Creating synthetic test video...")
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Video properties
    width, height = 640, 480
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    total_frames = duration * fps
    
    for i in range(total_frames):
        # Create frame
        frame = np.ones((height, width, 3), dtype=np.uint8) * 200
        
        # Animate face position
        x = 320 + int(50 * np.sin(2 * np.pi * i / total_frames))
        y = 240
        
        # Draw face
        cv2.circle(frame, (x, y), 120, (150, 150, 150), -1)
        
        # Eyes
        cv2.circle(frame, (x - 40, y - 30), 20, (50, 50, 50), -1)
        cv2.circle(frame, (x + 40, y - 30), 20, (50, 50, 50), -1)
        
        # Nose
        pts = np.array([[x, y], [x - 10, y + 40], [x + 10, y + 40]], np.int32)
        cv2.fillPoly(frame, [pts], (120, 120, 120))
        
        # Mouth (animate)
        mouth_curve = 180 if (i // (total_frames // 4)) % 2 == 0 else -180
        cv2.ellipse(frame, (x, y + 60), (40, 20), 0, 0, abs(mouth_curve), (80, 80, 80), 3)
        
        # Frame number
        cv2.putText(frame, f"Frame {i+1}/{total_frames}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 50, 50), 2)
        
        writer.write(frame)
    
    writer.release()
    print(f"✓ Test video created: {output_path} ({duration}s, {fps} fps)")
    
    return output_path


def test_image_inference(model_path, config_path):
    """Test image inference."""
    print("\n" + "=" * 60)
    print("TESTING IMAGE INFERENCE")
    print("=" * 60)
    
    # Create test image
    test_image = create_test_image()
    
    # Import and run
    from image_inference import ImageEmotionInference
    
    inference = ImageEmotionInference(
        model_path=model_path,
        config_path=config_path
    )
    
    try:
        results = inference.predict_from_path(test_image, visualize=True)
        
        print(f"\n✓ Image inference successful!")
        print(f"  Emotion: {results['emotion']}")
        print(f"  Confidence: {results['confidence']*100:.2f}%")
        
        # Save result
        output_dir = 'test_outputs/image'
        os.makedirs(output_dir, exist_ok=True)
        cv2.imwrite(f"{output_dir}/result.jpg", results['visualization'])
        print(f"  Result saved to {output_dir}/result.jpg")
        
    except Exception as e:
        print(f"\n✗ Image inference failed: {str(e)}")
        import traceback
        traceback.print_exc()
    
    finally:
        inference.close()


def test_video_inference(model_path, config_path):
    """Test video inference."""
    print("\n" + "=" * 60)
    print("TESTING VIDEO INFERENCE")
    print("=" * 60)
    
    # Create test video
    test_video = create_test_video(duration=3, fps=10)
    
    # Import and run
    from video_inference import VideoEmotionInference
    
    inference = VideoEmotionInference(
        model_path=model_path,
        config_path=config_path,
        sequence_length=10,
        frame_stride=1
    )
    
    try:
        results = inference.process_video(test_video, progress=True)
        
        print(f"\n✓ Video inference successful!")
        print(f"  Emotion: {results['emotion']}")
        print(f"  Confidence: {results['confidence']*100:.2f}%")
        print(f"  Valid frames: {results['valid_frames']}")
        
    except Exception as e:
        print(f"\n✗ Video inference failed: {str(e)}")
        import traceback
        traceback.print_exc()
    
    finally:
        inference.close()


def test_inference_utils():
    """Test utility functions."""
    print("\n" + "=" * 60)
    print("TESTING INFERENCE UTILITIES")
    print("=" * 60)
    
    from inference_utils import aggregate_predictions
    
    # Mock predictions
    predictions = [
        ('Happy', 0.8, np.array([0.1, 0.05, 0.0, 0.8, 0.05, 0.0, 0.0])),
        ('Happy', 0.85, np.array([0.05, 0.05, 0.0, 0.85, 0.0, 0.05, 0.0])),
        ('Sad', 0.6, np.array([0.1, 0.05, 0.1, 0.05, 0.6, 0.05, 0.05])),
    ]
    
    # Test majority vote
    emotion, confidence, probs = aggregate_predictions(predictions, method='majority_vote')
    
    print(f"✓ Aggregation test passed")
    print(f"  Aggregated emotion: {emotion}")
    print(f"  Aggregated confidence: {confidence:.2f}")


def show_usage_examples():
    """Display usage examples."""
    print("\n" + "=" * 60)
    print("USAGE EXAMPLES")
    print("=" * 60)
    
    examples = """
1. IMAGE INFERENCE:
   python inference/image_inference.py \\
       --model checkpoints/best_model.pth \\
       --image photo.jpg

2. VIDEO INFERENCE:
   python inference/video_inference.py \\
       --model checkpoints/best_model.pth \\
       --video clip.mp4 \\
       --seq-len 16 \\
       --stride 2

3. WEBCAM (REAL-TIME):
   python inference/realtime_demo.py \\
       --model checkpoints/best_model.pth \\
       --source webcam

4. UNIFIED INTERFACE:
   # Image
   python inference/realtime_demo.py \\
       --model checkpoints/best_model.pth \\
       --source image \\
       --path photo.jpg
   
   # Video
   python inference/realtime_demo.py \\
       --model checkpoints/best_model.pth \\
       --source video \\
       --path clip.mp4 \\
       --summary-video
   
   # Webcam
   python inference/realtime_demo.py \\
       --model checkpoints/best_model.pth \\
       --source webcam \\
       --camera 0
    """
    
    print(examples)


def main():
    """Main test function."""
    parser = argparse.ArgumentParser(description='Test Inference Modules')
    parser.add_argument('--model', type=str, default='checkpoints/best_model.pth',
                       help='Path to trained model')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                       help='Path to config file')
    parser.add_argument('--mode', type=str, default='all',
                       choices=['all', 'image', 'video', 'utils', 'examples'],
                       help='Test mode')
    
    args = parser.parse_args()
    
    print("\n" + "=" * 60)
    print("EMOTION RECOGNITION - INFERENCE TEST SUITE")
    print("=" * 60)
    
    # Check if model exists
    if not os.path.exists(args.model) and args.mode != 'examples':
        print(f"\n⚠ Warning: Model not found at {args.model}")
        print("Some tests will be skipped.")
        print("\nTo train a model, run:")
        print("  python training/train.py --data data/fer2013/fer2013.csv")
        
        show_usage_examples()
        return
    
    # Run tests
    if args.mode in ['all', 'utils']:
        test_inference_utils()
    
    if args.mode in ['all', 'image']:
        test_image_inference(args.model, args.config)
    
    if args.mode in ['all', 'video']:
        test_video_inference(args.model, args.config)
    
    if args.mode == 'examples':
        show_usage_examples()
    
    print("\n" + "=" * 60)
    print("TEST SUITE COMPLETED")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Review test outputs in test_outputs/")
    print("2. Try inference on your own images/videos")
    print("3. Run webcam demo for real-time testing")
    print("\n")


if __name__ == "__main__":
    main()
