"""
Video Inference for Emotion Recognition

Offline video emotion prediction using Hybrid CNN + LSTM architecture.
"""

import os
import sys
import argparse
import cv2
import torch
import numpy as np
from collections import deque
from tqdm import tqdm

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from inference_utils import InferenceBase, aggregate_predictions, save_prediction_report


class VideoEmotionInference(InferenceBase):
    """
    Emotion inference from video files.
    
    Uses Hybrid CNN + LSTM architecture with frame sampling and sequence processing.
    """
    
    def __init__(self,
                 model_path: str,
                 config_path: str = 'configs/config.yaml',
                 sequence_length: int = 16,
                 frame_stride: int = 2):
        """
        Initialize video inference.
        
        Args:
            model_path: Path to trained model checkpoint
            config_path: Path to configuration file
            sequence_length: Number of frames per sequence
            frame_stride: Frame sampling stride (1 = use every frame)
        """
        super().__init__(model_path, config_path)
        
        self.sequence_length = sequence_length
        self.frame_stride = frame_stride
        
        print(f"Sequence length: {sequence_length} frames")
        print(f"Frame stride: {frame_stride} (sample every {frame_stride} frames)")
    
    def extract_video_info(self, video_path: str) -> dict:
        """
        Extract video metadata.
        
        Args:
            video_path: Path to video file
            
        Returns:
            Dictionary with video information
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        info = {
            'path': video_path,
            'fps': cap.get(cv2.CAP_PROP_FPS),
            'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'duration': cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)
        }
        
        cap.release()
        return info
    
    def process_video(self, 
                     video_path: str,
                     progress: bool = True) -> dict:
        """
        Process entire video and predict emotion.
        
        Args:
            video_path: Path to video file
            progress: Show progress bar
            
        Returns:
            Dictionary with prediction results
        """
        # Get video info
        print(f"\nAnalyzing video: {video_path}")
        video_info = self.extract_video_info(video_path)
        
        print(f"Duration: {video_info['duration']:.2f}s")
        print(f"Resolution: {video_info['width']}x{video_info['height']}")
        print(f"FPS: {video_info['fps']:.2f}")
        print(f"Total frames: {video_info['frame_count']}")
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        
        # Storage for features and predictions
        all_features = []
        frame_predictions = []
        valid_frames = 0
        skipped_frames = 0
        
        # Process frames with sampling
        frame_idx = 0
        sampled_frames = []
        
        iterator = range(video_info['frame_count'])
        if progress:
            iterator = tqdm(iterator, desc="Processing frames")
        
        for _ in iterator:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Sample frames according to stride
            if frame_idx % self.frame_stride == 0:
                # Process frame
                frame_data = self.process_single_frame(frame)
                
                if frame_data is not None:
                    # Extract CNN features
                    features = self.extract_cnn_features(frame_data)
                    all_features.append(features)
                    sampled_frames.append({
                        'index': frame_idx,
                        'data': frame_data,
                        'frame': frame
                    })
                    valid_frames += 1
                else:
                    skipped_frames += 1
            
            frame_idx += 1
        
        cap.release()
        
        print(f"\n✓ Processed {valid_frames} valid frames ({skipped_frames} frames skipped)")
        
        if valid_frames == 0:
            raise ValueError("No valid faces detected in video")
        
        # Process sequences with LSTM
        print("\nGenerating predictions from sequences...")
        sequence_predictions = self._process_sequences(all_features)
        
        # Aggregate predictions
        print("Aggregating predictions...")
        final_emotion, final_confidence, avg_probs = aggregate_predictions(
            sequence_predictions,
            method='majority_vote'
        )
        
        print(f"✓ Final prediction: {final_emotion} ({final_confidence*100:.1f}%)")
        
        # Prepare results
        results = {
            'emotion': final_emotion,
            'confidence': final_confidence,
            'probabilities': avg_probs,
            'emotions': self.emotions,
            'video_info': video_info,
            'valid_frames': valid_frames,
            'skipped_frames': skipped_frames,
            'sequence_predictions': sequence_predictions,
            'sampled_frames': sampled_frames
        }
        
        return results
    
    def _process_sequences(self, features_list: list) -> list:
        """
        Process features in sequences using LSTM.
        
        Args:
            features_list: List of feature tensors
            
        Returns:
            List of predictions
        """
        if len(features_list) < self.sequence_length:
            print(f"Warning: Only {len(features_list)} frames available, "
                  f"need {self.sequence_length} for sequence")
            # Pad with last frame
            while len(features_list) < self.sequence_length:
                features_list.append(features_list[-1])
        
        predictions = []
        
        # Create overlapping sequences
        num_sequences = max(1, len(features_list) - self.sequence_length + 1)
        
        for i in range(0, num_sequences, max(1, self.sequence_length // 2)):
            if i + self.sequence_length > len(features_list):
                break
            
            # Extract sequence
            sequence_features = features_list[i:i + self.sequence_length]
            
            # Stack features: (seq_len, feature_dim) -> (1, seq_len, feature_dim)
            sequence_tensor = torch.stack(sequence_features, dim=0).unsqueeze(0)
            
            # Predict
            emotion, confidence, probs = self.predict_with_lstm(sequence_tensor)
            predictions.append((emotion, confidence, probs))
        
        return predictions
    
    def process_and_save(self,
                        video_path: str,
                        output_dir: str = 'outputs/video_inference',
                        save_visualization: bool = True,
                        create_summary_video: bool = False):
        """
        Process video and save results.
        
        Args:
            video_path: Path to input video
            output_dir: Directory to save outputs
            save_visualization: Save annotated frames
            create_summary_video: Create summary video with predictions
        """
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Get base filename
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        
        # Process video
        results = self.process_video(video_path, progress=True)
        
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
                'Source': video_path,
                'Type': 'Video',
                'Duration': f"{results['video_info']['duration']:.2f}s",
                'Resolution': f"{results['video_info']['width']}x{results['video_info']['height']}",
                'FPS': f"{results['video_info']['fps']:.2f}",
                'Valid Frames': results['valid_frames'],
                'Skipped Frames': results['skipped_frames']
            }
        )
        
        # Save visualization of key frames
        if save_visualization and results['sampled_frames']:
            print("\nSaving visualization frames...")
            vis_dir = os.path.join(output_dir, f"{base_name}_frames")
            os.makedirs(vis_dir, exist_ok=True)
            
            # Save first, middle, and last frames
            key_indices = [
                0,
                len(results['sampled_frames']) // 2,
                len(results['sampled_frames']) - 1
            ]
            
            for idx in key_indices:
                if idx < len(results['sampled_frames']):
                    frame_info = results['sampled_frames'][idx]
                    vis_frame = self.visualize_prediction(
                        frame_info['frame'],
                        results['emotion'],
                        results['confidence'],
                        results['probabilities'],
                        frame_info['data']['landmarks']
                    )
                    
                    vis_path = os.path.join(vis_dir, f"frame_{frame_info['index']:06d}.jpg")
                    cv2.imwrite(vis_path, vis_frame)
            
            print(f"✓ Visualization frames saved to {vis_dir}")
        
        # Create summary video
        if create_summary_video and results['sampled_frames']:
            print("\nCreating summary video...")
            self._create_summary_video(
                results,
                os.path.join(output_dir, f"{base_name}_summary.mp4")
            )
        
        return results
    
    def _create_summary_video(self, results: dict, output_path: str):
        """
        Create annotated summary video.
        
        Args:
            results: Prediction results
            output_path: Path to save video
        """
        if not results['sampled_frames']:
            print("No frames to create video")
            return
        
        # Get video properties
        video_info = results['video_info']
        fps = video_info['fps'] / self.frame_stride
        width = video_info['width']
        height = video_info['height']
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Write frames
        for frame_info in tqdm(results['sampled_frames'], desc="Writing video"):
            vis_frame = self.visualize_prediction(
                frame_info['frame'],
                results['emotion'],
                results['confidence'],
                results['probabilities'],
                frame_info['data']['landmarks']
            )
            writer.write(vis_frame)
        
        writer.release()
        print(f"✓ Summary video saved: {output_path}")


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(
        description='Emotion Recognition from Video File',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python inference/video_inference.py --model checkpoints/best_model.pth --video input.mp4
  
  # Custom sequence length and stride
  python inference/video_inference.py --model checkpoints/best_model.pth --video input.mp4 --seq-len 32 --stride 3
  
  # Create summary video
  python inference/video_inference.py --model checkpoints/best_model.pth --video input.mp4 --summary-video
        """
    )
    
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--video', type=str, required=True,
                       help='Path to input video')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--output', type=str, default='outputs/video_inference',
                       help='Output directory for results')
    parser.add_argument('--seq-len', type=int, default=16,
                       help='Sequence length for LSTM')
    parser.add_argument('--stride', type=int, default=2,
                       help='Frame sampling stride')
    parser.add_argument('--summary-video', action='store_true',
                       help='Create annotated summary video')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.model):
        print(f"✗ Error: Model not found at {args.model}")
        return
    
    if not os.path.exists(args.video):
        print(f"✗ Error: Video not found at {args.video}")
        return
    
    # Print header
    print("\n" + "=" * 60)
    print("  EMOTION RECOGNITION - VIDEO FILE")
    print("=" * 60)
    
    try:
        # Create inference object
        inference = VideoEmotionInference(
            model_path=args.model,
            config_path=args.config,
            sequence_length=args.seq_len,
            frame_stride=args.stride
        )
        
        # Process and save
        results = inference.process_and_save(
            video_path=args.video,
            output_dir=args.output,
            save_visualization=True,
            create_summary_video=args.summary_video
        )
        
        # Print summary
        print("\n" + "=" * 60)
        print("  PREDICTION SUMMARY")
        print("=" * 60)
        print(f"  Emotion:     {results['emotion']}")
        print(f"  Confidence:  {results['confidence']*100:.2f}%")
        print(f"\n  Video Statistics:")
        print(f"    Duration:      {results['video_info']['duration']:.2f}s")
        print(f"    Total Frames:  {results['video_info']['frame_count']}")
        print(f"    Valid Frames:  {results['valid_frames']}")
        print(f"    Skipped:       {results['skipped_frames']}")
        print("\n  Probability Distribution:")
        for emotion, prob in zip(results['emotions'], results['probabilities']):
            bar = "█" * int(prob * 30)
            print(f"    {emotion:10s} {bar:30s} {prob*100:5.2f}%")
        print("=" * 60 + "\n")
        
        # Cleanup
        inference.close()
        
        print("✓ Video inference completed successfully")
        
    except Exception as e:
        print(f"\n✗ Error during inference: {str(e)}")
        import traceback
        traceback.print_exc()
        return


if __name__ == "__main__":
    main()
