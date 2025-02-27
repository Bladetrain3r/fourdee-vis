import json
import os
import numpy as np
import pygame
from math import sin, cos, pi
from OpenGL.GL import *
from OpenGL.GLU import *
import multiprocessing
from multiprocessing import shared_memory, Manager, Lock
import traceback
import gc
import atexit
import signal
import time
from scipy.io import wavfile
from scipy import signal as sci_signal
import librosa
import sys
import psutil
import tempfile
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger('hypersphere')

class BeatDetector:
    def __init__(self, audio_path):
        """Initialize beat detector with audio file path"""
        self.audio_path = audio_path
        self.beats = None
        self.duration = None
        self.sample_rate = None
        
    def analyze(self):
        """Analyze audio file to detect beats"""
        logger.info(f"Analyzing audio file: {self.audio_path}")
        
        try:
            # Load audio file using librosa for better beat detection
            y, sr = librosa.load(self.audio_path)
            self.sample_rate = sr
            
            # Get duration in seconds
            self.duration = librosa.get_duration(y=y, sr=sr)
            logger.info(f"Audio duration: {self.duration:.2f} seconds")
            
            # Run beat detection
            tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
            logger.info(f"Detected tempo: {tempo:.2f} BPM")
            
            # Convert beat frames to time in seconds
            self.beats = librosa.frames_to_time(beat_frames, sr=sr)
            logger.info(f"Detected {len(self.beats)} beats")
            
            return self.beats, self.duration
            
        except Exception as e:
            logger.error(f"Error analyzing audio: {str(e)}")
            logger.error(traceback.format_exc())
            return [], 0
    
    def get_zoom_envelope(self, fps, smooth_factor=0.15):
        """
        Create a zoom envelope based on beats
        
        Args:
            fps: Frames per second of the animation
            smooth_factor: Controls how quickly zoom transitions (lower = smoother)
        
        Returns:
            Array of zoom factors for each frame
        """
        if self.beats is None or self.duration is None:
            logger.warning("No beat data available. Run analyze() first.")
            return []
        
        # Calculate total frames based on duration and fps
        # Add a small buffer to ensure enough memory
        total_frames = int(self.duration * fps) + 10  # Add buffer frames
        
        # Initialize zoom envelope with base value
        zoom_envelope = np.ones(total_frames) * 2.0  # Base zoom factor
        
        # Convert beat times to frame indices
        beat_frames = [int(beat * fps) for beat in self.beats]
        
        # Apply exponential decay from each beat
        for beat_frame in beat_frames:
            if beat_frame >= total_frames:
                continue
                
            # Calculate decay for frames after the beat
            for i in range(beat_frame, min(beat_frame + int(fps), total_frames)):
                # Normalized position in the decay (0 to 1)
                decay_pos = (i - beat_frame) / (fps * smooth_factor)
                
                # Exponential decay function
                decay_factor = np.exp(-decay_pos) * 3.0  # Amplify by 3.0
                
                # Add the decay factor to the base zoom (peak at beat, then decay)
                zoom_envelope[i] = max(zoom_envelope[i], 2.0 + decay_factor)
        
        return zoom_envelope


class FrameTracker:
    """Class to track progress of frame rendering across processes"""
    def __init__(self, total_frames, manager):
        self.completed = manager.Value('i', 0)
        self.failed = manager.Value('i', 0)
        self.total = total_frames
        self.lock = manager.Lock()
        self.last_report_time = time.time()
        self.report_interval = 2  # seconds
        
    def mark_complete(self):
        with self.lock:
            self.completed.value += 1
            self._report_progress()
    
    def mark_failed(self):
        with self.lock:
            self.failed.value += 1
            self._report_progress()
    
    def _report_progress(self):
        # Only report progress at intervals to reduce console spam
        current_time = time.time()
        if current_time - self.last_report_time >= self.report_interval:
            self.last_report_time = current_time
            progress = (self.completed.value / self.total) * 100
            logger.info(f"Progress: {self.completed.value}/{self.total} frames completed ({progress:.1f}%), {self.failed.value} failed")


class HyperspherePipeline:
    def __init__(self, config_path='hypersphere_config.json', audio_path=None):
        # Load configuration
        try:
            with open(config_path, 'r') as config_file:
                self.config = json.load(config_file)
        except FileNotFoundError:
            logger.warning(f"Config file {config_path} not found, using defaults")
            self.config = {
                "color_depth": 8,
                "resolution": [1280, 720],
                "framerate": 60,
                "point_count": 25000,
                "duration": 6,
                "color_gradient": {
                    "low": [0, 0, 255],
                    "high": [255, 0, 0]
                },
                "batch_size": 120  # Default batch size (2 seconds at 60fps)
            }
        
        # Audio path
        self.audio_path = audio_path
        self.zoom_envelope = None
        
        # Memory optimization parameters
        # Use smaller batch sizes for large frame counts to prevent memory issues
        self.batch_size = self.config.get("batch_size", 120)
        
        # If audio path is provided, analyze and override duration
        if audio_path and os.path.exists(audio_path):
            beat_detector = BeatDetector(audio_path)
            beats, duration = beat_detector.analyze()
            
            # Override duration from config with audio duration
            self.config['duration'] = int(round(duration))
            
            # Generate zoom envelope based on beats
            self.zoom_envelope = beat_detector.get_zoom_envelope(self.config.get('framerate', 60))
            
            # Adjust batch size for longer durations
            total_frames = self.config['duration'] * self.config.get('framerate', 60)
            if total_frames > 5000:
                # For large frame counts, use smaller batches to avoid memory issues
                # This scales inversely with frame count to manage memory better
                adjusted_batch = max(30, min(self.batch_size, 600000 // total_frames))
                logger.info(f"Adjusting batch size to {adjusted_batch} for large frame count ({total_frames} frames)")
                self.batch_size = adjusted_batch
        
        # Get thread count from environment variable or set based on available memory
        mem_info = psutil.virtual_memory()
        # Estimate memory per thread (conservative estimate)
        # 25k points * 4 coordinates * 4 bytes ~ 400KB + buffer for processing
        mem_per_thread = 100 * 1024 * 1024  # 10MB per thread as a conservative estimate
        
        # Calculate thread count based on memory and CPU
        mem_based_threads = max(1, int(mem_info.available * 0.7 / mem_per_thread))
        cpu_based_threads = max(1, multiprocessing.cpu_count() - 1)  # Leave 1 core free
        
        # Use the minimum of memory-based and CPU-based thread counts
        suggested_threads = min(mem_based_threads, cpu_based_threads)
        
        # Get from environment or use calculated value
        self.thread_count = int(os.environ.get('RENDER_THREADS', suggested_threads))
        logger.info(f"Using {self.thread_count} render threads (memory: {mem_info.available/1024/1024:.0f}MB available)")
        
        # Silence audio
        os.environ['SDL_AUDIODRIVER'] = 'dummy'
        
        # Color depth configuration
        self.color_depth = self.config.get('color_depth', 8)  # Default to 8-bit
        
        # Initialize rendering parameters
        self.width = self.config.get('resolution', [1280, 720])[0]
        self.height = self.config.get('resolution', [1280, 720])[1]
        self.framerate = self.config.get('framerate', 60)
        self.point_count = self.config.get('point_count', 25000)
        
        # Color configuration
        self.color_gradient = self.config.get('color_gradient', {
            'low': [0, 0, 255],   # Blue (adjusted for 8-bit by default)
            'high': [255, 0, 0]   # Red (adjusted for 8-bit by default)
        })
        
        # Adjust color depth if using 10-bit
        if self.color_depth == 10:
            max_val = 1023
            if self.color_gradient['low'][2] == 255:
                self.color_gradient['low'][2] = max_val
            if self.color_gradient['high'][0] == 255:
                self.color_gradient['high'][0] = max_val
                
        # Setup exit handler to ensure shared memory cleanup
        atexit.register(self.cleanup_resources)
        signal.signal(signal.SIGINT, self.signal_handler)
        
        # Track shared memory resources
        self.shared_resources = []
        self.manager = Manager()
        
        # Create shared memory for points
        self.points_4d = self.generate_hypersphere_points_optimized()
        self.points_shm = shared_memory.SharedMemory(create=True, size=self.points_4d.nbytes)
        self.shared_resources.append(self.points_shm)
        
        # Create a NumPy array backed by shared memory
        shared_points = np.ndarray(self.points_4d.shape, dtype=self.points_4d.dtype, buffer=self.points_shm.buf)
        
        # Copy data to shared array
        np.copyto(shared_points, self.points_4d)
        
        # Create shared memory for zoom envelope if available
        if self.zoom_envelope is not None:
            self.zoom_shm = shared_memory.SharedMemory(create=True, size=self.zoom_envelope.nbytes)
            self.shared_resources.append(self.zoom_shm)
            
            # Create a NumPy array backed by shared memory
            shared_zoom = np.ndarray(self.zoom_envelope.shape, dtype=self.zoom_envelope.dtype, buffer=self.zoom_shm.buf)
            
            # Copy data to shared array
            np.copyto(shared_zoom, self.zoom_envelope)
            
            self.zoom_shm_name = self.zoom_shm.name
        else:
            self.zoom_shm_name = None

    def cleanup_resources(self):
        """Clean up any shared memory resources"""
        for shm in self.shared_resources:
            try:
                shm.close()
                shm.unlink()
                logger.info(f"Cleaned up shared memory resource: {shm.name}")
            except Exception as e:
                logger.error(f"Error cleaning up shared memory: {str(e)}")
        
        # Clear list after cleanup
        self.shared_resources = []

    def signal_handler(self, sig, frame):
        """Handle keyboard interrupts gracefully"""
        logger.info("\nInterrupted! Cleaning up resources...")
        self.cleanup_resources()
        sys.exit(0)

    @staticmethod
    def render_frame_static(frame_args):
        """Static method for multiprocessing to avoid class method complexities"""
        try:
            (frame, points_shm_name, zoom_shm_name, point_count, output_dir, 
             width, height, color_depth, color_gradient, framerate, duration,
             tracker) = frame_args
            
            # Access shared memory for points
            existing_shm = shared_memory.SharedMemory(name=points_shm_name)
            points_4d = np.ndarray((point_count, 4), dtype=np.float32, buffer=existing_shm.buf)
            
            # Set default zoom factor
            zoom_factor = 2.0  # Default value when no audio is used
            
            # Access shared memory for zoom envelope if available
            if zoom_shm_name:
                zoom_shm = shared_memory.SharedMemory(name=zoom_shm_name)
                
                # Get size of the shared memory buffer in terms of float64 items
                buffer_size = len(zoom_shm.buf) // np.dtype(np.float64).itemsize
                
                # Create the array using the actual buffer size
                zoom_envelope = np.ndarray((buffer_size,), dtype=np.float64, buffer=zoom_shm.buf)
                
                # Get zoom factor for current frame with safe indexing
                if frame < buffer_size:
                    zoom_factor = zoom_envelope[frame]
            
            # Initialize pygame for this process
            os.environ['SDL_VIDEO_WINDOW_POS'] = '0,0'  # Position window to reduce flicker
            # Set GPU to dGPU (if available) to avoid issues with OpenGL
            os.environ['DRI_PRIME'] = '1'
            pygame.init()
            display = pygame.display.set_mode(
                (width, height), 
                pygame.OPENGL | pygame.DOUBLEBUF | pygame.HIDDEN
            )
            
            # Local setup of OpenGL
            glEnable(GL_DEPTH_TEST)
            glEnable(GL_POINT_SMOOTH)
            glHint(GL_POINT_SMOOTH_HINT, GL_NICEST)
            glPointSize(2.0)
            
            glMatrixMode(GL_PROJECTION)
            glLoadIdentity()
            gluPerspective(45, (width / height), 0.1, 50.0)
            glMatrixMode(GL_MODELVIEW)

            # Clear buffers
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            glLoadIdentity()
            
            # Position camera
            glTranslatef(0, 0, -1)  # Move back to see the points
            
            # Calculate current state based on frame using the passed duration
            # This ensures we use the audio-based duration
            total_frames = framerate * duration
            w_position = frame / total_frames
            w_angle = w_position * pi * 2
            
            # Add pitch rotation
            pitch_angle = w_angle #  / 2  # Slower pitch than rotation

            if frame / total_frames > 0.5:
                pitch_angle = -pitch_angle  # Reverse pitch rotation

            glRotatef(np.degrees(pitch_angle), 0.5, 0.5, 0.0) # Slighly faster pitch rotation, changed to Y from X
            
            # Project points using the optimized method with beat-reactive color
            points_3d, colors = HyperspherePipeline.project_4d_to_3d_optimized(
                points_4d, w_angle, color_depth, color_gradient, zoom_factor
            )
            
            # Render points using OpenGL
            glBegin(GL_POINTS)
            for i in range(len(points_3d)):
                glColor3fv(colors[i])
                glVertex3fv(points_3d[i])
            glEnd()
            
            # Ensure all commands are executed
            glFinish()
            
            # Capture frame
            frame_data = glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE)
            
            # Convert to Pygame surface and save
            surface = pygame.image.fromstring(frame_data, (width, height), 'RGB')
            
            # Flip the image vertically (OpenGL coordinates are bottom-left)
            surface = pygame.transform.flip(surface, False, True)
            
            # Create output directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)
            
            # Save frame
            # For 10-bit color depth, use 32-bit BMP format
            if color_depth == 10:
                output_path = os.path.join(output_dir, f'frame_{frame:08d}.bmp')
                # Convert surface to 32-bit format
                surface = surface.convert(32, pygame.SRCALPHA)
                pygame.image.save(surface, output_path)
            else:
                output_path = os.path.join(output_dir, f'frame_{frame:08d}.bmp')
                pygame.image.save(surface, output_path)
            
            # Write to a temporary file first, then move it to the final location
            # This helps avoid potential file corruption if the process is interrupted
            temp_dir = os.path.join(output_dir, 'temp')
            os.makedirs(temp_dir, exist_ok=True)
            
            temp_path = os.path.join(temp_dir, f'temp_{frame:08d}.bmp')
            pygame.image.save(surface, temp_path)
            os.replace(temp_path, output_path)
            
            # Mark frame as complete
            tracker.mark_complete()
            
            # Clean up resources
            existing_shm.close()  # Important: just close, don't unlink
            if zoom_shm_name:
                zoom_shm.close()  # Close but don't unlink zoom shared memory
            pygame.quit()
            
            # Free memory
            del points_3d
            del colors
            del surface
            del frame_data
            gc.collect()

        except Exception as e:
            if tracker:
                tracker.mark_failed()
            logger.error(f"Error rendering frame {frame}: {str(e)}")
            logger.error(traceback.format_exc())
            
            # Make sure to close shared memory even on error
            try:
                existing_shm.close()
                if zoom_shm_name:
                    zoom_shm.close()
            except:
                pass

    @staticmethod
    def project_4d_to_3d_optimized(points_4d, w_angle, color_depth, color_gradient, zoom_factor):
        """Vectorized projection of 4D points to 3D with colors"""
        # Extract components for clarity
        x4, y4, z4, w4 = points_4d[:, 0], points_4d[:, 1], points_4d[:, 2], points_4d[:, 3]
        
        # 4D rotation (vectorized)
        cos_w, sin_w = np.cos(w_angle), np.sin(w_angle)
        rotated_w4 = w4 * cos_w - x4 * sin_w
        rotated_x4 = w4 * sin_w + x4 * cos_w

        # Get Frame rate and duration
        with open('hypersphere_config.json') as f:
            config = json.load(f)
        framerate = config.get('framerate', 60)
        duration = config.get('duration', 6)

        # Static projection factor
        total_frames = framerate * duration
        factor = 1.25  # Static factor as per your modification
        
        # Apply projection (vectorized)
        points_3d = np.column_stack([
            rotated_x4 * factor,
            y4 * factor,
            z4 * factor
        ])
        
        # Color interpolation (vectorized)
        max_color_value = 2**color_depth - 1
        t = (w4 + 1) / 2  # Normalize w4 to [0,1]
        
        # Normalize zoom factor to [0,1] range and combine with t
        zoom_influence = (zoom_factor - 2.0) / 3.0  # Normalize zoom from [2,5] to [0,1]
        zoom_influence = np.clip(zoom_influence, 0, 1)  # Ensure it stays in [0,1]
        t_modified = t * (1 + zoom_influence)  # Boost t based on zoom
        t_modified = np.clip(t_modified, 0, 1)  # Ensure final t stays in [0,1]
        
        # Create color arrays
        low = np.array(color_gradient['low']) / max_color_value
        high = np.array(color_gradient['high']) / max_color_value
        
        # Calculate colors all at once
        colors = np.outer(1-t_modified, low) + np.outer(t_modified, high)
        
        return points_3d, colors

    def generate_hypersphere_points_optimized(self):
        """Generate points on a 4D hypersphere using vectorized operations"""
        # Generate random spherical coordinates using numpy
        u = np.random.random(self.point_count) * 2 * np.pi
        v = np.random.random(self.point_count) * np.pi
        w = np.random.random(self.point_count) * np.pi
        
        # Calculate the coordinates in one step
        sin_u, cos_u = np.sin(u), np.cos(u)
        sin_v, cos_v = np.sin(v), np.cos(v)
        sin_w, cos_w = np.sin(w), np.cos(w)
        
        # Create the 4D points array directly
        points_4d = np.column_stack([
            sin_u * sin_v * sin_w,  # x4
            sin_u * sin_v * cos_w,  # y4
            sin_u * cos_v,          # z4
            cos_u                   # w4
        ]).astype(np.float32)  # Use float32 to reduce memory
        
        return points_4d

    def generate_animation_frames(self, output_dir='/mnt/d/hypersphere_frames'):
        """Generate animation frames in batches with progress tracking"""
        try:
            # Total frames calculation
            total_frames = self.framerate * self.config.get('duration', 6)
            logger.info(f"Preparing to render {total_frames} frames using {self.thread_count} processes")
            
            # Setup progress tracking
            tracker = FrameTracker(total_frames, self.manager)

            self.batch_size = self.batch_size
            
            # Process frames in batches to manage memory better
            for batch_start in range(0, total_frames, self.batch_size):
                batch_end = min(batch_start + self.batch_size, total_frames)
                batch_size = batch_end - batch_start  # Double buffer for better memory management
                
                logger.info(f"Processing batch {batch_start//self.batch_size + 1} " 
                          f"(frames {batch_start}-{batch_end-1}, {batch_size} frames)")
                
                # Prepare frame rendering arguments for this batch
                frame_args = [
                    (frame, self.points_shm.name, self.zoom_shm_name, self.point_count, output_dir, 
                    self.width, self.height, 
                    self.color_depth, self.color_gradient, 
                    self.framerate, self.config.get('duration', 6), tracker) 
                    for frame in range(batch_start, batch_end)
                ]
                
                # Process this batch
                with multiprocessing.Pool(processes=self.thread_count) as pool:
                    pool.map(self.render_frame_static, frame_args)
                
                # Explicitly call garbage collection between batches
                gc.collect()
                
                # Check if we should take a short break between batches to let system recover
                if batch_end < total_frames:
                    logger.info("Short pause between batches to manage memory...")
                    time.sleep(1)  # Short pause between batches
            
            # Verify all frames were created
            missing_frames = []
            for frame in range(total_frames):
                frame_path = os.path.join(output_dir, f'frame_{frame:08d}.bmp')
                if not os.path.exists(frame_path):
                    missing_frames.append(frame)
            
            if missing_frames:
                logger.warning(f"Missing {len(missing_frames)} frames. First few: {missing_frames[:5]}")
                
                # Re-render missing frames if there aren't too many
                if len(missing_frames) < total_frames * 0.1:  # Less than 10% missing
                    logger.info(f"Re-rendering {len(missing_frames)} missing frames...")
                    
                    # Prepare frame rendering arguments for missing frames
                    frame_args = [
                        (frame, self.points_shm.name, self.zoom_shm_name, self.point_count, output_dir, 
                        self.width, self.height, 
                        self.color_depth, self.color_gradient, 
                        self.framerate, self.config.get('duration', 6), tracker) 
                        for frame in missing_frames
                    ]
                    
                    # Process missing frames
                    with multiprocessing.Pool(processes=min(self.thread_count, len(missing_frames))) as pool:
                        pool.map(self.render_frame_static, frame_args)
            
            # Remove temporary directory
            temp_dir = os.path.join(output_dir, 'temp')
            if os.path.exists(temp_dir):
                try:
                    for file in os.listdir(temp_dir):
                        os.remove(os.path.join(temp_dir, file))
                    os.rmdir(temp_dir)
                except Exception as e:
                    logger.warning(f"Could not remove temp directory: {str(e)}")
            
            logger.info(f"Rendering complete. Frames saved to {os.path.abspath(output_dir)}")
            logger.info(f"Final stats: {tracker.completed.value} completed, {tracker.failed.value} failed")
            
        except Exception as e:
            logger.error(f"Error in animation generation: {str(e)}")
            logger.error(traceback.format_exc())
        finally:
            # Final cleanup
            self.cleanup_resources()

def main():
    # Set display if not set (for headless environments)
    if 'DISPLAY' not in os.environ:
        os.environ['DISPLAY'] = ':0'
    
    # Suppress pygame welcome message
    os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'
    
    # Import sys for exit handling
    import sys
    
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Generate 4D hypersphere animation with beat detection')
    parser.add_argument('--audio', '-a', help='Path to audio file (WAV) for beat detection')
    parser.add_argument('--config', '-c', default='hypersphere_config.json', help='Path to config file')
    parser.add_argument('--output', '-o', default='/mnt/d/hypersphere_frames', help='Output directory for frames')
    parser.add_argument('--batch-size', '-b', type=int, help='Override batch size for frame processing')
    parser.add_argument('--threads', '-t', type=int, help='Override number of processing threads')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose logging')
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    # Create and run the pipeline
    pipeline = HyperspherePipeline(config_path=args.config, audio_path=args.audio)
    
    # Override batch size if specified
    if args.batch_size:
        pipeline.batch_size = args.batch_size
        logger.info(f"Overriding batch size to {args.batch_size}")
    
    # Override thread count if specified
    if args.threads:
        pipeline.thread_count = args.threads
        logger.info(f"Overriding thread count to {args.threads}")
    
    try:
        pipeline.generate_animation_frames(output_dir=args.output)
    except KeyboardInterrupt:
        logger.info("\nInterrupted by user. Cleaning up...")
        pipeline.cleanup_resources()
        sys.exit(0)

if __name__ == "__main__":
    main()
