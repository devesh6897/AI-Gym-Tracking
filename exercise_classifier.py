import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from collections import deque
import argparse
import math

# Pose keypoint mapping (MediaPipe indices to custom format)
KEYPOINT_MAP = {
    0: 0,    # nose
    2: 1,    # left eye
    5: 2,    # right eye
    7: 3,    # left ear
    8: 4,    # right ear
    11: 5,   # left shoulder
    12: 6,   # right shoulder
    13: 7,   # left elbow
    14: 8,   # right elbow
    15: 9,   # left wrist
    16: 10,  # right wrist
    23: 11,  # left hip
    24: 12,  # right hip
    25: 13,  # left knee
    26: 14,  # right knee
    27: 15,  # left ankle
    28: 16,  # right ankle
    1: 17,   # mid-point reference
}

class PoseEstimation:
    """Handles the pose estimation and landmark processing."""
    
    def __init__(self):
        self.pose_tracker = mp.solutions.pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.pose_drawer = mp.solutions.drawing_utils
        
    def detect_pose(self, frame):
        """Process the frame and detect pose landmarks."""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return self.pose_tracker.process(rgb_frame)
    
    def draw_landmarks(self, frame, landmarks):
        """Draw the pose landmarks on the frame."""
        self.pose_drawer.draw_landmarks(
            frame, 
            landmarks,
            mp.solutions.pose.POSE_CONNECTIONS,
            self.pose_drawer.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
            self.pose_drawer.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
        )
    
    def draw_landmark_numbers(self, frame, landmarks, show_original_indices=True):
        """Draw joint numbers on landmarks according to the specified mode."""
        h, w, c = frame.shape
        keypoint_map = KEYPOINT_MAP
        
        # Draw numbers at each landmark
        for mp_idx, custom_idx in keypoint_map.items():
            # Skip mid-point reference if not visible
            if mp_idx == 1 and landmarks.landmark[mp_idx].visibility < 0.5:
                continue
                
            # Get landmark coordinates
            landmark = landmarks.landmark[mp_idx]
            cx, cy = int(landmark.x * w), int(landmark.y * h)
            
            # Skip landmarks with low visibility
            if landmark.visibility < 0.5:
                continue
                
            # Choose which index to display based on mode
            if show_original_indices:
                # Show MediaPipe original indices
                index_text = str(mp_idx)
            else:
                # Show custom mapping indices
                index_text = str(custom_idx)
                
            # Draw circle at landmark for better visibility
            cv2.circle(frame, (cx, cy), 5, (0, 255, 255), -1)
            
            # Adjust text position based on landmark location to prevent overlapping
            # Face landmarks (nose, eyes, ears)
            if mp_idx in [0, 1, 2, 5, 7, 8]:
                # Spread out face landmark labels
                if mp_idx == 0:  # nose
                    text_x, text_y = cx - 20, cy - 10  # left side
                elif mp_idx == 2:  # left eye
                    text_x, text_y = cx - 20, cy - 10  # left side
                elif mp_idx == 5:  # right eye
                    text_x, text_y = cx + 10, cy - 10  # right side
                elif mp_idx == 7:  # left ear
                    text_x, text_y = cx - 25, cy  # far left
                elif mp_idx == 8:  # right ear
                    text_x, text_y = cx + 10, cy  # far right
                elif mp_idx == 1:  # mid-point reference
                    text_x, text_y = cx, cy + 15  # below
                else:
                    text_x, text_y = cx + 8, cy - 8  # default offset
            # Body landmarks
            elif mp_idx in [11, 12]:  # shoulders
                text_x, text_y = cx, cy - 10  # above
            elif mp_idx in [23, 24]:  # hips
                text_x, text_y = cx, cy - 10  # above
            elif mp_idx in [13, 14]:  # elbows
                text_x, text_y = cx + 10, cy  # right
            elif mp_idx in [15, 16]:  # wrists
                text_x, text_y = cx + 10, cy  # right
            elif mp_idx in [25, 26]:  # knees
                text_x, text_y = cx - 20, cy  # left
            elif mp_idx in [27, 28]:  # ankles
                text_x, text_y = cx - 20, cy  # left
            else:
                text_x, text_y = cx + 6, cy + 6  # default offset
            
            # Draw simple index number without background
            cv2.putText(frame, index_text, 
                       (text_x, text_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
                       (0, 0, 0), 1)

class BiomechanicsAnalyzer:
    """Analyzes body mechanics and joint movements for exercise evaluation."""
    
    @staticmethod
    def calculate_joint_angle(point1, point2, point3):
        """Calculate angle between three points (joint angle)."""
        # Convert to vectors
        p1 = np.array([point1.x, point1.y])
        p2 = np.array([point2.x, point2.y])
        p3 = np.array([point3.x, point3.y])
        
        # Calculate vectors from point2 to point1 and point3
        v1 = p1 - p2
        v2 = p3 - p2
        
        # Calculate dot product and magnitudes
        dot_product = np.dot(v1, v2)
        magnitude_v1 = np.linalg.norm(v1)
        magnitude_v2 = np.linalg.norm(v2)
        
        # Calculate angle using arc cosine of the normalized dot product
        cosine_angle = dot_product / (magnitude_v1 * magnitude_v2)
        angle_radians = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
        
        # Convert to degrees
        angle_degrees = np.degrees(angle_radians)
        
        return angle_degrees
    
    def analyze_body_angles(self, landmarks, pose_landmarks):
        """Compute all relevant body angles for exercise analysis."""
        # Body landmarks
        mp_pose = mp.solutions.pose.PoseLandmark
        
        # Elbow angles
        left_elbow = self.calculate_joint_angle(
            landmarks[mp_pose.LEFT_SHOULDER.value],
            landmarks[mp_pose.LEFT_ELBOW.value],
            landmarks[mp_pose.LEFT_WRIST.value]
        )
        
        right_elbow = self.calculate_joint_angle(
            landmarks[mp_pose.RIGHT_SHOULDER.value],
            landmarks[mp_pose.RIGHT_ELBOW.value],
            landmarks[mp_pose.RIGHT_WRIST.value]
        )
        
        # Knee angles
        left_knee = self.calculate_joint_angle(
            landmarks[mp_pose.LEFT_HIP.value],
            landmarks[mp_pose.LEFT_KNEE.value],
            landmarks[mp_pose.LEFT_ANKLE.value]
        )
        
        right_knee = self.calculate_joint_angle(
            landmarks[mp_pose.RIGHT_HIP.value],
            landmarks[mp_pose.RIGHT_KNEE.value],
            landmarks[mp_pose.RIGHT_ANKLE.value]
        )
        
        # Hip angle
        hip_angle = self.calculate_joint_angle(
            landmarks[mp_pose.LEFT_SHOULDER.value],
            landmarks[mp_pose.LEFT_HIP.value],
            landmarks[mp_pose.LEFT_KNEE.value]
        )
        
        # Return dictionary of angles
        return {
            'elbow': (left_elbow + right_elbow) / 2,
            'knee': (left_knee + right_knee) / 2,
            'hip': hip_angle
        }

class PoseFeatureExtractor:
    """Extracts and normalizes features from pose landmarks for machine learning."""
    
    def __init__(self):
        self.keypoint_map = KEYPOINT_MAP
        self.num_keypoints = len(self.keypoint_map)
    
    def extract_features(self, landmarks):
        """Convert landmarks to feature vector for model input."""
        if landmarks is None:
            return None
            
        # Initialize feature arrays
        x_coords = np.zeros(self.num_keypoints)
        y_coords = np.zeros(self.num_keypoints)
        
        # Extract coordinates from landmarks
        for mp_idx, custom_idx in self.keypoint_map.items():
            x_coords[custom_idx] = landmarks[mp_idx].x
            y_coords[custom_idx] = landmarks[mp_idx].y
        
        # Find valid coordinates (non-zero)
        valid_indices = np.where((x_coords != 0) & (y_coords != 0))[0]
        
        if len(valid_indices) > 0:
            # Find body center using valid keypoints
            center_x = np.mean(x_coords[valid_indices])
            center_y = np.mean(y_coords[valid_indices])
            
            # Center the coordinates
            x_centered = x_coords - center_x
            y_centered = y_coords - center_y
            
            # Use torso height for scaling
            shoulder_idx = 5  # Left shoulder
            hip_idx = 11      # Left hip
            
            torso_height = np.sqrt(
                (x_coords[shoulder_idx] - x_coords[hip_idx])**2 + 
                (y_coords[shoulder_idx] - y_coords[hip_idx])**2
            )
            
            # Avoid division by zero
            if torso_height < 1e-6:
                torso_height = 1.0
                
            # Scale coordinates by torso height
            x_normalized = x_centered / torso_height
            y_normalized = y_centered / torso_height
            
            # Create feature vector by concatenating x and y coordinates
            feature_vector = np.concatenate([x_normalized, y_normalized])
            return feature_vector.reshape(1, -1)
        
        return None

class ActivityClassifier:
    """ML model for exercise classification from normalized pose features."""
    
    def __init__(self, model_path):
        # Initialize TFLite interpreter
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        
        # Get model details
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        # Define exercise classes
        self.activity_types = ["squats", "pushups"]
    
    def predict_activity(self, features):
        """Classify the exercise type from pose features."""
        if features is None:
            return "unknown"
        
        # Set input tensor
        self.interpreter.set_tensor(self.input_details[0]['index'], features.astype(np.float32))
        
        # Run inference
        self.interpreter.invoke()
        
        # Get output
        predictions = self.interpreter.get_tensor(self.output_details[0]['index'])
        
        # Get predicted class
        class_idx = np.argmax(predictions[0])
        return self.activity_types[class_idx]

class RepetitionCounter:
    """Tracks and counts exercise repetitions based on movement patterns."""
    
    def __init__(self):
        self.count = 0
        self.phase = 0  # 0: going down, 1: going up
        self.form_quality = 0  # 0: incorrect form, 1: correct form
        self.coaching_cue = "Fix form"
        
        # Thresholds for peak detection with hysteresis
        self.down_threshold = 10  # Lower position threshold
        self.up_threshold = 90    # Upper position threshold
        
        # Smoothing for stability
        self.position_buffer = deque(maxlen=5)
    
    def update_position_buffer(self, position):
        """Update position buffer and get smoothed position."""
        self.position_buffer.append(position)
        return sum(self.position_buffer) / len(self.position_buffer)
    
    def track_pushup(self, angles, progress):
        """Track pushup repetitions and form."""
        elbow_angle = angles['elbow']
        hip_angle = angles['hip']
        
        # Form validation for pushups - straight back
        if elbow_angle > 160 and hip_angle > 160:
            self.form_quality = 1
        
        if self.form_quality == 1:
            # Smooth the progress value
            smoothed_progress = self.update_position_buffer(progress)
            
            # Bottom position (arms bent)
            if smoothed_progress < self.down_threshold:
                if elbow_angle <= 90 and hip_angle > 160:
                    self.coaching_cue = "Push up"
                    if self.phase == 0:
                        self.count += 0.5
                        self.phase = 1
                else:
                    self.coaching_cue = "Keep back straight"
            
            # Top position (arms extended)
            if smoothed_progress > self.up_threshold:
                if elbow_angle > 160 and hip_angle > 160:
                    self.coaching_cue = "Lower down"
                    if self.phase == 1:
                        self.count += 0.5
                        self.phase = 0
                else:
                    self.coaching_cue = "Extend arms fully"
    
    def track_squat(self, angles, progress):
        """Track squat repetitions and form."""
        knee_angle = angles['knee']
        hip_angle = angles['hip']
        
        # Form validation for squats - upright posture
        if knee_angle > 170:
            self.form_quality = 1
        
        if self.form_quality == 1:
            # Smooth the progress value
            smoothed_progress = self.update_position_buffer(progress)
            
            # Bottom position (knees bent)
            if smoothed_progress < self.down_threshold:
                if knee_angle <= 100:
                    self.coaching_cue = "Stand up"
                    if self.phase == 0:
                        self.count += 0.5
                        self.phase = 1
                else:
                    self.coaching_cue = "Squat deeper"
            
            # Top position (knees extended)
            if smoothed_progress > self.up_threshold:
                if knee_angle > 170:
                    self.coaching_cue = "Bend knees"
                    if self.phase == 1:
                        self.count += 0.5
                        self.phase = 0
                else:
                    self.coaching_cue = "Stand fully upright"
    
    def get_metrics(self):
        """Get current exercise metrics."""
        return {
            'count': self.count,
            'form': self.form_quality,
            'phase': self.phase,
            'feedback': self.coaching_cue
        }

class ExerciseVisualizer:
    """Handles visualization of exercise feedback and metrics."""
    
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.colors = {
            'progress_bar': (0, 255, 0),
            'counter_bg': (0, 255, 0),
            'counter_text': (255, 0, 0),
            'feedback_bg': (255, 255, 255),
            'feedback_text': (0, 255, 0),
            'exercise_bg': (255, 255, 255),
            'exercise_text': (0, 0, 255),
            'instruction_text': (255, 255, 255)
        }
    
    def draw_progress_indicator(self, frame, progress, form_quality):
        """Draw exercise progress bar."""
        # Set bar dimensions and position
        bar_x = int(self.width - 40)
        bar_width = 20
        bar_top = 30
        bar_bottom = self.height - 100
        
        # Draw outline
        cv2.rectangle(frame, 
                     (bar_x, bar_top), 
                     (bar_x + bar_width, bar_bottom), 
                     self.colors['progress_bar'], 3)
        
        # Draw fill based on progress when form is correct
        if form_quality == 1:
            # Map progress (0-100) to y-coordinate
            progress_y = np.interp(progress, (0, 100), (bar_bottom, bar_top))
            
            # Fill bar
            cv2.rectangle(frame, 
                         (bar_x, int(progress_y)), 
                         (bar_x + bar_width, bar_bottom), 
                         self.colors['progress_bar'], cv2.FILLED)
            
            # Show percentage
            percentage_x = bar_x - 15
            percentage_y = self.height - 50
            cv2.putText(frame, f'{int(progress)}%', 
                       (percentage_x, percentage_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, 
                       self.colors['counter_text'], 2)
    
    def draw_rep_counter(self, frame, count):
        """Draw repetition counter."""
        # Set counter dimensions and position
        counter_width = 100
        counter_height = 100
        counter_x = 0
        counter_y = self.height - counter_height
        
        # Draw counter background
        cv2.rectangle(frame, 
                     (counter_x, counter_y), 
                     (counter_x + counter_width, self.height), 
                     self.colors['counter_bg'], cv2.FILLED)
        
        # Draw count number
        count_x = counter_x + 25
        count_y = counter_y + 75
        cv2.putText(frame, str(int(count)), 
                   (count_x, count_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 2.0, 
                   self.colors['counter_text'], 5)
    
    def draw_feedback(self, frame, feedback):
        """Draw form feedback text."""
        # Set feedback dimensions and position
        feedback_width = 140
        feedback_x = self.width - feedback_width
        feedback_y = 0
        feedback_height = 40
        
        # Draw feedback background
        cv2.rectangle(frame, 
                     (feedback_x, feedback_y), 
                     (self.width, feedback_y + feedback_height), 
                     self.colors['feedback_bg'], cv2.FILLED)
        
        # Draw feedback text
        text_x = feedback_x + 5
        text_y = feedback_y + 30
        cv2.putText(frame, feedback, 
                   (text_x, text_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, 
                   self.colors['feedback_text'], 2)
    
    def draw_exercise_type(self, frame, exercise):
        """Draw exercise type label."""
        # Set exercise label dimensions and position
        label_x = 0
        label_y = 0
        label_width = 225
        label_height = 40
        
        # Draw exercise label background
        cv2.rectangle(frame, 
                     (label_x, label_y), 
                     (label_x + label_width, label_y + label_height), 
                     self.colors['exercise_bg'], cv2.FILLED)
        
        # Draw exercise label text
        text_x = label_x + 5
        text_y = label_y + 30
        cv2.putText(frame, f"Exercise: {exercise}", 
                   (text_x, text_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, 
                   self.colors['exercise_text'], 2)
    
    def draw_instructions(self, frame):
        """Draw user instructions."""
        instruction_x = 10
        instruction_y = self.height - 10
        cv2.putText(frame, "Press ESC or 'q' to exit", 
                   (instruction_x, instruction_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
                   self.colors['instruction_text'], 1)

def parse_command_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Exercise Analysis System')
    
    parser.add_argument('--video', type=str, default=None,
                      help='Path to video file for analysis')
    parser.add_argument('--image', type=str, default=None,
                      help='Path to image file for analysis')
    parser.add_argument('--webcam', action='store_true', default=True,
                      help='Use webcam as input source (default)')
    parser.add_argument('--width', type=int, default=640,
                      help='Display width (default: 640)')
    parser.add_argument('--height', type=int, default=480,
                      help='Display height (default: 480)')
    parser.add_argument('--index-mode', choices=['original', 'custom'], default='original',
                      help='Joint index display mode: original MediaPipe indices or custom mapping (default: original)')
    
    return parser.parse_args()

def process_image(image_path, width, height, show_original_indices):
    """Process a single image file."""
    # Initialize components
    pose_estimator = PoseEstimation()
    biomechanics_analyzer = BiomechanicsAnalyzer()
    feature_extractor = PoseFeatureExtractor()
    visualizer = ExerciseVisualizer(width, height)
    
    # Try to load the ML model
    model_path = "models/exercise_classifier.tflite"
    try:
        classifier = ActivityClassifier(model_path)
        model_available = True
        print(f"ML model loaded successfully from {model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Running in manual mode - no automatic exercise classification")
        model_available = False
    
    # Read the image
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"Error: Could not read image file {image_path}")
        return
    
    # Resize while preserving aspect ratio
    h, w = frame.shape[:2]
    aspect = w / h
    if width / height > aspect:
        new_w = int(height * aspect)
        new_h = height
    else:
        new_w = width
        new_h = int(width / aspect)
    
    frame = cv2.resize(frame, (new_w, new_h))
    
    # Create a black canvas of the target size and place the resized image in the center
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    y_offset = (height - new_h) // 2
    x_offset = (width - new_w) // 2
    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = frame
    frame = canvas
    
    # Detect pose
    pose_results = pose_estimator.detect_pose(frame)
    
    if pose_results.pose_landmarks:
        # Draw landmarks
        pose_estimator.draw_landmarks(frame, pose_results.pose_landmarks)
        
        # Draw landmark numbers
        pose_estimator.draw_landmark_numbers(frame, pose_results.pose_landmarks, show_original_indices)
        
        # Get landmarks for further processing
        landmarks = pose_results.pose_landmarks.landmark
        
        # Extract joint angles
        angles = biomechanics_analyzer.analyze_body_angles(landmarks, mp.solutions.pose.PoseLandmark)
        
        # Extract features for classification
        features = feature_extractor.extract_features(landmarks)
        
        # Classify activity if model is available
        activity_type = "unknown"
        if model_available and features is not None:
            try:
                activity_type = classifier.predict_activity(features)
            except Exception as e:
                print(f"Classification error: {e}")
        
        # Draw exercise type label
        visualizer.draw_exercise_type(frame, activity_type)
    
    # Display frame
    index_mode = "MediaPipe Original" if show_original_indices else "Custom Mapping"
    cv2.imshow(f'Exercise Analysis - Image (Index Mode: {index_mode})', frame)
    print("Press any key to close the window...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main():
    # Get command line arguments
    args = parse_command_arguments()
    
    # Determine index mode
    show_original_indices = args.index_mode == 'original'
    index_mode_text = "MediaPipe Original" if show_original_indices else "Custom Mapping"
    print(f"Using index mode: {index_mode_text}")
    
    # Process image if specified
    if args.image is not None:
        print(f"Processing image file: {args.image}")
        process_image(args.image, args.width, args.height, show_original_indices)
        return
    
    # Initialize video source
    if args.video is not None:
        print(f"Loading video file: {args.video}")
        video_source = cv2.VideoCapture(args.video)
        if not video_source.isOpened():
            print(f"Error: Could not open video file {args.video}")
            return
    else:
        print("Initializing webcam...")
        video_source = cv2.VideoCapture(0)
        if not video_source.isOpened():
            print("Error: Could not access webcam")
            return
    
    # Initialize components
    pose_estimator = PoseEstimation()
    biomechanics_analyzer = BiomechanicsAnalyzer()
    feature_extractor = PoseFeatureExtractor()
    rep_counter = RepetitionCounter()
    visualizer = ExerciseVisualizer(args.width, args.height)
    
    # Initialize activity prediction smoothing
    activity_buffer = deque(maxlen=10)
    
    # Try to load the ML model
    model_path = "models/exercise_classifier.tflite"
    try:
        classifier = ActivityClassifier(model_path)
        model_available = True
        print(f"ML model loaded successfully from {model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Running in manual mode - no automatic exercise classification")
        model_available = False
    
    # Main processing loop
    while video_source.isOpened():
        success, frame = video_source.read()
        if not success:
            print("Video ended or frame capture failed")
            break
        
        # Resize while preserving aspect ratio
        h, w = frame.shape[:2]
        aspect = w / h
        if args.width / args.height > aspect:
            new_w = int(args.height * aspect)
            new_h = args.height
        else:
            new_w = args.width
            new_h = int(args.width / aspect)
        
        frame = cv2.resize(frame, (new_w, new_h))
        
        # Create a black canvas of the target size and place the resized image in the center
        canvas = np.zeros((args.height, args.width, 3), dtype=np.uint8)
        y_offset = (args.height - new_h) // 2
        x_offset = (args.width - new_w) // 2
        canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = frame
        frame = canvas
        
        # Mirror webcam feed
        if args.video is None:
            frame = cv2.flip(frame, 1)
        
        # Detect pose
        pose_results = pose_estimator.detect_pose(frame)
        
        if pose_results.pose_landmarks:
            # Draw landmarks
            pose_estimator.draw_landmarks(frame, pose_results.pose_landmarks)
            
            # Draw landmark numbers
            pose_estimator.draw_landmark_numbers(frame, pose_results.pose_landmarks, show_original_indices)
            
            # Get landmarks for further processing
            landmarks = pose_results.pose_landmarks.landmark
            
            # Extract joint angles
            angles = biomechanics_analyzer.analyze_body_angles(landmarks, mp.solutions.pose.PoseLandmark)
            
            # Extract features for classification
            features = feature_extractor.extract_features(landmarks)
            
            # Classify activity if model is available
            activity_type = "unknown"
            if model_available and features is not None:
                try:
                    activity_type = classifier.predict_activity(features)
                    activity_buffer.append(activity_type)
                    
                    # Use majority voting for stability
                    if len(activity_buffer) > 0:
                        activity_type = max(set(activity_buffer), key=activity_buffer.count)
                except Exception as e:
                    print(f"Classification error: {e}")
                    model_available = False
            
            # Exercise specific processing
            if activity_type == "pushups":
                # Calculate progress
                progress = np.interp(angles['elbow'], (90, 160), (0, 100))
                
                # Update rep counter
                rep_counter.track_pushup(angles, progress)
                
            elif activity_type == "squats":
                # Calculate progress
                progress = np.interp(angles['knee'], (90, 170), (0, 100))
                
                # Update rep counter
                rep_counter.track_squat(angles, progress)
                
            else:
                # Default values for unknown activity
                progress = 0
            
            # Get current metrics
            metrics = rep_counter.get_metrics()
            
            # Update UI
            visualizer.draw_progress_indicator(frame, progress, metrics['form'])
            visualizer.draw_rep_counter(frame, metrics['count'])
            visualizer.draw_feedback(frame, metrics['feedback'])
            visualizer.draw_exercise_type(frame, activity_type)
        
        # Draw instructions
        visualizer.draw_instructions(frame)
        
        # Add instruction for toggling index mode
        instruction_x = 10
        instruction_y = args.height - 30
        cv2.putText(frame, "Press 'm' to toggle index mode", 
                   (instruction_x, instruction_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
                   (255, 255, 255), 1)
        
        # Display frame with current index mode
        index_mode_text = "MediaPipe Original" if show_original_indices else "Custom Mapping"
        cv2.imshow(f'Exercise Analysis System (Index Mode: {index_mode_text})', frame)
        
        # Check for exit keys
        key = cv2.waitKey(10) & 0xFF
        if key == ord('q') or key == 27:  # ESC key
            print("Exiting program...")
            break
        elif key == ord('m'):  # Toggle index mode
            show_original_indices = not show_original_indices
            index_mode_text = "MediaPipe Original" if show_original_indices else "Custom Mapping"
            print(f"Switched to index mode: {index_mode_text}")
    
    # Clean up
    video_source.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 