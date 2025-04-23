import numpy as np
import tensorflow as tf
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from sklearn.metrics import confusion_matrix, classification_report

# Define exercise categories for classification
EXERCISE_CATEGORIES = ['squats', 'pushups']

class DataProcessor:
    """Handles loading and processing training data for the exercise classifier"""
    
    @staticmethod
    def load_pose_data(file_path):
        """
        Load skeletal pose data from numpy file
        
        Args:
            file_path: Path to the .npy file
            
        Returns:
            Numpy array containing pose keypoints
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Missing pose data file: {file_path}")
            
        return np.load(file_path)
    
    @staticmethod
    def load_activity_annotations(file_path):
        """
        Load activity annotation data from CSV file
        
        Args:
            file_path: Path to the annotation CSV file
            
        Returns:
            List of annotations in format [start_frame, end_frame, repetition, activity_name]
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Missing annotation file: {file_path}")
        
        annotations = []
        with open(file_path, 'r') as f:
            for line in f:
                if line.strip() and not line.startswith('#'):
                    parts = line.strip().split(',')
                    if len(parts) >= 4:
                        annotations.append([
                            int(parts[0]),   # Start frame
                            int(parts[1]),   # End frame
                            int(parts[2]),   # Repetition count
                            parts[3]         # Activity name
                        ])
        
        return annotations
    
    @staticmethod
    def create_dataset(workout_ids, data_path):
        """
        Create a dataset from workout files
        
        Args:
            workout_ids: List of workout IDs to process
            data_path: Base path to data directory
            
        Returns:
            DataFrame containing features and labels
        """
        from tqdm import tqdm
        
        # Create category to index mapping
        category_to_idx = {category: idx for idx, category in enumerate(EXERCISE_CATEGORIES)}
        
        # Initialize empty dataframe for collected data
        combined_data = pd.DataFrame()
        
        for workout_id in tqdm(workout_ids, desc="Processing workout data"):
            data_dir = os.path.join(data_path, workout_id)
            
            # Load skeletal pose data
            pose_data = DataProcessor.load_pose_data(os.path.join(data_dir, f"{workout_id}_pose_2d.npy"))
            
            # Reshape pose data (frame, joint, xy) to (frame, features)
            pose_features = pose_data.transpose(1, 0, 2).reshape(-1, 38)
            
            # Load activity annotations
            annotations = DataProcessor.load_activity_annotations(os.path.join(data_dir, f"{workout_id}_labels.csv"))
            
            # Extract frames with target exercises
            labeled_frames = []
            for annotation in annotations:
                start_frame, end_frame, rep_count, activity = annotation
                if activity in EXERCISE_CATEGORIES:
                    for frame in range(start_frame, end_frame+1):
                        labeled_frames.append([frame, category_to_idx[activity]])
            
            if len(labeled_frames) > 0:  # Only process files with relevant activities
                # Convert to DataFrame
                label_df = pd.DataFrame(labeled_frames, columns=['frame', 'activity_idx'])
                pose_df = pd.DataFrame(pose_features)
                
                # First column is frame index, drop the neck/mid-shoulders placeholder (column 19)
                pose_df = pose_df.rename(columns={0: 'frame'}).drop(19, axis=1)
                
                # Merge pose features with labels on frame number
                merged_df = pose_df.merge(label_df, how='inner', on='frame')
                
                # Add to combined dataset
                combined_data = pd.concat([combined_data, merged_df])
        
        return combined_data
    
    @staticmethod
    def normalize_poses(features):
        """
        Normalize pose features by centering and scaling
        
        Args:
            features: Pose features array
            
        Returns:
            Normalized pose features
        """
        # Split into x and y coordinates
        num_joints = 18  # Number of body joints
        x_coords = features[:, :num_joints]
        y_coords = features[:, num_joints:]
        
        # Find body center for each pose
        x_center = np.median(x_coords, axis=1, keepdims=True)
        y_center = np.median(y_coords, axis=1, keepdims=True)
        
        # Center coordinates to body center
        x_centered = x_coords - x_center
        y_centered = y_coords - y_center
        
        # Scale by torso height (distance between neck and hip)
        neck_idx = 1     # Index for neck joint
        hip_idx = 8      # Index for hip joint
        
        torso_height = np.sqrt(
            (x_coords[:, neck_idx] - x_coords[:, hip_idx])**2 + 
            (y_coords[:, neck_idx] - y_coords[:, hip_idx])**2
        ).reshape(-1, 1)
        
        # Avoid division by zero
        torso_height = np.maximum(torso_height, 1e-6)
        
        # Scale coordinates 
        x_normalized = x_centered / torso_height
        y_normalized = y_centered / torso_height
        
        # Combine back into feature array
        normalized_features = np.hstack((x_normalized, y_normalized))
        
        return normalized_features


class ExerciseClassifier:
    """Neural network model for exercise classification from pose data"""
    
    def __init__(self, input_dim, num_classes=len(EXERCISE_CATEGORIES)):
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.model = self._build_model()
        
    def _build_model(self):
        """Create neural network architecture"""
        model = Sequential([
            # Input layer
            Dense(256, activation='relu', input_shape=(self.input_dim,)),
            BatchNormalization(),
            Dropout(0.4),
            
            # Hidden layers
            Dense(128, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            
            Dense(64, activation='relu'),
            BatchNormalization(),
            Dropout(0.2),
            
            # Output layer
            Dense(self.num_classes, activation='softmax')
        ])
        
        # Compile model with optimizer and loss function
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train(self, X_train, y_train, X_val, y_val, batch_size=32, epochs=50):
        """Train the model with early stopping"""
        
        # Calculate class weights to handle imbalanced data
        class_weights = class_weight.compute_class_weight(
            'balanced',
            classes=np.unique(y_train),
            y=y_train
        )
        class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}
        
        # Set up callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=7,
                min_lr=1e-6,
                verbose=1
            )
        ]
        
        # Train model
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            class_weight=class_weight_dict,
            verbose=1
        )
        
        return history
    
    def evaluate(self, X_test, y_test):
        """Evaluate model on test data"""
        return self.model.evaluate(X_test, y_test)
    
    def save(self, keras_path, tflite_path):
        """Save model in both Keras and TFLite formats"""
        # Save Keras model
        self.model.save(keras_path)
        
        # Convert to TFLite format
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        tflite_model = converter.convert()
        
        # Save TFLite model
        with open(tflite_path, 'wb') as f:
            f.write(tflite_model)


def visualize_training_metrics(history):
    """
    Visualize training and validation metrics
    
    Args:
        history: Training history from model.fit()
    """
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot accuracy
    ax1.plot(history.history['accuracy'], label='Training')
    ax1.plot(history.history['val_accuracy'], label='Validation')
    ax1.set_title('Model Accuracy', fontsize=14)
    ax1.set_ylabel('Accuracy', fontsize=12)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.legend(loc='lower right')
    ax1.grid(True, linestyle='--', alpha=0.6)
    
    # Plot loss
    ax2.plot(history.history['loss'], label='Training')
    ax2.plot(history.history['val_loss'], label='Validation')
    ax2.set_title('Model Loss', fontsize=14)
    ax2.set_ylabel('Loss', fontsize=12)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.legend(loc='upper right')
    ax2.grid(True, linestyle='--', alpha=0.6)
    
    # Add overall title and adjust layout
    plt.suptitle('Exercise Classification Training Metrics', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save figure
    plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
    plt.show()


def visualize_class_distribution(y_train, y_val, y_test):
    """
    Visualize the distribution of classes in training, validation and test sets
    
    Args:
        y_train: Training labels
        y_val: Validation labels
        y_test: Test labels
    """
    # Count class occurrences
    train_counts = pd.Series(y_train).value_counts().sort_index()
    val_counts = pd.Series(y_val).value_counts().sort_index()
    test_counts = pd.Series(y_test).value_counts().sort_index()
    
    # Create DataFrame
    df = pd.DataFrame({
        'Training': train_counts,
        'Validation': val_counts,
        'Test': test_counts
    })
    
    # Add class names
    df.index = [EXERCISE_CATEGORIES[i] for i in df.index]
    
    # Calculate percentages
    total_train = y_train.shape[0]
    total_val = y_val.shape[0]
    total_test = y_test.shape[0]
    
    df_percent = pd.DataFrame({
        'Training': (train_counts / total_train * 100).round(1),
        'Validation': (val_counts / total_val * 100).round(1),
        'Test': (test_counts / total_test * 100).round(1)
    })
    df_percent.index = df.index
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
    
    # Plot counts
    df.plot(kind='bar', ax=ax1)
    ax1.set_title('Class Distribution (Counts)', fontsize=14)
    ax1.set_ylabel('Number of Samples', fontsize=12)
    ax1.set_xlabel('Exercise Class', fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.6, axis='y')
    
    # Add count labels
    for container in ax1.containers:
        ax1.bar_label(container, fontsize=10)
    
    # Plot percentages
    df_percent.plot(kind='bar', ax=ax2)
    ax2.set_title('Class Distribution (Percentages)', fontsize=14)
    ax2.set_ylabel('Percentage (%)', fontsize=12)
    ax2.set_xlabel('Exercise Class', fontsize=12)
    ax2.grid(True, linestyle='--', alpha=0.6, axis='y')
    
    # Add percentage labels
    for container in ax2.containers:
        ax2.bar_label(container, fmt='%.1f%%', fontsize=10)
    
    plt.suptitle('Dataset Class Distribution', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig('class_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print tabular distribution
    print("\nClass Distribution Summary:")
    print("--------------------------")
    print(df)
    print("\nPercentage Distribution:")
    print("--------------------------")
    print(df_percent)


def visualize_normalization_effect(X_original, X_normalized, num_samples=5):
    """
    Visualize the effect of pose normalization on a few sample poses
    
    Args:
        X_original: Original pose features
        X_normalized: Normalized pose features
        num_samples: Number of random samples to visualize
    """
    # Select random samples
    total_samples = X_original.shape[0]
    sample_indices = np.random.choice(total_samples, size=num_samples, replace=False)
    
    # Number of joints
    num_joints = 18
    
    # Create a figure with rows for samples and columns for original vs normalized
    fig, axes = plt.subplots(num_samples, 2, figsize=(14, 4 * num_samples))
    
    for i, idx in enumerate(sample_indices):
        # Original pose
        orig_x = X_original[idx, :num_joints]
        orig_y = X_original[idx, num_joints:]
        
        # Normalized pose
        norm_x = X_normalized[idx, :num_joints]
        norm_y = X_normalized[idx, num_joints:]
        
        # Plot original pose
        axes[i, 0].scatter(orig_x, orig_y, s=50, c=range(num_joints), cmap='viridis')
        axes[i, 0].set_title(f'Original Pose (Sample {idx})')
        axes[i, 0].set_xlabel('X Coordinate')
        axes[i, 0].set_ylabel('Y Coordinate')
        axes[i, 0].grid(True, linestyle='--', alpha=0.6)
        axes[i, 0].axis('equal')
        
        # Plot normalized pose
        axes[i, 1].scatter(norm_x, norm_y, s=50, c=range(num_joints), cmap='viridis')
        axes[i, 1].set_title(f'Normalized Pose (Sample {idx})')
        axes[i, 1].set_xlabel('X Coordinate')
        axes[i, 1].set_ylabel('Y Coordinate')
        axes[i, 1].grid(True, linestyle='--', alpha=0.6)
        axes[i, 1].axis('equal')
    
    plt.tight_layout()
    plt.savefig('normalization_effect.png', dpi=300, bbox_inches='tight')
    plt.show()


def evaluate_classifier_performance(model, X_test, y_test):
    """
    Evaluate classifier performance and generate detailed metrics and visualizations
    
    Args:
        model: Trained classifier model
        X_test: Test features
        y_test: Test labels
        
    Returns:
        Dict containing evaluation metrics
    """
    # Get predictions
    y_pred_proba = model.predict(X_test)
    y_pred = np.argmax(y_pred_proba, axis=1)
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Create class names
    class_names = EXERCISE_CATEGORIES
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Calculate and print classification report
    report = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    
    print("\nClassification Report:")
    print("--------------------------")
    print(pd.DataFrame(report).transpose().round(3))
    
    # Plot precision, recall, and f1-score
    plt.figure(figsize=(12, 6))
    metrics_df = report_df.iloc[:-3]  # Exclude averages
    metrics_df = metrics_df[['precision', 'recall', 'f1-score']]
    
    metrics_df.plot(kind='bar', rot=0)
    plt.title('Classification Metrics by Class')
    plt.ylabel('Score')
    plt.xlabel('Class')
    plt.ylim(0, 1.0)
    plt.grid(True, linestyle='--', alpha=0.6, axis='y')
    plt.tight_layout()
    plt.savefig('classification_metrics.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Return evaluation metrics
    return {
        'confusion_matrix': cm,
        'classification_report': report
    }


def main():
    """Main function to run the model training pipeline"""
    # Define data paths
    data_path = 'data/mm-fit/'
    
    # Create output directory
    os.makedirs('models', exist_ok=True)
    os.makedirs('visualizations', exist_ok=True)
    
    # Define train/validation/test workout IDs
    train_workout_ids = ['w01', 'w02', 'w03', 'w04', 'w06', 'w07', 'w08', 'w16', 
                        'w17', 'w18', 'w14', 'w15', 'w19', 'w20', 'w12', 'w13']
    
    test_workout_ids = ['w05', 'w09', 'w10', 'w11']
    
    # Create datasets
    print("Loading and processing training data...")
    train_df = DataProcessor.create_dataset(train_workout_ids, data_path)
    
    print("Loading and processing test data...")
    test_df = DataProcessor.create_dataset(test_workout_ids, data_path)
    
    # Check if data was found
    if train_df.empty or test_df.empty:
        print("Error: Insufficient data for target exercises. Please check your dataset.")
        return
    
    print(f"Training samples: {train_df.shape[0]}")
    print(f"Test samples: {test_df.shape[0]}")
    
    # Extract features and labels
    X_train = train_df.iloc[:, 1:-1].values.astype(np.float32)  # Skip frame column and activity column
    y_train = train_df.iloc[:, -1].values.astype(np.int32)      # Activity column
    
    X_test = test_df.iloc[:, 1:-1].values.astype(np.float32)
    y_test = test_df.iloc[:, -1].values.astype(np.int32)
    
    # Normalize features
    print("Normalizing pose features...")
    X_train_norm = DataProcessor.normalize_poses(X_train)
    X_test_norm = DataProcessor.normalize_poses(X_test)
    
    # Visualize normalization effect on a few samples
    print("Generating normalization visualization...")
    visualize_normalization_effect(X_train, X_train_norm, num_samples=3)
    
    # Split training data to create validation set
    X_train_final, X_val, y_train_final, y_val = train_test_split(
        X_train_norm, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    
    # Visualize class distribution
    print("Generating class distribution visualization...")
    visualize_class_distribution(y_train_final, y_val, y_test)
    
    # Initialize and train model
    print("Building and training model...")
    classifier = ExerciseClassifier(input_dim=X_train_norm.shape[1])
    classifier.model.summary()
    
    # Train model
    history = classifier.train(
        X_train_final, y_train_final,
        X_val, y_val,
        batch_size=32,
        epochs=100  # Maximum epochs (early stopping will prevent overfitting)
    )
    
    # Evaluate model
    print("Evaluating model on test set...")
    loss, accuracy = classifier.evaluate(X_test_norm, y_test)
    print(f"Test accuracy: {accuracy:.4f}")
    print(f"Test loss: {loss:.4f}")
    
    # Visualize training metrics
    visualize_training_metrics(history)
    
    # Generate detailed performance evaluation
    print("Generating detailed performance evaluation...")
    evaluate_classifier_performance(classifier.model, X_test_norm, y_test)
    
    # Save model
    print("Saving model...")
    classifier.save('models/exercise_classifier.h5', 'models/exercise_classifier.tflite')
    
    print("Training pipeline completed successfully!")

if __name__ == '__main__':
    main() 