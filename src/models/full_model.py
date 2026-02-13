"""
Full End-to-End Hybrid Emotion Recognition Model
Combines zone-based CNNs, global CNN, and temporal LSTM
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
import numpy as np

class HybridEmotionModel:
    """
    Complete hybrid emotion recognition system
    Architecture: Zone CNNs + Global CNN → Feature Fusion → LSTM → Classification
    """
    
    def __init__(self, 
                 num_emotions=7,
                 sequence_length=16,
                 global_img_size=224,
                 zone_img_size=48,
                 lstm_units=[256, 128],
                 dropout_rate=0.5):
        """
        Initialize the hybrid model
        
        Args:
            num_emotions: Number of emotion classes (default 7: angry, disgust, fear, happy, sad, surprise, neutral)
            sequence_length: Number of frames in temporal sequence
            global_img_size: Size of global face image
            zone_img_size: Size of each zone image
            lstm_units: List of LSTM layer units
            dropout_rate: Dropout rate for regularization
        """
        self.num_emotions = num_emotions
        self.sequence_length = sequence_length
        self.global_img_size = global_img_size
        self.zone_img_size = zone_img_size
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        
        # Emotion labels
        self.emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
        
        # Build the complete model
        self.model = None
        self.feature_extractor = None
        
    def build_zone_cnn(self, name_prefix):
        """
        Build a lightweight CNN for zone-specific feature extraction
        Input: 48x48 grayscale zone image
        Output: 128-dim feature vector
        """
        input_layer = layers.Input(shape=(self.zone_img_size, self.zone_img_size, 1), 
                                   name=f'{name_prefix}_input')
        
        # Conv Block 1
        x = layers.Conv2D(32, (3, 3), padding='same', name=f'{name_prefix}_conv1')(input_layer)
        x = layers.BatchNormalization(name=f'{name_prefix}_bn1')(x)
        x = layers.Activation('relu', name=f'{name_prefix}_relu1')(x)
        x = layers.MaxPooling2D((2, 2), name=f'{name_prefix}_pool1')(x)
        
        # Conv Block 2
        x = layers.Conv2D(64, (3, 3), padding='same', name=f'{name_prefix}_conv2')(x)
        x = layers.BatchNormalization(name=f'{name_prefix}_bn2')(x)
        x = layers.Activation('relu', name=f'{name_prefix}_relu2')(x)
        x = layers.MaxPooling2D((2, 2), name=f'{name_prefix}_pool2')(x)
        
        # Conv Block 3
        x = layers.Conv2D(128, (3, 3), padding='same', name=f'{name_prefix}_conv3')(x)
        x = layers.BatchNormalization(name=f'{name_prefix}_bn3')(x)
        x = layers.Activation('relu', name=f'{name_prefix}_relu3')(x)
        x = layers.MaxPooling2D((2, 2), name=f'{name_prefix}_pool3')(x)
        
        # Flatten and Dense
        x = layers.Flatten(name=f'{name_prefix}_flatten')(x)
        x = layers.Dense(128, activation='relu', name=f'{name_prefix}_dense')(x)
        x = layers.Dropout(0.3, name=f'{name_prefix}_dropout')(x)
        
        return Model(inputs=input_layer, outputs=x, name=f'{name_prefix}_cnn')
    
    def build_global_cnn(self):
        """
        Build CNN for global face feature extraction
        Input: 224x224 grayscale face image
        Output: 512-dim feature vector
        """
        input_layer = layers.Input(shape=(self.global_img_size, self.global_img_size, 1), 
                                   name='global_input')
        
        # Conv Block 1
        x = layers.Conv2D(64, (3, 3), padding='same', name='global_conv1')(input_layer)
        x = layers.BatchNormalization(name='global_bn1')(x)
        x = layers.Activation('relu', name='global_relu1')(x)
        x = layers.MaxPooling2D((2, 2), name='global_pool1')(x)
        
        # Conv Block 2
        x = layers.Conv2D(128, (3, 3), padding='same', name='global_conv2')(x)
        x = layers.BatchNormalization(name='global_bn2')(x)
        x = layers.Activation('relu', name='global_relu2')(x)
        x = layers.MaxPooling2D((2, 2), name='global_pool2')(x)
        
        # Conv Block 3
        x = layers.Conv2D(256, (3, 3), padding='same', name='global_conv3')(x)
        x = layers.BatchNormalization(name='global_bn3')(x)
        x = layers.Activation('relu', name='global_relu3')(x)
        x = layers.MaxPooling2D((2, 2), name='global_pool3')(x)
        
        # Conv Block 4
        x = layers.Conv2D(512, (3, 3), padding='same', name='global_conv4')(x)
        x = layers.BatchNormalization(name='global_bn4')(x)
        x = layers.Activation('relu', name='global_relu4')(x)
        x = layers.MaxPooling2D((2, 2), name='global_pool4')(x)
        
        # Flatten and Dense
        x = layers.Flatten(name='global_flatten')(x)
        x = layers.Dense(512, activation='relu', name='global_dense')(x)
        x = layers.Dropout(self.dropout_rate, name='global_dropout')(x)
        
        return Model(inputs=input_layer, outputs=x, name='global_cnn')
    
    def build_feature_extractor(self):
        """
        Build the hybrid feature extraction network
        Combines global CNN + 5 zone CNNs
        """
        # Input layers for each zone and global face
        global_input = layers.Input(shape=(self.global_img_size, self.global_img_size, 1), 
                                   name='global_face')
        forehead_input = layers.Input(shape=(self.zone_img_size, self.zone_img_size, 1), 
                                     name='forehead_zone')
        left_eye_input = layers.Input(shape=(self.zone_img_size, self.zone_img_size, 1), 
                                     name='left_eye_zone')
        right_eye_input = layers.Input(shape=(self.zone_img_size, self.zone_img_size, 1), 
                                      name='right_eye_zone')
        nose_input = layers.Input(shape=(self.zone_img_size, self.zone_img_size, 1), 
                                 name='nose_zone')
        mouth_input = layers.Input(shape=(self.zone_img_size, self.zone_img_size, 1), 
                                  name='mouth_zone')
        
        # Build CNNs
        global_cnn = self.build_global_cnn()
        forehead_cnn = self.build_zone_cnn('forehead')
        left_eye_cnn = self.build_zone_cnn('left_eye')
        right_eye_cnn = self.build_zone_cnn('right_eye')
        nose_cnn = self.build_zone_cnn('nose')
        mouth_cnn = self.build_zone_cnn('mouth')
        
        # Extract features
        global_features = global_cnn(global_input)  # 512-dim
        forehead_features = forehead_cnn(forehead_input)  # 128-dim
        left_eye_features = left_eye_cnn(left_eye_input)  # 128-dim
        right_eye_features = right_eye_cnn(right_eye_input)  # 128-dim
        nose_features = nose_cnn(nose_input)  # 128-dim
        mouth_features = mouth_cnn(mouth_input)  # 128-dim
        
        # Concatenate all features: 512 + 5*128 = 1152-dim hybrid feature
        hybrid_features = layers.Concatenate(name='feature_fusion')([
            global_features,
            forehead_features,
            left_eye_features,
            right_eye_features,
            nose_features,
            mouth_features
        ])
        
        inputs = [global_input, forehead_input, left_eye_input, 
                 right_eye_input, nose_input, mouth_input]
        
        return Model(inputs=inputs, outputs=hybrid_features, name='hybrid_feature_extractor')
    
    def build_temporal_classifier(self, feature_dim=1152):
        """
        Build LSTM-based temporal classifier
        Input: Sequence of hybrid features (sequence_length × feature_dim)
        Output: Emotion probabilities
        """
        input_layer = layers.Input(shape=(self.sequence_length, feature_dim), 
                                  name='feature_sequence')
        
        # LSTM layers
        x = layers.LSTM(self.lstm_units[0], 
                       return_sequences=True, 
                       name='lstm_1')(input_layer)
        x = layers.Dropout(0.3, name='lstm_dropout_1')(x)
        
        x = layers.LSTM(self.lstm_units[1], 
                       return_sequences=False, 
                       name='lstm_2')(x)
        x = layers.Dropout(0.3, name='lstm_dropout_2')(x)
        
        # Classification head
        x = layers.Dense(256, activation='relu', name='fc_1')(x)
        x = layers.Dropout(self.dropout_rate, name='fc_dropout')(x)
        
        output = layers.Dense(self.num_emotions, activation='softmax', name='emotion_output')(x)
        
        return Model(inputs=input_layer, outputs=output, name='temporal_classifier')
    
    def build_complete_model(self):
        """
        Build the complete end-to-end model
        Architecture: Frame Inputs → Feature Extraction → Temporal Modeling → Classification
        """
        # Build feature extractor
        self.feature_extractor = self.build_feature_extractor()
        
        # Input: Sequences of frames for each zone
        # Shape: (batch, sequence_length, height, width, channels)
        global_seq = layers.Input(shape=(self.sequence_length, self.global_img_size, 
                                        self.global_img_size, 1), name='global_sequence')
        forehead_seq = layers.Input(shape=(self.sequence_length, self.zone_img_size, 
                                          self.zone_img_size, 1), name='forehead_sequence')
        left_eye_seq = layers.Input(shape=(self.sequence_length, self.zone_img_size, 
                                          self.zone_img_size, 1), name='left_eye_sequence')
        right_eye_seq = layers.Input(shape=(self.sequence_length, self.zone_img_size, 
                                           self.zone_img_size, 1), name='right_eye_sequence')
        nose_seq = layers.Input(shape=(self.sequence_length, self.zone_img_size, 
                                      self.zone_img_size, 1), name='nose_sequence')
        mouth_seq = layers.Input(shape=(self.sequence_length, self.zone_img_size, 
                                       self.zone_img_size, 1), name='mouth_sequence')
        
        # Apply feature extractor to each frame in sequence using TimeDistributed
        feature_sequences = layers.TimeDistributed(self.feature_extractor, 
                                                   name='time_distributed_features')(
            [global_seq, forehead_seq, left_eye_seq, right_eye_seq, nose_seq, mouth_seq]
        )
        
        # Build temporal classifier
        temporal_classifier = self.build_temporal_classifier()
        
        # Get emotion predictions
        emotion_output = temporal_classifier(feature_sequences)
        
        # Complete model
        self.model = Model(
            inputs=[global_seq, forehead_seq, left_eye_seq, right_eye_seq, nose_seq, mouth_seq],
            outputs=emotion_output,
            name='hybrid_emotion_recognition_model'
        )
        
        return self.model
    
    def compile_model(self, learning_rate=0.0001):
        """Compile the model with optimizer and loss"""
        if self.model is None:
            self.build_complete_model()
        
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        self.model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy', keras.metrics.TopKCategoricalAccuracy(k=3, name='top_3_accuracy')]
        )
        
        return self.model
    
    def summary(self):
        """Print model architecture summary"""
        if self.model is None:
            self.build_complete_model()
        
        print("\n" + "="*80)
        print("HYBRID EMOTION RECOGNITION MODEL SUMMARY")
        print("="*80)
        print(f"Total Parameters: {self.model.count_params():,}")
        print(f"Sequence Length: {self.sequence_length} frames")
        print(f"Number of Emotions: {self.num_emotions}")
        print(f"Emotion Labels: {', '.join(self.emotion_labels)}")
        print("="*80 + "\n")
        
        self.model.summary()
    
    def predict_emotion(self, frame_sequences):
        """
        Predict emotion from a sequence of frames
        
        Args:
            frame_sequences: Dict with keys ['global', 'forehead', 'left_eye', 
                            'right_eye', 'nose', 'mouth'], each containing 
                            numpy array of shape (sequence_length, height, width, 1)
        
        Returns:
            emotion_name: Predicted emotion label
            probabilities: Array of probabilities for each emotion
        """
        # Prepare input
        inputs = [
            np.expand_dims(frame_sequences['global'], axis=0),
            np.expand_dims(frame_sequences['forehead'], axis=0),
            np.expand_dims(frame_sequences['left_eye'], axis=0),
            np.expand_dims(frame_sequences['right_eye'], axis=0),
            np.expand_dims(frame_sequences['nose'], axis=0),
            np.expand_dims(frame_sequences['mouth'], axis=0)
        ]
        
        # Predict
        probabilities = self.model.predict(inputs, verbose=0)[0]
        emotion_idx = np.argmax(probabilities)
        emotion_name = self.emotion_labels[emotion_idx]
        
        return emotion_name, probabilities


if __name__ == "__main__":
    # Example usage
    print("Building Hybrid Emotion Recognition Model...")
    
    # Initialize model
    model = HybridEmotionModel(
        num_emotions=7,
        sequence_length=16,
        global_img_size=224,
        zone_img_size=48,
        lstm_units=[256, 128],
        dropout_rate=0.5
    )
    
    # Build and compile
    model.compile_model(learning_rate=0.0001)
    
    # Display summary
    model.summary()
    
    # Test with random data
    print("\nTesting with random input...")
    test_sequences = {
        'global': np.random.rand(16, 224, 224, 1).astype(np.float32),
        'forehead': np.random.rand(16, 48, 48, 1).astype(np.float32),
        'left_eye': np.random.rand(16, 48, 48, 1).astype(np.float32),
        'right_eye': np.random.rand(16, 48, 48, 1).astype(np.float32),
        'nose': np.random.rand(16, 48, 48, 1).astype(np.float32),
        'mouth': np.random.rand(16, 48, 48, 1).astype(np.float32)
    }
    
    emotion, probs = model.predict_emotion(test_sequences)
    print(f"\nPredicted Emotion: {emotion}")
    print(f"Probabilities: {probs}")
    print("\nModel built successfully!")