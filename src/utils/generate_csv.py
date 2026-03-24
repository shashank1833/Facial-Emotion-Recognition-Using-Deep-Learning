import os
import pandas as pd

# Define paths relative to the project root
raf_train_path = "data/Raf-DB/DATASET/train"
affectnet_train_path = "data/AffectNet/Train"
affectnet_test_path = "data/AffectNet/Test"

def normalize_label(label):
    # Mapping for standard labels and Raf-DB numerical labels
    mapping = {
        "angry": "Angry", "anger": "Angry", "6": "Angry",
        "disgust": "Disgust", "3": "Disgust",
        "fear": "Fear", "2": "Fear",
        "happy": "Happy", "happiness": "Happy", "4": "Happy",
        "sad": "Sad", "sadness": "Sad", "5": "Sad",
        "surprise": "Surprise", "1": "Surprise",
        "neutral": "Neutral", "7": "Neutral"
    }
    return mapping.get(label.lower(), None)

def create_df(path):
    data = []
    if not os.path.exists(path):
        print(f"Warning: Path {path} does not exist.")
        return pd.DataFrame(columns=["image_path", "label"])
        
    for label in os.listdir(path):
        label_path = os.path.join(path, label)
        if os.path.isdir(label_path):
            norm = normalize_label(label)
            if norm:
                print(f"Processing {label} -> {norm} in {path}")
                for img in os.listdir(label_path):
                    if img.lower().endswith(('.png', '.jpg', '.jpeg')):
                        # Use forward slashes for cross-platform compatibility (Windows and Colab)
                        full_path = os.path.join(label_path, img).replace('\\', '/')
                        data.append([full_path, norm])
    return pd.DataFrame(data, columns=["image_path", "label"])

print("Creating training dataframe...")
train_raf = create_df(raf_train_path)
train_affectnet = create_df(affectnet_train_path)

if not train_raf.empty or not train_affectnet.empty:
    train_df = pd.concat([train_raf, train_affectnet]).sample(frac=1).reset_index(drop=True)
    train_df.to_csv("train.csv", index=False)
    print(f"Saved {len(train_df)} samples to train.csv")
else:
    print("No training data found.")

print("\nCreating test dataframe...")
test_df = create_df(affectnet_test_path)
if not test_df.empty:
    test_df.to_csv("test.csv", index=False)
    print(f"Saved {len(test_df)} samples to test.csv")
else:
    print("No test data found.")

print("\nDone ✅")
