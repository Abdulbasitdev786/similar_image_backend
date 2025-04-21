import os
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt

# ====== CONFIG ======
DATASET_PATH = "C:\\Users\\sb\\Desktop\\ITU_PROJECT\\newData"       # Folder with images
QUERY_IMAGE = "C:\\Users\\sb\\Downloads\\My data\\17.jpg"       # Path to the query image

# Load the face cascade classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def detect_face(image_path):
    # Read the image
    img = cv2.imread(image_path)
    if img is None:
        return None
        
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    
    if len(faces) == 0:
        return None
        
    # Get the first face
    (x, y, w, h) = faces[0]
    
    # Extract and resize face
    face = img[y:y+h, x:x+w]
    face = cv2.resize(face, (224, 224))  # Resize to standard size
    
    return face

def create_embeddings(dataset_path):
    faces = []
    filenames = []
    print(f"Processing images in: {dataset_path}")
    
    try:
        files = os.listdir(dataset_path)
        print(f"Found {len(files)} files in directory")
    except Exception as e:
        print(f"❌ Error reading directory: {e}")
        return [], []

    image_files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    total_images = len(image_files)
    print(f"Found {total_images} image files")
    
    for i, img_file in enumerate(image_files, 1):
        img_path = os.path.join(dataset_path, img_file)
        if os.path.isfile(img_path):
            try:
                print(f"\nProcessing [{i}/{total_images}]: {img_path}")
                face = detect_face(img_path)
                
                if face is None:
                    print(f"❌ No face detected in: {img_path}")
                    continue
                    
                # For now, just flatten the image as a simple feature vector
                feature = face.flatten()
                faces.append(feature)
                filenames.append(img_path)
                print(f"✅ Successfully processed: {img_path}")
            except Exception as e:
                print(f"❌ Failed to process {img_path}: {str(e)}")
                import traceback
                print(traceback.format_exc())
    
    if not faces:
        print("⚠️ No faces were successfully detected!")
        return [], []
        
    print(f"\n✅ Successfully processed {len(faces)} images out of {total_images} total images")
    return faces, filenames

def save_embeddings(embeddings, filenames, file_path="face_embeddings.pkl"):
    try:
        if not embeddings:
            print("❌ No embeddings to save!")
            return
            
        df = pd.DataFrame(embeddings)
        df["filename"] = filenames
        df.to_pickle(file_path)
        print(f"✅ Successfully saved {len(embeddings)} embeddings to {file_path}")
    except Exception as e:
        print(f"❌ Failed to save embeddings: {str(e)}")
        import traceback
        print(traceback.format_exc())

def load_embeddings(file_path="face_embeddings.pkl"):
    try:
        if not os.path.exists(file_path):
            print(f"❌ Embeddings file not found: {file_path}")
            return pd.DataFrame()
            
        df = pd.read_pickle(file_path)
        print(f"✅ Successfully loaded {len(df)} embeddings from {file_path}")
        return df
    except Exception as e:
        print(f"❌ Failed to load embeddings: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return pd.DataFrame()

def find_similar_faces(query_image_path, df, top_n=5):
    # Detect face in query image
    query_face = detect_face(query_image_path)
    if query_face is None:
        print("❌ No face detected in query image")
        return []
        
    # Normalize the query feature
    query_feature = query_face.flatten().astype(np.float32)
    query_feature = query_feature / np.linalg.norm(query_feature)
    
    features = df.drop(columns=["filename"]).values.astype(np.float32)
    if features.shape[0] == 0:
        print("⚠️ No embeddings found in dataset.")
        return []

    # Normalize all features
    features = features / np.linalg.norm(features, axis=1)[:, np.newaxis]
    
    # Calculate cosine similarity
    similarities = np.dot(features, query_feature)
    top_indices = similarities.argsort()[-top_n:][::-1]

    print("\n🎯 Top Similar Faces:")
    for idx in top_indices:
        print(f"{df['filename'].iloc[idx]} — Similarity: {similarities[idx]:.4f}")
    return [df["filename"].iloc[i] for i in top_indices]

def show_results(query_path, similar_paths):
    plt.figure(figsize=(15, 5))
    plt.subplot(1, len(similar_paths) + 1, 1)
    plt.imshow(cv2.cvtColor(cv2.imread(query_path), cv2.COLOR_BGR2RGB))
    plt.title("Query")
    plt.axis("off")

    for i, path in enumerate(similar_paths):
        img = cv2.imread(path)
        if img is not None:
            plt.subplot(1, len(similar_paths) + 1, i + 2)
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            plt.title(f"Match {i+1}")
            plt.axis("off")
    plt.tight_layout()
    plt.show()

# ====== MAIN ======
if __name__ == "__main__":
    print("Starting face detection and feature extraction...")
    print(f"Dataset path: {DATASET_PATH}")
    
    try:
        embeddings, filenames = create_embeddings(DATASET_PATH)
        if embeddings:
            save_embeddings(embeddings, filenames)
            
        df = load_embeddings()
        if not df.empty:
            similar_faces = find_similar_faces(QUERY_IMAGE, df)
            if similar_faces:
                show_results(QUERY_IMAGE, similar_faces)
    except Exception as e:
        print(f"❌ Main process error: {str(e)}")
        import traceback
        print(traceback.format_exc())
