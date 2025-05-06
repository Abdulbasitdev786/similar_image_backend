import os
import uuid
import clip
import torch
from PIL import Image
from qdrant_client import QdrantClient
from qdrant_client.http import models
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
import io
import base64

# ======== CONFIG ========
IMAGE_FOLDER = "C:\\Users\\sb\\Downloads\\common_frontal"
UPLOAD_FOLDER = "uploads"
COLLECTION_NAME = "image_vectors"
VECTOR_SIZE = 512
QDRANT_URL = "http://localhost:6333"
TOP_K = 5
BATCH_SIZE = 100
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
# ========================

app = Flask(__name__)
CORS(app)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Create upload folder if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize CLIP model and Qdrant client
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
client = QdrantClient(url=QDRANT_URL)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_image_embedding(image_path_or_obj):
    if isinstance(image_path_or_obj, str):
        image = Image.open(image_path_or_obj).convert("RGB")
    else:
        image = image_path_or_obj.convert("RGB")
        
    image = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = model.encode_image(image)
    return embedding.cpu().numpy().flatten()

def is_already_indexed(image_path):
    hits = client.scroll(
        collection_name=COLLECTION_NAME,
        scroll_filter=models.Filter(
            must=[
                models.FieldCondition(
                    key="image_path",
                    match=models.MatchValue(value=image_path)
                )
            ]
        ),
        limit=1
    )
    return len(hits[0]) > 0

def create_collection_if_needed():
    if not client.collection_exists(COLLECTION_NAME):
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=models.VectorParams(size=VECTOR_SIZE, distance=models.Distance.COSINE)
        )

@app.route('/api/embed', methods=['POST'])
def embed_all_images_endpoint():
    create_collection_if_needed()
    
    image_paths = [
        os.path.join(IMAGE_FOLDER, f)
        for f in os.listdir(IMAGE_FOLDER)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ]
    
    new_points = []
    processed_count = 0
    skipped_count = 0

    for path in image_paths:
        if is_already_indexed(path):
            skipped_count += 1
            continue
            
        try:
            vector = get_image_embedding(path)
            new_points.append(
                models.PointStruct(
                    id=str(uuid.uuid4()),
                    vector=vector.tolist(),
                    payload={"image_path": path}
                )
            )
            processed_count += 1

            if len(new_points) >= BATCH_SIZE:
                client.upsert(collection_name=COLLECTION_NAME, points=new_points)
                new_points = []
        except Exception as e:
            print(f"Error processing {path}: {str(e)}")

    if new_points:
        client.upsert(collection_name=COLLECTION_NAME, points=new_points)

    return jsonify({
        "success": True,
        "message": "Embedding and upload complete",
        "processed": processed_count,
        "skipped": skipped_count
    })

# @app.route('/api/search', methods=['POST'])
# def search_similar_images_endpoint():
#     if 'image' not in request.files and 'image_base64' not in request.json:
#         return jsonify({"error": "No image provided"}), 400
        
#     try:
#         # Get number of results to return (default to TOP_K)
#         limit = request.args.get('limit', TOP_K, type=int)
        
#         # Handle regular file upload
#         if 'image' in request.files:
#             file = request.files['image']
#             if file.filename == '' or not allowed_file(file.filename):
#                 return jsonify({"error": "Invalid file"}), 400
                
#             # Save the uploaded file
#             filename = secure_filename(file.filename)
#             filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
#             file.save(filepath)
            
#             # Get embedding
#             query_vector = get_image_embedding(filepath)
            
#         # Handle base64 encoded image data
#         elif 'image_base64' in request.json:
#             try:
#                 image_data = base64.b64decode(request.json['image_base64'])
#                 image = Image.open(io.BytesIO(image_data))
#                 query_vector = get_image_embedding(image)
#             except Exception as e:
#                 return jsonify({"error": f"Invalid base64 image: {str(e)}"}), 400
        
#         # Search for similar images
#         results = client.search(
#             collection_name=COLLECTION_NAME,
#             query_vector=query_vector,
#             limit=limit
#         )
        
#         # Format results
#         formatted_results = []
#         for r in results:
#             formatted_results.append({
#                 "score": float(r.score),
#                 "image_path": r.payload['image_path']
#             })
            
#         return jsonify({
#             "results": formatted_results
#         })
        
#     except Exception as e:
#         return jsonify({"error": str(e)}), 500
@app.route('/api/search', methods=['POST'])
def search_similar_images_endpoint():
    if 'image' not in request.files and 'image_base64' not in request.json:
        return jsonify({"error": "No image provided"}), 400

    try:
        # Get number of results to return (default to TOP_K)
        limit = request.args.get('limit', TOP_K, type=int)

        # Handle regular file upload
        if 'image' in request.files:
            file = request.files['image']
            if file.filename == '' or not allowed_file(file.filename):
                return jsonify({"error": "Invalid file"}), 400

            # Save the uploaded file
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Get embedding
            query_vector = get_image_embedding(filepath)

        # Handle base64 encoded image data
        elif 'image_base64' in request.json:
            try:
                image_data = base64.b64decode(request.json['image_base64'])
                image = Image.open(io.BytesIO(image_data))
                query_vector = get_image_embedding(image)
            except Exception as e:
                return jsonify({"error": f"Invalid base64 image: {str(e)}"}), 400

        # Search for similar images
        results = client.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_vector,
            limit=limit
        )

        # Format results with base64 images
        formatted_results = []
        for r in results:
            image_path = r.payload['image_path']
            try:
                with open(image_path, "rb") as img_file:
                    encoded = base64.b64encode(img_file.read()).decode('utf-8')
                    mime_type = "image/jpeg" if image_path.lower().endswith((".jpg", ".jpeg")) else "image/png"
                    base64_image = f"data:{mime_type};base64,{encoded}"
                # with open(image_path, "rb") as img_file:
                #     # base64_image = base64.b64encode(img_file.read()).decode('utf-8')
                #     encoded = base64.b64encode(img_file.read()).decode('utf-8')
                #     mime_type = "image/jpeg" if image_path.lower().endswith((".jpg", ".jpeg")) else "image/png"
                #     base64_image = f"data:{mime_type};base64,{encoded}"
            except Exception as e:
                base64_image = None  # Optional: you can handle the error if needed

            formatted_results.append({
                "score": float(r.score),
                "image_path": image_path,
                "image_base64": base64_image
            })

        return jsonify({
            "results": formatted_results
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    create_collection_if_needed()
    app.run(debug=True, host='0.0.0.0', port=8080)
