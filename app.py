import os
import face_recognition
import numpy as np
import shutil
from sklearn.cluster import DBSCAN
from collections import defaultdict
import cv2
from PIL import Image

def extract_faces(image_directory):
    """Extract faces and their encodings from images in a directory."""
    known_face_encodings = []
    face_image_paths = []
    
    # Iterate through each image file in the directory
    for filename in os.listdir(image_directory):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            # Load the image
            image_path = os.path.join(image_directory, filename)
            try:
                # Use face_recognition library to find faces
                image = face_recognition.load_image_file(image_path)
                face_locations = face_recognition.face_locations(image)
                face_encodings = face_recognition.face_encodings(image, face_locations)
                
                # Store encodings and paths for each face found
                for encoding in face_encodings:
                    known_face_encodings.append(encoding)
                    face_image_paths.append(image_path)
                
                print(f"Processed {filename}, found {len(face_encodings)} faces")
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")
    
    return known_face_encodings, face_image_paths

def cluster_faces(face_encodings, face_paths, eps=0.5):
    """
    Cluster face encodings using DBSCAN algorithm.
    eps: The maximum distance between two samples for them to be considered in the same cluster.
    """
    # Convert face encodings list to numpy array
    X = np.array(face_encodings)
    
    # Perform DBSCAN clustering
    clustering = DBSCAN(eps=eps, min_samples=3, metric="euclidean").fit(X)
    
    # Get the cluster labels assigned to each face
    labels = clustering.labels_
    
    # Group faces by cluster label
    face_clusters = defaultdict(list)
    for i, label in enumerate(labels):
        # Skip noise points (label -1)
        if label >= 0:
            face_clusters[label].append(face_paths[i])
    
    return face_clusters

def extract_face_from_image(image_path, face_location, output_path):
    """Extract a face from an image and save it to output_path."""
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        return False
    
    # Convert face_location (top, right, bottom, left) to OpenCV format (x, y, w, h)
    top, right, bottom, left = face_location
    x, y, w, h = left, top, right-left, bottom-top
    
    # Add padding around the face
    padding = int(0.3 * max(w, h))
    x = max(0, x - padding)
    y = max(0, y - padding)
    w = min(image.shape[1] - x, w + 2*padding)
    h = min(image.shape[0] - y, h + 2*padding)
    
    # Extract the face region
    face_image = image[y:y+h, x:x+w]
    
    # Save the face image
    cv2.imwrite(output_path, face_image)
    return True

def organize_face_clusters(image_directory, output_directory, eps=0.5):
    """
    Organize faces in the input directory into clusters in the output directory.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_directory, exist_ok=True)
    
    # Extract faces from images
    face_encodings, face_paths = extract_faces(image_directory)
    
    if not face_encodings:
        print("No faces found in the input directory")
        return
    
    # Cluster the faces
    face_clusters = cluster_faces(face_encodings, face_paths, eps)
    
    # Process each cluster
    for cluster_id, image_paths in face_clusters.items():
        # Create a directory for this cluster
        cluster_dir = os.path.join(output_directory, f"person_{cluster_id}")
        os.makedirs(cluster_dir, exist_ok=True)
        
        # Process each image in the cluster
        for i, image_path in enumerate(image_paths):
            try:
                # Load the image
                image = face_recognition.load_image_file(image_path)
                face_locations = face_recognition.face_locations(image)
                face_encodings = face_recognition.face_encodings(image, face_locations)
                
                # Find which face in this image belongs to this cluster
                best_match_idx = None
                best_match_distance = float('inf')
                
                for idx, face_encoding in enumerate(face_encodings):
                    # Compare with a representative encoding from the cluster
                    # In this simple case, we use the first face encoding we found for this cluster
                    distance = face_recognition.face_distance([face_encodings[0]], face_encoding)[0]
                    if distance < best_match_distance:
                        best_match_distance = distance
                        best_match_idx = idx
                
                if best_match_idx is not None:
                    # Extract and save this face
                    face_location = face_locations[best_match_idx]
                    output_filename = f"{os.path.basename(image_path).split('.')[0]}_face_{i}.jpg"
                    output_path = os.path.join(cluster_dir, output_filename)
                    extract_face_from_image(image_path, face_location, output_path)
                    
            except Exception as e:
                print(f"Error processing {image_path} for cluster {cluster_id}: {str(e)}")
    
    # Count and print results
    cluster_counts = {cluster_id: len(paths) for cluster_id, paths in face_clusters.items()}
    print(f"Found {len(cluster_counts)} distinct people in the photos")
    print(f"Face count per person: {cluster_counts}")

if __name__ == "__main__":
    # Example usage
    input_directory = "family_photos"
    output_directory = "sorted_faces"
    
    # Set eps lower for stricter clustering, higher for more lenient grouping
    organize_face_clusters(input_directory, output_directory, eps=0.45)
