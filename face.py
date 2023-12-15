import os
import cv2
import shutil

def detect_faces(image):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    return faces

def generate_identifier(image_path, face):
    return f"face_{face[0]}_{face[1]}_{face[0] + face[2]}_{face[1] + face[3]}"

def create_directories(face_identifiers):
    for identifier in face_identifiers:
        os.makedirs(identifier, exist_ok=True)

def organize_images(image_folder):
    face_identifiers = set()

    for filename in os.listdir(image_folder):
        image_path = os.path.join(image_folder, filename)
        print(f"Processing image: {image_path}")

        # Read the image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Failed to read image: {image_path}")
            continue  # Skip to the next iteration

        # Detect faces
        try:
            faces = detect_faces(image)
        except Exception as e:
            print(f"Error detecting faces in {image_path}: {e}")
            continue

        # Organize images based on detected faces
        for face in faces:
            identifier = generate_identifier(image_path, face)
            face_identifiers.add(identifier)

    # Create directories
    create_directories(face_identifiers)

    # Move images to respective directories
    for filename in os.listdir(image_folder):
        image_path = os.path.join(image_folder, filename)
        image = cv2.imread(image_path)
        faces = detect_faces(image)

        for face in faces:
            identifier = generate_identifier(image_path, face)
            destination_path = os.path.join(identifier, filename)
            shutil.move(image_path, destination_path)

if __name__ == "__main__":
    image_folder = r'D:\Photos\hi'
    organize_images(image_folder)
