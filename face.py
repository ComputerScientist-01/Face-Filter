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

        # Check if the source file exists
        if not os.path.exists(image_path):
            print(f"Source file not found: {image_path}")
            continue  # Skip to the next iteration

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

            # Check if the destination directory exists
            if not os.path.exists(identifier):
                os.makedirs(identifier, exist_ok=True)

            face_identifiers.add(identifier)

    # Move images to respective directories
    for filename in os.listdir(image_folder):
        image_path = os.path.join(image_folder, filename)

        # Check if the source file exists
        if not os.path.exists(image_path):
            print(f"Source file not found: {image_path}")
            continue  # Skip to the next iteration

        image = cv2.imread(image_path)
        faces = detect_faces(image)

        for face in faces:
            identifier = generate_identifier(image_path, face)
            destination_path = os.path.join(identifier, filename)

            # Check if the destination directory exists
            if not os.path.exists(identifier):
                os.makedirs(identifier, exist_ok=True)

            try:
                print(f"Moving {image_path} to {destination_path}")
                shutil.move(image_path, destination_path)
            except FileNotFoundError as e:
                print(f"Error moving file: {e}")
            except Exception as e:
                print(f"Unexpected error: {e}")

if __name__ == "__main__":
    # Replace 'your_image_folder' with the path to your folder containing images
    image_folder = r'C:\Users\YourUsername\Desktop\images'
    organize_images(image_folder)
