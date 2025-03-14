# Face-Filter
I'll create a Python script to identify and group similar faces in photos using facial recognition.

This script - app.py uses the `face_recognition` library (which uses dlib under the hood) to identify and cluster similar faces. The key components:

1. Face detection and encoding extraction from photos
2. Clustering similar faces using DBSCAN algorithm
3. Organizing photos by person into separate folders

To use this code:

1. Install requirements:
```bash
pip install face_recognition numpy scikit-learn opencv-python pillow
```

2. Adjust the parameters at the bottom of the script:
   - `input_directory`: Folder containing family photos
   - `output_directory`: Where to save sorted faces
   - `eps`: Controls clustering sensitivity (lower = stricter matching)

3. Run the script and check the results:
   - Tune the `eps` parameter if needed (0.4-0.6 range works well)
   - Each person gets their own subfolder named "person_X"

The script handles different lighting conditions and face angles reasonably well. For better results, ensure your photos have good lighting and clear faces.
