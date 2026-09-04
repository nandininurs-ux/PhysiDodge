import os
import urllib.request
import cv2

# 1. Automatically download the XML file if it doesn't exist locally
xml_filename = "haarcascade_frontalface_default.xml"
if not os.path.exists(xml_filename):
    print("Downloading face detection model file...")
    url = (
        "https://raw.githubusercontent.com/opencv/opencv/master/data/"
        "haarcascades/haarcascade_frontalface_default.xml"
    )
    urllib.request.urlretrieve(url, xml_filename)
    print("Download complete!")

# 2. Load the face classifier
face_cascade = cv2.CascadeClassifier(xml_filename)

# 3. Start webcam
cap = cv2.VideoCapture(0)

print("Press 'q' in the video window to exit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture image from camera.")
        break

    # Convert to grayscale for detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
    )

    # Draw green boxes around faces
    for x, y, w, h in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display live feed
    cv2.imshow("Real-Time Face Detection", frame)

    # Exit on 'q' press
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()