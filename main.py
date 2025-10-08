import sys
import cv2
import face_recognition
import os
import pickle
import csv
from datetime import datetime

ATTENDANCE_FILE = "attendance.csv"

# === Utility: Log attendance ===
def mark_attendance(name):
    """
    Log employee attendance into CSV.
    One entry per person per day.
    """
    today = datetime.now().strftime("%Y-%m-%d")
    now_time = datetime.now().strftime("%H:%M:%S")

    # If file doesn't exist, create with header
    if not os.path.exists(ATTENDANCE_FILE):
        with open(ATTENDANCE_FILE, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Name", "Date", "Time"])

    # Check if already logged today
    with open(ATTENDANCE_FILE, "r") as f:
        lines = f.readlines()
        for line in lines:
            if name in line and today in line:
                return  # Already logged today

    # Append new record
    with open(ATTENDANCE_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([name, today, now_time])
    print(f"✅ Attendance marked for {name} at {now_time}")

# === STEP 1: Encode known faces ===
def encode_known_faces(folder_path="known_faces", save_encodings=True):
    """
    Load images and create face encodings.
    Folder structure: known_faces/PersonName/image1.jpg, image2.jpg...
    """
    known_encodings = []
    known_names = []
    
    if not os.path.exists(folder_path):
        print(f"Creating {folder_path} folder...")
        os.makedirs(folder_path)
        print(f"\nPlease create folders for each person:")
        print(f"  {folder_path}/John/photo1.jpg")
        print(f"  {folder_path}/Jane/photo1.jpg")
        return known_encodings, known_names
    
    print("Encoding faces from known_faces folder...\n")
    
    # Loop through each person's folder
    for person_name in os.listdir(folder_path):
        person_path = os.path.join(folder_path, person_name)
        
        if not os.path.isdir(person_path):
            continue
        
        print(f"Processing: {person_name}")
        
        # Loop through images for this person
        for image_name in os.listdir(person_path):
            if image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(person_path, image_name)
                
                try:
                    # Load image
                    image = face_recognition.load_image_file(image_path)
                    
                    # Get face encodings
                    encodings = face_recognition.face_encodings(image)
                    
                    if len(encodings) > 0:
                        known_encodings.append(encodings[0])
                        known_names.append(person_name)
                        print(f"  ✓ {image_name}")
                    else:
                        print(f"  ✗ No face found in {image_name}")
                
                except Exception as e:
                    print(f"  ✗ Error loading {image_name}: {e}")
    
    print(f"\n✓ Encoded {len(known_encodings)} faces from {len(set(known_names))} people\n")
    
    # Save encodings for faster loading next time
    if save_encodings and len(known_encodings) > 0:
        with open('face_encodings.pkl', 'wb') as f:
            pickle.dump({'encodings': known_encodings, 'names': known_names}, f)
        print("✓ Saved encodings to face_encodings.pkl\n")
    
    return known_encodings, known_names


# === STEP 2: Load saved encodings ===
def load_encodings():
    """Load previously saved encodings"""
    if os.path.exists('face_encodings.pkl'):
        with open('face_encodings.pkl', 'rb') as f:
            data = pickle.load(f)
            print(f"✓ Loaded {len(data['encodings'])} saved encodings\n")
            return data['encodings'], data['names']
    return None, None


# === STEP 3: Recognize faces in a single image ===
def recognize_image(image_path, known_encodings, known_names, show_result=True):
    """
    Recognize faces in a single image
    """
    print(f"Processing: {image_path}")
    
    # Load the image
    image = face_recognition.load_image_file(image_path)
    
    # Find all face locations and encodings
    face_locations = face_recognition.face_locations(image)
    face_encodings = face_recognition.face_encodings(image, face_locations)
    
    print(f"Found {len(face_locations)} face(s)\n")
    
    # Convert to BGR for OpenCV
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    # Process each face
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # Compare with known faces
        matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=0.6)
        name = "Unknown"
        confidence = 0
        
        # Find best match
        if True in matches:
            face_distances = face_recognition.face_distance(known_encodings, face_encoding)
            best_match_index = face_distances.argmin()
            
            if matches[best_match_index]:
                name = known_names[best_match_index]
                confidence = (1 - face_distances[best_match_index]) * 100
                print(f"  → {name} ({confidence:.1f}% confident)")
        
        if name == "Unknown":
            print(f"  → Unknown person")
        
        # Draw box and name
        color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
        cv2.rectangle(image_bgr, (left, top), (right, bottom), color, 2)
        
        # Draw label background
        cv2.rectangle(image_bgr, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
        
        # Put name text
        label = f"{name}" if name == "Unknown" else f"{name} ({confidence:.0f}%)"
        cv2.putText(image_bgr, label, (left + 6, bottom - 6), 
                    cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)
    
    if show_result:
        # Show result
        cv2.imshow('Face Recognition - Press any key to close', image_bgr)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    # Save result
    output_path = "recognized_" + os.path.basename(image_path)
    cv2.imwrite(output_path, image_bgr)
    print(f"\n✓ Result saved to: {output_path}\n")


# === STEP 4: Real-time webcam recognition ===
def recognize_webcam(known_encodings, known_names):
    """
    Real-time face recognition from webcam
    Press 'q' to quit
    """
    print("Starting webcam...")
    print("Press 'q' to quit\n")
    
    video_capture = cv2.VideoCapture(0)
    
    if not video_capture.isOpened():
        print("Error: Could not open webcam")
        return
    
    process_this_frame = True
    
    while True:
        ret, frame = video_capture.read()
        
        if not ret:
            break
        
        # Process every other frame for speed
        if process_this_frame:
            # Resize for faster processing
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
            
            # Find faces
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
            
            face_names = []
            for face_encoding in face_encodings:
                matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=0.6)
                name = "Unknown"
                
                if True in matches:
                    face_distances = face_recognition.face_distance(known_encodings, face_encoding)
                    best_match_index = face_distances.argmin()
                    if matches[best_match_index]:
                        name = known_names[best_match_index]
                
                face_names.append(name)

                # ✅ Log attendance if employee recognized
                if name != "Unknown":
                    mark_attendance(name)
        
        process_this_frame = not process_this_frame
        
        # Draw results
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            # Scale back up
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4
            
            # Draw box
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            
            # Draw label
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
            cv2.putText(frame, name, (left + 6, bottom - 6), 
                       cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)
        
        # Display
        cv2.imshow('Face Recognition - Press Q to quit', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    video_capture.release()
    cv2.destroyAllWindows()
    print("\n✓ Webcam closed")


# === MAIN PROGRAM ===
if __name__ == "__main__":
    print("=" * 50)
    print("       FACE ATTENDANCE SYSTEM")
    print("=" * 50)
    print()

    # Load or encode known faces
    known_encodings, known_names = load_encodings()
    if known_encodings is None:
        known_encodings, known_names = encode_known_faces("dataset")

    if len(known_encodings) == 0:
        print("⚠ No known faces loaded!")
        exit()

    while True:
        print("=" * 50)
        print("What would you like to do?")
        print("=" * 50)
        print("1. Recognize faces in an image")
        print("2. Real-time webcam attendance")
        print("3. Re-encode known faces")
        print("4. Exit")
        print()

        choice = input("Enter choice (1-4): ").strip()
        print()

        if choice == "1":
            image_path = input("Enter image path: ").strip()
            if os.path.exists(image_path):
                recognize_image(image_path, known_encodings, known_names)
            else:
                print("✗ Image not found!\n")

        elif choice == "2":
            recognize_webcam(known_encodings, known_names)

        elif choice == "3":
            known_encodings, known_names = encode_known_faces("dataset", save_encodings=True)

        elif choice == "4":
            print("Goodbye!")
            sys.exit()
            break

        else:
            print("✗ Invalid choice!\n")