import cv2

# https://github.com/opencv/opencv/blob/master/data/haarcascades

# Load the pre-trained model for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load the pre-trained model for person detection
person_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_fullbody.xml')

# Load the pre-trained model for person detection by upperbody
person_cascade2 = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_upperbody.xml')

video_capture = cv2.VideoCapture(0)  # Use 0 for the default webcam
total_users = 0
previous_user_count = 0

while True:
    # Read the current frame from the video stream
    ret, frame = video_capture.read()

    # Perform object detection on the frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Detect people
    people = person_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    # Detect people
    people_upperbody = person_cascade2.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Detect people
    # people_by_upperbody = person_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Update the user count based on the detected objects

    if len(faces) > len(people):
        total_users = len(faces) + len(people_upperbody)
    current_user_count = len(people) # len(faces) # + len(people)
    if current_user_count > previous_user_count:
        total_users = current_user_count - previous_user_count + len(people_upperbody)
    previous_user_count = current_user_count

    # Add a frame around each detected user
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Draw the user count on the frame
    cv2.putText(frame, "Users: " + str(total_users), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Display the frame with the user count
    cv2.imshow("Users Counter", frame)

    # Check for user interrupt to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close the windows
video_capture.release()
cv2.destroyAllWindows()
