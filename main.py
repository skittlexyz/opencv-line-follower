import cv2
import numpy as np
import time

def calculate_centroid(contours):
    centroid_x = 0
    centroid_y = 0
    total_points = 0

    for contour in contours:
        for point in contour:
            x, y = point[0]
            centroid_x += x
            centroid_y += y
            total_points += 1

    centroid_x /= total_points
    centroid_y /= total_points

    return int(centroid_x), int(centroid_y)

# Function to process the webcam feed
def process_webcam_feed():
    cap = cv2.VideoCapture(1)  # Initialize webcam feed

    while True:
        ret, frame = cap.read()  # Read a frame from the webcam feed

        if not ret:
            break

        # Convert frame to HSV color space
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Define lower and upper bounds for the color you want to detect (here, green)
        lower_color = np.array([0, 0, 0])
        upper_color = np.array([50, 50, 50])

        # Create a mask using the defined color range
        mask = cv2.inRange(hsv, lower_color, upper_color)

        # Find contours in the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Draw contours around detected objects
        if contours:
            for contour in contours:
                cv2.drawContours(frame, [contour], -1, (0, 0, 255), 5)
            centroid_x, centroid_y = calculate_centroid(contours)

            # Draw a line based on centroid
            cv2.line(frame, (centroid_x - 50, centroid_y), (centroid_x + 50, centroid_y), (0, 255, 0), 5)
            cv2.line(frame, (centroid_x, centroid_y - 50), (centroid_x, centroid_y + 50), (0, 255, 0), 5)

        # Printing contours
        if contours:
            print("Contours:")
            for i, contour in enumerate(contours):
                print(f"Contour {i+1}:")
                for point in contour:
                    print(f"\t{point[0]}")

        # Display the frame with the drawn contours
        cv2.imshow('Color Detection', frame)

        # Press 'q' to exit the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        time.sleep(0.5)

    cap.release()
    cv2.destroyAllWindows()

# Run the function to process the webcam feed
process_webcam_feed()
