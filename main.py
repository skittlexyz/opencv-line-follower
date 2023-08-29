import cv2
import numpy as np

cap = cv2.VideoCapture(1)

variant = 'color'

# line configuration
horizontal_blur = 25
vertical_blur = 25

edge_min_threshold = 50
edge_max_threshold = 100

line_threshold = 50
min_line_lenght = 50
max_line_gap = 1

line_color = (50, 255, 0)
line_thickness = 2

# color configuration
lower_red = np.array([0, 0, 230])
upper_red = np.array([255, 100, 255])

while True:
    ret, frame = cap.read()
    if not ret:
        break
    if variant == 'color':
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        mask = cv2.inRange(hsv_frame, lower_red, upper_red)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            cv2.drawContours(frame, [contour], -1, (0, 255, 0), 2)

        cv2.imshow('Color Detection', frame)
    
    if variant == 'lines':
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        blurred = cv2.GaussianBlur(gray, (horizontal_blur, vertical_blur), 1)
        edges = cv2.Canny(blurred, edge_min_threshold, edge_max_threshold)

        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, line_threshold, min_line_lenght, max_line_gap)

        line_frame = frame.copy()

        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                #cv2.line(line_frame, (x1, y1), (x2, y2), line_color, line_thickness)
                #print(line)

        cv2.imshow('Line Detection', edges)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
