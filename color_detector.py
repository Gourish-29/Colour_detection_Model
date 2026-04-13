print("PROGRAM STARTED")

import cv2
import numpy as np

# Open camera (change to 1 if 0 doesn't work)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Camera not accessible")
    exit()

BOX_SIZE = 120

# HSV color ranges (10+ colors)
colors = {
    "Red":      [(0, 120, 70), (10, 255, 255)],
    "Dark Red": [(170, 120, 70), (180, 255, 255)],
    "Green":    [(36, 50, 70), (89, 255, 255)],
    "Blue":     [(90, 50, 70), (128, 255, 255)],
    "Yellow":   [(15, 100, 100), (35, 255, 255)],
    "Orange":   [(10, 100, 20), (25, 255, 255)],
    "Purple":   [(129, 50, 70), (158, 255, 255)],
    "Pink":     [(159, 50, 70), (169, 255, 255)],
    "Brown":    [(10, 100, 20), (20, 200, 200)],
    "Black":    [(0, 0, 0), (180, 255, 30)],
    "White":    [(0, 0, 200), (180, 30, 255)]
}

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to read frame")
        break

    height, width, _ = frame.shape
    cx, cy = width // 2, height // 2

    x1, y1 = cx - BOX_SIZE // 2, cy - BOX_SIZE // 2
    x2, y2 = cx + BOX_SIZE // 2, cy + BOX_SIZE // 2

    # Draw green box
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    roi = frame[y1:y2, x1:x2]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    detected_color = ""

    # -------- COLOR DETECTION --------
    for name, (lower, upper) in colors.items():
        lower = np.array(lower)
        upper = np.array(upper)

        mask = cv2.inRange(hsv, lower, upper)
        ratio = cv2.countNonZero(mask) / (BOX_SIZE * BOX_SIZE)

        if ratio > 0.25:
            detected_color = name
            break

    # -------- DISPLAY COLOR NAME ON TOP OF BOX --------
    if detected_color:
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2

        text_size, _ = cv2.getTextSize(
            detected_color, font, font_scale, thickness
        )

        # center text horizontally
        text_x = x1 + (BOX_SIZE - text_size[0]) // 2

        # position text just above the box
        text_y = y1 - 8

        # if text goes outside frame, move it below the box
        if text_y < text_size[1]:
            text_y = y2 + text_size[1] + 8

        cv2.putText(
            frame,
            detected_color,
            (text_x, text_y),
            font,
            font_scale,
            (0, 255, 0),
            thickness
        )

    cv2.imshow("Real-Time Color Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
