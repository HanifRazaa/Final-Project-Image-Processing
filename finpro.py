import cv2
import numpy as np
import time

cam = cv2.VideoCapture(0)
prev_time = time.time()
frame_count = 0
fps = 0

while True:
    ret, frame = cam.read()
    if not ret:
        break

    frame_count += 1
    curr_time = time.time()
    if curr_time - prev_time >= 1.0:
        fps = frame_count / (curr_time - prev_time)
        prev_time = curr_time
        frame_count = 0

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    bilateral_filtered = cv2.bilateralFilter(gray, 5, 50, 50)

    # Apply adaptive thresholding to highlight edges
    edges = cv2.adaptiveThreshold(
        bilateral_filtered, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2
    )

    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 500:  # Ignore small contours (noise)
            continue

        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        if len(approx) >= 3:
            cv2.drawContours(frame, [approx], 0, (0, 255, 0), 2)

            vertices = len(approx)
        shape = "Unidentified"
        if vertices == 3:
            shape = "Triangle"
        elif vertices == 4:
            (x, y, w, h) = cv2.boundingRect(approx)
            ar = w / float(h)
            shape = "Square" if 0.95 <= ar <= 1.05 else "Rectangle"
        else:
            perimeter = cv2.arcLength(contour, True)
            area = cv2.contourArea(contour)
            if perimeter == 0:
                continue
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            if 0.8 < circularity <= 1.2:
                shape = "Circle"
            else:
                shape = "Polygon"

            # Annotate shape label
            x, y = approx[0][0]
            cv2.putText(frame, shape, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    # Overlay FPS counter
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.imshow('Shape Detection', frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):
        filename = f"shape_capture_{int(time.time())}.png"
        cv2.imwrite(filename, frame)
        print(f"Saved: {filename}")

cam.release()
cv2.destroyAllWindows()