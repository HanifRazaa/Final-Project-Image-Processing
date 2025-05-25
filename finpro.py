import cv2
import numpy as np
import time

# Initialize camera
cap = cv2.VideoCapture(0)  # Change to 0 for webcam or 'video.mp4' for a video file
ret, img = cap.read()
if not ret:
    print("Failed to open camera.")
    exit()
preproc_contour = img.copy()

def preproc_gray(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.GaussianBlur(gray, (5, 5), 0)

def preproc_canny(frame, t1=100, t2=150):
    return cv2.Canny(preproc_gray(frame), t1, t2)

def preproc_dilate(canny_frame):
    kernel = np.ones((5, 5), np.uint8)
    return cv2.dilate(canny_frame, kernel, iterations=1)

def getContour(dilate_img, draw_img, areaMin=1000):
    contours, _ = cv2.findContours(dilate_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > areaMin:
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            x, y, w, h = cv2.boundingRect(approx)
            cv2.drawContours(draw_img, [approx], -1, (255, 0, 255), 2)
            cv2.rectangle(draw_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(draw_img, f"Points: {len(approx)}", (x + w + 10, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            edges = len(approx)
            circularity = 4 * np.pi * (area / (peri * peri)) if peri != 0 else 0
            label = "Polygon"
            if circularity > 0.8:
                label = "Circle"
            elif edges == 3:
                label = "Triangle"
            elif edges == 4:
                label = "Rectangle"
            elif edges == 8 and circularity < 0.6:
                label = "Octagon"
            cv2.putText(draw_img, f"Shape: {label}", (x + w + 10, y + 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

prev_time = time.time()
frame_count = 0
fps = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    curr_time = time.time()
    if curr_time - prev_time >= 1.0:
        fps = frame_count / (curr_time - prev_time)
        prev_time = curr_time
        frame_count = 0

    # Preprocessing
    canny_output = preproc_canny(frame)
    dilate_output = preproc_dilate(canny_output)
    contour_img = frame.copy()
    getContour(dilate_output, contour_img)

    # FPS counter
    cv2.putText(contour_img, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Show result
    cv2.imshow("Shape Detection", contour_img)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):
        filename = f"shape_capture_{int(time.time())}.png"
        cv2.imwrite(filename, contour_img)
        print(f"Saved: {filename}")

cap.release()
cv2.destroyAllWindows()