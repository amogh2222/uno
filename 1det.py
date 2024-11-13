import cv2
import numpy as np

def region_of_interest(img, vertices):
    
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, vertices, 255)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def draw_lines(img, lines):
    
    if lines is None:
        return
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 3)

def process_frame(frame):

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    
    
    edges = cv2.Canny(blur, 50, 150)
    

    height, width = edges.shape
    roi_vertices = [(0, height), (width / 2, height / 2), (width, height)]
    roi = region_of_interest(edges, np.array([roi_vertices], np.int32))
    
    
    lines = cv2.HoughLinesP(roi, 1, np.pi / 180, threshold=50, minLineLength=100, maxLineGap=50)
    
    
    draw_lines(frame, lines)
    
    return frame

def main():
    
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
        
        processed_frame = process_frame(frame)
        
        
        cv2.imshow('Lane Detection', processed_frame)

        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()