from ultralytics import YOLO
import cv2
import numpy as np


def detect_burglar(frame):
          
    
    model = YOLO("SmartAICamera\\models\\yolo11n.pt")
    results = model(frame) # Passing frame to model
    for result in results:  # Loop through each detected object
            boxes = result.boxes  # get the boxes attribute
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()  # extract bounding box coordinates
                conf = box.conf  # confidence score
                class_id = box.cls  # class ID
                # Extracting Label from predictions.
                labelname = f"{model.names[int(class_id)]}"
                label = f"{labelname} {conf.item():.2f}"
                if labelname.lower()=="person": 
                     cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                     # Write Label on the detected person
                     cv2.putText(frame, "Intruder", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                     cv2.imshow("intruder",frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()  

def crop_image(image, pts):
    cv2.imshow("Test Image",image)
    # Points should be in the following order: top-left, top-right, bottom-right, bottom-left
    pts = np.array(pts, dtype="float32")

    # Compute the width and height of the new image (the cropped area)
    (tl, tr, br, bl) = pts

    # Calculate the width of the new image as the maximum distance between the corners
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = max(int(widthA), int(widthB))

    # Calculate the height of the new image
    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = max(int(heightA), int(heightB))

    # Define the destination points for the transformation
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]
    ], dtype="float32")

    # Apply the perspective transformation matrix
    M = cv2.getPerspectiveTransform(pts, dst)
    cropped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    cv2.imshow("Cropped Image",cropped)
    

    return cropped


