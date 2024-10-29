import cv2
from ultralytics import YOLO
def essentialfeatures(frame):
          
    
    model = YOLO("SmartAICamera\\models\\essential.pt")
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
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                # Write Label on the detected person
                cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
    return frame            
    
                