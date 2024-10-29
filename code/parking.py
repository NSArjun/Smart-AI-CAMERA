from ultralytics import YOLO
import pyttsx3
engine = pyttsx3.init()

model = YOLO("SmartAICamera\models\\topview.pt")
# Text to speech
def text_to_speech(text):
            
            engine.setProperty('rate', 150) 
    
            engine.setProperty('volume', 1)  
    
            engine.say(text)
    
            engine.runAndWait()
            

# coordinnates of parking slots
coordinates_list = [
    (294,129,319,196),# First region
    (263, 129, 293, 196),  # Second region
    (230, 129, 263, 196),   # Third region
    (200, 129, 230, 196),  # Fourth region
    (168, 129, 200, 196),  # Fifth region
    (136, 129, 168, 196),  # Sixth region
    (106, 129, 136, 196),  # Seventh region
    (76, 129, 106, 196),  # Eighth region
    (44, 129, 76, 196),  # Ninth region
    (12, 129, 44, 196), # Tenth region
     #---------------------- 
    (12, 50, 44, 120),  # Eleventh region
    (44, 50, 76, 120),  # Twelfth region
    (76, 50, 107, 120),  # Thirteenth region
    (107, 50, 137, 120),  # Fourteenth region
    (137, 50, 168, 120),  # Fifteenth region
    (168, 50, 200, 120),  # Sixteenth region
    (200, 50, 231, 120),  # Seventeenth region
    (231, 50, 263, 120),  # Eighteenth Region
    (263, 50, 294, 120),   # Nineteenth Region
    (293, 50, 319, 120) # Twenty


]

# Calculate InterSection Over Union :P
def calculate_iou(box1, box2):
    # Unpack the coordinates of the two boxes
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    
    
    # Determine the coordinates of the intersection rectangle
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)
    
    # Compute the area of intersection
    inter_area = max(0, inter_x_max - inter_x_min) * max(0, inter_y_max - inter_y_min)
    
    # Compute the area of both the prediction and ground-truth rectangles
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    
    # Compute the IoU by dividing the intersection area by the union area
    union_area = box1_area + box2_area - inter_area
    iou = inter_area / union_area if union_area > 0 else 0
    
    return iou

# Function to check if the prediction is within a specific bounding box (threshold for IoU can be adjusted)
def is_prediction_in_box(pred_box, target_box, iou_threshold=0.7):
    iou = calculate_iou(pred_box, target_box)
    return iou >= iou_threshold  # Return True if IoU exceeds the threshold

# Function To call as a whole
def parkingFunction(imgpath):

    # Defining List To store the boolean Values.
    boolea = []
    # Defining a list to store the bounding boxes of predictions
    predictions = []
    # Making predictions
    results = model.predict(source=imgpath)
    #iterating through predictions
    for result in results:
                boxes = result.boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()  # extract bounding box coordinates
                    predictions.append((x1,y1,x2,y2)) # Adding to predictions list
                    conf = box.conf  # confidence score
                    class_id = box.cls  # class ID
                    label = f"{model.names[int(class_id)]} {conf.item():.2f}" # Extracting the label
    # displays the results            
    result.show()
    # List to store empty slots
    emptyslots = []            
    
    # Checking if there is an intersection in predictions
    for i in coordinates_list:
          flag = 0
          for j in predictions:
                a = is_prediction_in_box(j,i)
                if a:
                      boolea.append(True)
                      flag = 1
          if flag!=1:
                boolea.append(False)
    
    # Getting the free slots
    for i,j in enumerate(boolea):
          if not j:
                emptyslots.append(i)

    # Telling the Parking slots            
    text_to_speech(f"Parking Slot Available in number {', '.join(str(i) for i in emptyslots)}")
   

if __name__ == "__main__":
      # pass any number 1 to 7
      parkingFunction("CAR PARKING\\5.png")
      
                              



