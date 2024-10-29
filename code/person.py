from ultralytics import YOLO
import cv2
import face
import datetime
import time

# Load the YOLOv11 model for persons detection
model = YOLO("SmartAICamera\\models\\yolo11n.pt")  
# Define the action to be taken when a person is detected
def person_detected_action():
    print("Person detected!")


tracker = cv2.TrackerCSRT_create() # Initialize a tracker (CSRT is more accurate, but slower)
tracking_person = False # Variable to store if we are tracking a person
bbox = None  # Bounding box for the tracker
success = False # Flag Variable - Boolean

# Function to detect the persons
def detect_persons(frame):
    global tracking_person, bbox, success # Providing global access to variables
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
            
            # If person is detected with a certain confidence level
            if labelname.lower() == 'person' and conf.item()>0.75:
                        # Draw a rectangle around the detected person
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                        # Write Label on the detected person
                        cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

                        # If no person is being tracked right now
                        if not tracking_person:
                                    
                                    # Defining bounding box Co-Ordinates
                                    bbox = (int(x1), int(y1), int(x2 - x1), int(y2 - y1))
                                    # Initialize tracker on the first detected person
                                    tracker.init(frame, bbox)


                                    # Checking if the face is there or not
                                    if face.checkface(frame):
                                          
                                          # Checking if face the faces is not known
                                          if not face.checkforfaces(frame= frame):
                                                
                                                # Extracting Current Time
                                                currenttime = str(datetime.datetime.now())
                                                currenttime = currenttime.replace(':','_')
                                                currenttime = currenttime.replace('-','_') 
                                                
                                                # Printing the Current Time
                                                print(f"Time is {datetime.datetime.now()}")
                                                # Log the Person entry time
                                                with open("files/logs.txt","a") as fp:
                                                      fp.write(f"Person Detected at {str(datetime.datetime.now())}\n")

                                                # Saving the image in a specific Directory      
                                                print(cv2.imwrite("/files/unknown/"+f"{currenttime}.jpg",img=frame))
                                    else:
                                          # Extracting Current Time
                                          currenttime = str(datetime.datetime.now())
                                          currenttime = currenttime.replace(':','_')
                                          currenttime = currenttime.replace('-','_') 
                                          # Log the Person entry time
                                          with open("files/logs.txt","a") as fp:
                                                fp.write(f"Person Detected but face not detected at {currenttime}\n")

                                          # Saving the image in a specific Directory      
                                          print(cv2.imwrite("files/unknown/"+f"{currenttime}.jpg",img=frame))
                                          
                                          # Printing No Face Detected
                                          print("No Face Detected")  
                                    
                                    # Initializing the flag as True                
                                    tracking_person = True
                                    break 
                        
                        else:
                                    # Update the tracker
                                    success, bbox = tracker.update(frame)

                                                       
                                    if success:
                                        # If tracking is successful, draw the bounding box
                                        x, y, w, h = [int(v) for v in bbox]
                                        
                                        # If a person moves out of frame. Breake the object tracking algorithm.
                                        if x + w > frame.shape[1] and y + h > frame.shape[0]:
                                                print("Tracking failed - bounding box out of frame")
                                                
                                                # Make The flag as False
                                                tracking_person = False
                                    
                                                # Wait for 2 seconds      
                                                time.sleep(2)
                                        # Draw a rectangle on the image and Annotate on the image with the label for cv2 tracking. Uncomment these lines to see results.      
                                        #cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                                        #cv2.putText(frame, "Tracking Person", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    return frame


