import cv2
import person 
import vehicles
import parking
import face
import essentialcam
import blocked
import burglar

print("What is the feature you want to implement")
print("1 - face detection feature\n 2 - vehicle detection\n 3 - parking facility\n 4 - Enter Entry to Vehicle Database\n 5 - To remember a face\n 6 - Essential Camera features \n 7- Camera Blocked Detection \n 8- Burglar Detection ")
choice = int(input())

# Initialize video capture
cap = cv2.VideoCapture(0)

if choice==1:
        
    # Continuously Reading Frames.
    while True:
        ret, frame = cap.read() # Reading the Frame
        if not ret: # if not initialized break
            break
        
        
        # Detect Persons in the frame
        frame = person.detect_persons(frame)
    
        # Display the frame with detections
        cv2.imshow("YOLO Object Detection", frame)
    
        # Exit the loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release video capture and close windows
    cap.release()
    cv2.destroyAllWindows()
elif choice==2:
    testimage = input("Enter your test image path") # Enter your Test image path here.
    vehicles.vehicle(testimage)
elif choice==3:
    imagenumber = int(input("Please Refer to the images in car parking folder. Enter a test image between 1 to 7.\n"))
    imgpath = f"CAR PARKING\\{imagenumber}.png"
    parking.parkingFunction(imgpath)
elif choice==4:
    number = int(input("Enter how many numbers you are going to enter.\n"))
    for i in range(number):
        vehiclenumber = input("Please Enter the vehicle number in Indian Format\n")
        vehicles.enterVehicleDB(vehiclenumber)
elif choice==5:
    print("Please be in a well lit place. Now starting...\n")
    face.rememberthisface()
    
elif choice==6:
    print("Now opening webcam for essential Camera.")
    # Continuously Reading Frames.
    while True:
        ret, frame = cap.read() # Reading the Frame
        if not ret: # if not initialized break
            break
        
        
        # Detect Persons in the frame
        frame = essentialcam.essentialfeatures  (frame)
    
        # Display the frame with detections
        cv2.imshow("YOLO Object Detection", frame)
    
        # Exit the loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release video capture and close windows
    cap.release()
    cv2.destroyAllWindows()
elif choice==7:
    # Start the webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open video stream.")
        exit()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break
    
        # Check if the camera is blocked
        if blocked.is_camera_blocked(frame):
            cv2.putText(frame, "Camera Blocked!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            cv2.putText(frame, "Camera Clear", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
        # Show the current frame
        cv2.imshow('Camera Feed', frame)
    
        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release the camera and close the window
    cap.release()
    cv2.destroyAllWindows()    

elif choice==8:
    burglarchoice = int(input("We have only two input images for this model. Enter 1 for image with burglar. Enter 0 for image without burglar."))
    choice = ["SmartAICamera\\images\\wall.webp","SmartAICamera\\images\\walltest.jpg"]
    # Load the image
    image = cv2.imread(choice[burglarchoice])
    print(image.shape)
    # Define the four corner points of the region to crop (in the original image)
    # coordinates: [(top-left), (top-right), (bottom-right), (bottom-left)]
    pts = [(0, 150), (748, 36), (748, 160), (0, 245)]
    
    # Crop the image using the four corner points
    cropped = burglar.crop_image(image, pts)      
    burglar.detect_burglar(cropped)


