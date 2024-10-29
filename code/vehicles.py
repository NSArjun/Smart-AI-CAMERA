from ultralytics import YOLO
import cv2
import easyocr
import datetime

# Defining the global variables
global x1,x2,y1,y2


#Function to format the detected license plate
def enforce_string_format(input_string):

    # Define the fallback dictionaries for possible misdetections. Note that this is made as per Indian Standards.
    dict_char_to_int = {'O': '0', 'I': '1', 'J': '3', 'A': '4', 'G': '6', 'S': '5','Z':'2','L':'4'}
    dict_int_to_char = {'0': 'O', '1': 'I', '3': 'J', '4': 'A', '6': 'G', '5': 'S','2':'Z'}

    # Initialize the result string
    result = ""

    # First two positions should be letters
    for i in range(2):
        if input_string[i].isalpha():
            result += input_string[i]
        elif input_string[i] in dict_int_to_char:  # Fallback for numbers
            result += dict_int_to_char[input_string[i]]
        else:  # Handle unexpected inputs
            result += "A"  # Default fallback for unexpected numbers

    # Next two positions should be numbers
    for i in range(2, 4):
        if input_string[i].isdigit():
            result += input_string[i]
        elif input_string[i] in dict_char_to_int:  # Fallback for letters
            result += dict_char_to_int[input_string[i]]
        else:  # Handle unexpected inputs
            result += "0"  # Default fallback for unexpected letters

    # Next two positions should be letters
    for i in range(4, 6):
        if input_string[i].isalpha():
            result += input_string[i]
        elif input_string[i] in dict_int_to_char:  # Fallback for numbers
            result += dict_int_to_char[input_string[i]]
        else:  # Handle unexpected inputs
            result += "A"  # Default fallback for unexpected numbers

    # Remaining positions should be numbers
    for i in range(6, len(input_string)):
        if len(result)>=10:
            break
        elif input_string[i].isdigit():
            result += input_string[i]
        elif input_string[i] in dict_char_to_int:  # Fallback for letters
            result += dict_char_to_int[input_string[i]]    
        else:  # Handle unexpected inputs
            result += "0"  # Default fallback for unexpected letters

    return result


#Defining Function for entry for vehicle
def vehicle(Frame):
    
    # Loading the vehicle detection model.
    vehicleModel = YOLO("SmartAICamera\\models\\vehicles.pt")
    # Passing the frame to the model to obtain predictions        
    results = vehicleModel(Frame)

    # Iterating through the list of results
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()  # extract bounding box coordinates
            conf = box.conf  # confidence score
            class_id = box.cls  # class ID
            label = f"{vehicleModel.names[int(class_id)]} {conf.item():.2f}" # Extracting the label

            # Draw the bounding box and label on the frame
            cv2.rectangle(Frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(Frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
    
    
    # Loading the number Plate Model
    numberPlateModel = YOLO("SmartAICamera\\models\\numberplatemodel.pt")
   
    # Ensure the coordinates are within the image boundaries
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(Frame.shape[1], x2)
    y2 = min(Frame.shape[0], y2)
    
    croppedVehicleimg = Frame[int(y1):int(y2), int(x1):int(x2)]  # Crop the image using the bounding box coordinates
    
    # Predicting using the model
    results = numberPlateModel(croppedVehicleimg)
    
    # Iterating through the results
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()  # extract bounding box coordinates
            conf = box.conf  # confidence score
            class_id = box.cls  # class ID
            label = f"{numberPlateModel.names[int(class_id)]} {conf.item():.2f}" # Extracting label name
            
            # Draw the bounding box and label on the frame
            cv2.rectangle(croppedVehicleimg, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(croppedVehicleimg, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
    
    # Converting the image into grayscale
    img_gray = cv2.cvtColor(croppedVehicleimg, cv2.COLOR_BGR2GRAY)
    # String to Integer
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    
    # Ensure the coordinates are within the image boundaries
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(Frame.shape[1], x2)
    y2 = min(Frame.shape[0], y2)
    
    plateROI = img_gray[y1:y2, x1:x2]  # Crop the image using the bounding box coordinates
    
    _, thresh_image = cv2.threshold(plateROI, 128, 255, cv2.THRESH_BINARY) # Performing Threshold
    denoised_image = cv2.GaussianBlur(thresh_image, (5, 5), 0) # Removing Noise
    
    # initialize the easyocr Reader object
    reader = easyocr.Reader(['en'])
    # detect text
    text = reader.readtext(denoised_image)
    
    # Initalizing the detected text image.
    detected = ""

    # Removing Ind if present
    if 'ind' in text[0][1].lower() or text[0][1].lower().startswith("in"):
        detected = text[1][1]
        #cv2.putText(image, text[1][1], (x1, y1-5), cv2.FONT_HERSHEY_COMPLEX, 0.9, (36, 255, 12), 2)
    else:
        detected = text[0][1]
        #cv2.putText(image, text[0][1], (x1, y1-5), cv2.FONT_HERSHEY_COMPLEX, 0.9, (36, 255, 12), 2)
    
    # Detecting vehicle Number
    detectedVehicleNumber =  enforce_string_format(detected.upper())
    print(f" Detected Vehicle Number: {detectedVehicleNumber}")
    
    # Reading Known Vehicle Database
    with open("files/vehicles","r") as fp:
        a = eval(fp.read())

    # If the vehicle number is not in the database    
    if detectedVehicleNumber not in a:

        # Extracting the current time
        currenttime = str(datetime.datetime.now())
        currenttime = currenttime.replace(':','_')
        currenttime = currenttime.replace('-','_') 

        # Printing the Current Time
        print(f"Time is {datetime.datetime.now()}")
        
        # Log the Vehicle number
        with open("files/logs.txt","a") as fp:
            fp.write(f"Vehicle number {detectedVehicleNumber} Detected at {str(datetime.datetime.now())}\n")

        # Save The vehicle image in a specific directory    
        print(cv2.imwrite("files/unknown/"+f"{currenttime}.jpg",img=croppedVehicleimg))

            


# Function to enter details into vehicle database        
def enterVehicleDB(vehiclenumber):
    with open("files/Vehicles",'r') as fp:
        a = eval(fp.read())
    a.append(vehiclenumber)    
    with open("files/Vehicles","w") as fp:
        fp.write(str(a))

if __name__ == "__main__":

    # Replace the file name with your test image
    imgg = cv2.imread("C:\\Users\\admin\\Downloads\\car.webp")         
    vehicle(imgg)
            


        

