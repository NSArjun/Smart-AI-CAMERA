import cv2
import face_recognition
import os
import numpy as np


face_cascade = cv2.CascadeClassifier('SmartAICamera\\models\\face_detector.xml') # Cascade For Face detection.


# Function to check if a face exists or not
def checkface(frame):
         
         # Converting the image to grayscale image
         gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
         # Detecting Faces using Face Cascade
         faces = face_cascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=5)
         # If no face is detected, return False
         if len(faces)==0:
             return False
         print("Human Face Detected.")
         # if face is detected, return True
         return True

# Function to remember a face.
def rememberthisface():
        
        # Reading the number of faces text to give each facial data a unique name.
        with open("files/numberoffaces.txt","r") as fp:
              number = int(fp.read())
              print(number)
        number+=1

        # Updating the number of faces
        with open("files/numberoffaces.txt","w") as fp:
              fp.write(str(number))

        #Initialize the camera      
        video_capture = cv2.VideoCapture(0)
        # Looping variables
        i = 0
        j = 0

        # List to store all the facial data of the person.
        total_encoding = []

        # Looping to take multiple images of the person
        while i<5 and j<10:
            
            # Reading the frame
            ret,unknown_image = video_capture.read() 

            # Converting the image to ND array                     
            img = np.array(unknown_image)

            # Detect if face is there
            faces = face_cascade.detectMultiScale(img,scaleFactor=1.1,minNeighbors=5)
            
            # Extract Facial Encodings from the face
            faceenc = face_recognition.face_encodings(img)
            if len(faces) == 0 or faceenc==[]:
                i-=1
                j+=1
                continue
            
            # Appending The facial encoding to the total array
            total_encoding.append(faceenc)
            i+=1
            j+=1
            
        # Release and destroy the cv2.windows    
        video_capture.release()
        cv2.destroyAllWindows()
        
        # Converting the number to string, to save the file
        number = str(number)
        nameencoding = number+"faceencoding.txt"

        # Storing it in a new array to save the dimensions when reading.        
        total_encodingdup = total_encoding.copy()
        total_encoding = np.array(total_encoding)

        # Writing the file with the facial embeddings
        with open("files\\faces"+"\\"+nameencoding,'w') as f:
            total_encodingS=str(total_encodingdup)
            total_encodingS = total_encodingS.replace('array','')
            total_encodingS = total_encodingS.replace('(','')
            total_encodingS = total_encodingS.replace(')','')
            f.write(str(total_encodingS))
        
        # Success !
        print(f"Face number {number} added !.\n")


# Function to check for remembered faces
def checkforfaces(frame = None):
         
         # passing the argument data to variable
         if frame is not None:
              unknown_image = frame

         # Invoking camera if frame is null     
         else:                   

              video_capture = cv2.VideoCapture(0)
              ret,unknown_image = video_capture.read()
              video_capture.release()
              cv2.destroyAllWindows()

         # Checking if face is detected     
         if not checkface(unknown_image):
                 
            print("No Face recognised.")
            return False
         
         # Converting it into ND array
         unknown_image =  np.array(unknown_image)

         # Extract Facial Embeddings from the image
         faceencoding = face_recognition.face_encodings(unknown_image)

         # Defining the directory for searching for facial data files
         directory = "files\\faces"

         # Iterating through the directory
         for root, dirs, files in os.walk(directory):
             flag = 0
             for filename in files:
               # If the file ends with this name
               if filename.endswith("faceencoding.txt"):
                 filepath = os.path.join(root, filename)
                 
                 # Reading the file and converting to ND array
                 with open(filepath, "r") as f:
                   text = f.read()
                   sfaces = np.array(eval(text))
                   faceencoding = np.array(faceencoding)

                   # If the encoding array is not in the proper dimension. i.e not saved as a proper vector. Return False
                   if faceencoding.shape!=(1,128):
                        print("Unfortunately image has not been captured properly.")
                        return False
                   
                   # Initialise the array to store results.
                   results = []

                   # Compare the facial data
                   for i in sfaces:
                        a = face_recognition.compare_faces(i,faceencoding,tolerance=0.5)
                   results.append(a) 
                   results = np.array(results)
                   # Validate the results
                   result = results.any()

                   # If the results is true, then known face is present.
                   if result:
                        print("Known face has been found in this image")
                        return True
                       
         # If the results is false, then no recognised face is present.
         print("No recognised people have been found in the current view")
         return False


if __name__ == "__main__":
     rememberthisface()
     checkforfaces()





