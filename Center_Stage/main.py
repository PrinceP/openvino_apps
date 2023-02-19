import flet as ft
import threading
import time

import cv2
import imutils
from imutils.video import VideoStream, FPS
import base64

cap = cv2.VideoCapture(0)

def main(page : ft.Page):
    page.title = "Center Stage"

    # Load the haar cascade face detector
    detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    # Set the tracker to None
    tracker = None

    # total number of frames processed thus far and skip frames
    totalFrames = 0
    skip_frames = 50

    def update_images(tracker, totalFrames, skip_frames):
        while(cap.isOpened()):
            # Capture frame-by-frame
            ret, frame = cap.read()

            # Run the face detector to find or update face position
            if tracker is None or totalFrames % skip_frames == 0:
                print("Detecting")
                # convert the frame to grayscale (haar cascades work with grayscale)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                # Detect all faces
                faces = detector.detectMultiScale(gray, scaleFactor=1.05,
                    minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)
                    
                # Check to see if a face was found
                if len(faces) > 0:

                    # Pick the most prominent face
                    initBB = faces[0]
                    
                    # start the tracker
                    # print("Found face. Starting the tracker")
                    tracker = cv2.legacy_TrackerKCF.create()
                    tracker.init(frame, tuple(initBB))
                else:
                    print("Face not found")
                    tracker = None
                
                # encode the resulting frame
                jpg_img = cv2.imencode('.jpg', frame)
                b64_string = base64.b64encode(jpg_img[1]).decode('utf-8')
                image_box.src_base64 = b64_string
                # jpg_img2 = cv2.imencode('.jpg', frame)
                # b64_string2 = base64.b64encode(jpg_img2[1]).decode('utf-8')
                # image_box2.src_base64 = b64_string2
                page.update()

            else:
                (success, box) = tracker.update(frame)
                if success:
                    (x,y,w,h) = [int(v) for v in box]
                    H, W, _ = frame.shape

                    centerX = int(x + (w / 2.0))
                    centerY = int(y + (h / 2.0))
                    
                    # encode the resulting frame
                    jpg_img = cv2.imencode('.jpg', frame)
                    b64_string = base64.b64encode(jpg_img[1]).decode('utf-8')
                    image_box.src_base64 = b64_string

                    # encode the resulting frame
                    # print(frame.shape)
                    # y1:y2+1, x1:x2+1

                    y1 = max(0, centerY - 250)
                    y2 = min(H, centerY + 250)    

                    x1 = max(0, centerX - 250)
                    x2 = min(W, centerX + 250)

                    frame = frame[y1:y2, x1:x2]
                    jpg_img2 = cv2.imencode('.jpg', frame)
                    b64_string2 = base64.b64encode(jpg_img2[1]).decode('utf-8')
                    image_box2.src_base64 = b64_string2
                    
                    page.update()
                else:
                    tracker = None
                    time.sleep(1/115)        
            totalFrames += 1
        
       
     
    b64_string = "x123"         
    image_box = ft.Image(src_base64=b64_string, width=500, height=500)
    video_container = ft.Container(image_box, alignment=ft.alignment.center, expand=True)
    
    image_box2 = ft.Image(src_base64=b64_string, width=500, height=500)
    video_container2 = ft.Container(image_box2, alignment=ft.alignment.center, expand=True)
    
    
    page.add(ft.Row([
        video_container,
        video_container2
    ]
        
       
    ))

    ## theading 
    update_image_thread = threading.Thread(target=update_images, args=(tracker, totalFrames, skip_frames,))
    update_image_thread.daemon = True
    update_image_thread.start()

    page.update()

ft.app(target=main)