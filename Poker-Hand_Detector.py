from ultralytics import YOLO
import cv2
import cvzone
import math
import numpy as np
import torch  # Add torch import
import PokerHandFunction

# Initialize video capture
cap = cv2.VideoCapture(0)  # Try webcam

cap.set(3, 1280)  # Width
cap.set(4, 720)  # Height

# Check for GPU availability and print detailed information
print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("CUDA version:", torch.version.cuda)
    print("GPU device name:", torch.cuda.get_device_name(0))
    device = 'cuda:0'
else:
    print("No CUDA GPU available. Using CPU.")
    device = 'cpu'
print(f"Using device: {device}")

# Load YOLO model
model = YOLO("Project_4_Poker_Hand_Detector_1/playingCards.pt").to(device)

classNames = ['10C', '10D', '10H', '10S',
              '2C', '2D', '2H', '2S',
              '3C', '3D', '3H', '3S',
              '4C', '4D', '4H', '4S',
              '5C', '5D', '5H', '5S',
              '6C', '6D', '6H', '6S',
              '7C', '7D', '7H', '7S',
              '8C', '8D', '8H', '8S',
              '9C', '9D', '9H', '9S',
              'AC', 'AD', 'AH', 'AS',
              'JC', 'JD', 'JH', 'JS',
              'KC', 'KD', 'KH', 'KS',
              'QC', 'QD', 'QH', 'QS']

hand = []

try:
    while True:
        success, img = cap.read()
        if not success or img is None:
            print("Failed to read frame")
            break

        # Run YOLO detection
        results = model(img, stream=True, device=device)

        for r in results:
            boxes = r.boxes
            for box in boxes:
                # Bounding Box
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                w, h = x2 - x1, y2 - y1
                cvzone.cornerRect(img, (x1, y1, w, h),l=9)

                # Confidence
                conf = math.ceil((box.conf[0] * 100)) / 100

                # Class Name
                cls = int(box.cls[0])
                cvzone.putTextRect(img, f'{classNames[cls]} {conf}',
                                   (max(0, x1), max(35, y1)),
                                   scale=0.6, thickness=1)\
                
                if conf > 0.5:
                    hand.append(classNames[cls])
        print(hand)
        hand = list(set(hand))
        print(hand)
        if len(hand) == 5:
            
            results = PokerHandFunction.findPokerHand(hand)
            print(results)
            cvzone.putTextRect(img, f'Your Hand: {results} ',(300,75),scale=3, thickness=5)
            
        # Display the frame
        cv2.imshow("Object Detection", img)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except Exception as e:
    print(f"An error occurred: {str(e)}")

finally:
    # Release resources
    cap.release()
    cv2.destroyAllWindows()
