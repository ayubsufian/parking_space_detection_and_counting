import cv2
import pickle
import numpy as np
from tensorflow.keras.models import load_model

# Load the trained model

model = load_model('model/model_final.h5')
class_dictionary = {0: 'Empty', 1: 'Full'}

video = cv2.VideoCapture("assets/car_test.mp4")

with open('model/carposition.pkl', 'rb') as f:
    positionList = pickle.load(f)

width = 130
height = 65

def checkingCarParking(img):
    imgCrops = []
    spaceCounter = 0
    for pos in positionList:
        x, y = pos
        cropped_img = img[y:y+height, x:x+width]
        imgResized = cv2.resize(cropped_img, (48,48))
        imgNormalized = imgResized / 255.0
        imgCrops.append(imgNormalized)
    imgCrops = np.array(imgCrops)
    predictions = model.predict(imgCrops)

    # Draw a rectangle on the image
    for i, pos in enumerate(positionList):
        x, y = pos
        inId = np.argmax(predictions[i])
        label = class_dictionary[inId]
        if label == 'Empty':
            color = (100, 220, 100)
            thickness = 2
            spaceCounter += 1
            textColor = (0,0,0)
            rectangleColor = (0,255,0)
        else:
            color = (100,100,200)
            thickness = 2
            textColor = (255,255,255)
            rectangleColor = (0,0,255)

        cv2.rectangle(image, pos, (pos[0]+width, pos[1]+height), rectangleColor, thickness)
        font_scale = 0.5
        text_thickness = 1
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_thickness)[0]
        textX = x
        textY = y + height-5
        cv2.rectangle(img, (textX, textY - text_size[1] - 5), (textX + text_size[0]+6, textY + 2), color, -1)
        cv2.putText(img, label, (textX + 3, textY - 3), cv2.FONT_HERSHEY_SIMPLEX, font_scale, textColor, text_thickness)
    cv2.putText(image, f'Space Counter: {spaceCounter}', (100,50), (cv2.FONT_HERSHEY_SIMPLEX), 1, textColor, 2)

while True:
    if video.get(cv2.CAP_PROP_POS_FRAMES) == video.get(cv2.CAP_PROP_FRAME_COUNT):
        video.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ret, image = video.read()
    image = cv2.resize(image, (1280,720))
    if not ret:
        break

    checkingCarParking(image)
    cv2.imshow("image", image)
    if cv2.waitKey(10) == ord('q'):
        break

video.release()
cv2.destroyAllWindows()  
