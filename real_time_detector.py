import numpy as np
import cv2
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

threshold = 0.75    
min_area = 2000    
model = tf.keras.models.load_model('traffic_model.keras')

def getClassName(classNo):
    classes = { 
        0: 'Give Way', 
        1: 'No Entry', 
        2: 'One-way Traffic', 
        3: 'One-way Traffic',
        4: 'No Vehicles Both Directions', 
        5: 'No Entry for Cycles', 
        6: 'No Entry for Goods Vehicles',
        7: 'No Entry for Pedestrians', 
        8: 'No Entry for Bullock Carts', 
        9: 'No Entry for Hand Carts',
        10: 'No Entry for Motor Vehicles', 
        11: 'Height Limit', 
        12: 'Weight Limit',
        13: 'Axle Weight Limit', 
        14: 'Length Limit', 
        15: 'No Left Turn',
        16: 'No Right Turn', 
        17: 'No Overtaking', 
        18: 'Max Speed 90',
        19: 'Max Speed 110', 
        20: 'Horn Prohibited', 
        21: 'No Parking',
        22: 'No Stopping', 
        23: 'Turn Left Ahead', 
        24: 'Turn Right Ahead',
        25: 'Steep Descent', 
        26: 'Steep Ascent', 
        27: 'Narrow Road',
        28: 'Narrow Bridge', 
        29: 'Unprotected Quay', 
        30: 'Road Hump',
        31: 'Dip', 
        32: 'Loose Gravel', 
        33: 'Falling Rocks', 
        34: 'Cattle',
        35: 'Crossroads', 
        36: 'Side Road Junction', 
        37: 'Side Road Junction',
        38: 'Oblique Side Road', 
        39: 'Oblique Side Road', 
        40: 'T-Junction',
        41: 'Y-Junction', 
        42: 'Staggered Side Road', 
        43: 'Staggered Side Road',
        44: 'Roundabout', 
        45: 'Guarded Level Crossing', 
        46: 'Unguarded Level Crossing',
        47: 'Countdown Marker (100m)', 
        48: 'Countdown Marker (200m)', 
        49: 'Countdown Marker (300m)',
        50: 'Countdown Marker (400m)', 
        51: 'Parking', 
        52: 'Bus Stop',
        53: 'First Aid Post', 
        54: 'Telephone', 
        55: 'Filling Station',
        56: 'Hotel', 
        57: 'Restaurant', 
        58: 'Refreshments' }
    return classes.get(classNo, "Unknown Sign")

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

print("Starting...")

while True:
    success, imgOriginal = cap.read()
    if not success: break
    
    # Saturation Filter 
    hsv = cv2.cvtColor(imgOriginal, cv2.COLOR_BGR2HSV)
    lower_vivid = np.array([0, 80, 50]) 
    upper_vivid = np.array([179, 255, 255])
    mask = cv2.inRange(hsv, lower_vivid, upper_vivid)
    
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=1)

    # Find Objects
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for c in contours:
        area = cv2.contourArea(c)
        if area > min_area:
            # Shape Filter
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.04 * peri, True)
            objCor = len(approx)
            
            if objCor < 3 or (objCor > 4 and objCor < 8): 
                continue 
            
            x, y, w, h = cv2.boundingRect(c)
            
            # Aspect Ratio Filter
            aspect_ratio = w / float(h)
            if aspect_ratio < 0.5 or aspect_ratio > 1.5: continue

            # Crop
            pad = 20
            imgCrop = imgOriginal[max(0,y-pad):min(480,y+h+pad), max(0,x-pad):min(640,x+w+pad)]
            if imgCrop.size == 0: continue

            # Draw box if it passes the shape check
            cv2.rectangle(imgOriginal, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Predict
            try:
                img = np.asarray(imgCrop)
                img = cv2.resize(img, (64, 64)) 
                img = img / 255.0
                img = img.reshape(1, 64, 64, 3)
                
                predictions = model.predict(img, verbose=0)
                classIndex = np.argmax(predictions)
                probabilityValue = np.amax(predictions)
                
                # Higher threshold = Won't speak unless sure
                if probabilityValue > threshold:
                    label = getClassName(classIndex)
                    score = f"{round(probabilityValue*100, 1)}%"
                    cv2.putText(imgOriginal, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(imgOriginal, score, (x, y+h+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            except:
                pass

    cv2.imshow("Traffic Sign Detector", imgOriginal)
    
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()