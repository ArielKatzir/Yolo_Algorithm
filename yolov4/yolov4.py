# comments with # are comments
# comments with '''...''' are optional code lines


import cv2
import numpy as np

# Load YOLO

# creating a deep nural network object from the yolov3 files
net = cv2.dnn.readNet("yolov4.weights", "yolov4.cfg")

# load the classes from the files
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# get the names of the layers which will help us get the final object on the screen
layers_names = net.getLayerNames()
output_layers = [layers_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# creates image object
img = cv2.imread("pexels-photo-777157.jpeg")
vid = cv2.VideoCapture("basketball_video.mp4")

# While true for video analysis
while True:
    # shrinks the image window
    _, img = vid.read()
    img = cv2.resize(img, None, fx=1, fy=1)
    img_height, img_width, channels = img.shape


    # blob is way to extract the features and objects from the image
    # it will create 3 images: r g and b images which can be processes by the algorithm
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    # 0.00392 is a scale factor that was found to work best apparently
    # 0,0,0 is mean subtractions from each layer of the network
    # True means we inverting blue with red. opencv uses BGR instead of RGB(not sure why it matters)
    # crop is false since we don't want to crop the image

    # passing the blob into the network and getting the output
    net.setInput(blob)
    outs = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []

    def font_size(img_wh : (int)) -> (float):
        w,h = img_wh
        if w < 800 or h < 1000:
            return (0.4,1.0)
        else:
            return (1.0,2.0)

    for out in outs:
        # detecting confidence
        for detection in out:
            scores = detection[5:]
            # class_id is the number that tells us which object it is. we are looking for a car obviously
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            # only take high confidence values
            if confidence > 0.5:
                # get coordinates of boxes
                center_x = int(detection[0] * img_width)
                center_y = int(detection[1] * img_height)
                object_width = int(detection[2] * img_width)
                object_height = int(detection[3] * img_height)
                
                # draw a small circle in the center of each object detected
                "cv2.circle(img, (center_x, center_y), 10, (0, 255, 0), 2)"
                # 10 is r. 0,255,0 makes it green. and 2 is a thickness parameter

                # rectangle coordinates
                x = int(center_x - object_width / 2)
                y = int(center_y - object_height / 2)

                # append each box to an array with centre,w,h coordinates
                boxes.append([x, y, object_width, object_height])
                # append each confidence to an array
                confidences.append(float(confidence))
                # to know the names of the objects we detected
                class_ids.append(class_id)

    # non-max-supression is used to make sure only one object is detected per objectc present
    # (only one box is drawn per car)
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # looping through the max confidence objects to draw a box around them

    confidence_font = cv2.FONT_HERSHEY_SIMPLEX
    box_colours =  np.random.uniform(0 ,255, size=(len(boxes) , 3)) 


    for i in range(len(boxes)):
        if i in indexes:
            x, y, box_width, box_height = boxes[i]
            # names which type of object it is
            label = classes[class_ids[i]]

            colour = box_colours[i]

            font_s = font_size((img_width,img_height))

            # Drawing the boundong boxes
            cv2.rectangle(img, (x, y), (x + box_width, y + box_height), colour , 2)

            # Adding confidence score
            cv2.putText(img,f'{int(confidences[i]*100)}%  {label}',
                    (x, y-10), confidence_font, font_s[0], (0,0,255), int(font_s[1]))


    # shows the image
    cv2.imshow("Image", img)

    # keeps the image open
    key = cv2.waitKey(1)
    if key == 35:
        break

vid.release( )
# exits the window image
cv2.destroyAllWindows()