import cv2 as cv
import numpy as np


net = cv.dnn.readNet('best.onnx')
img = cv.imread('test1.jpeg')

def draw_label(im, label, x, y, colors):
    """Draw text onto image at location."""
    # Get text size.
    text_size = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.7, 1)
    dim, baseline = text_size[0], text_size[1]
    # Use text size to create a BLACK rectangle.
    
    for i in range(len(classes)):
        class_id = classes[i]

        color = colors[class_id % len(colors)]
    cv.rectangle(im, (x,y), (x + dim[0], y + dim[1] + baseline), color, cv.FILLED);
    # Display text inside the rectangle.
    cv.putText(im, label, (x, y + dim[1]), cv.FONT_HERSHEY_SIMPLEX, 0.7, color, 1, cv.LINE_AA)


def preprocess(frame, netw):
    
    blob = cv.dnn.blobFromImage(frame, 1/255, (640, 640), [0,0,0], 1, crop=False)
    netw.setInput(blob)
    outputs = netw.forward(net.getUnconnectedOutLayersNames())
    return outputs





def fetch_detection(input_image, output_data):
    class_ids = []
    confidences = []
    boxes = []

    rows = output_data[0].shape[1]

    image_width, image_height = input_image.shape[:2]

    x_factor = image_width / 640
    y_factor =  image_height / 640

    for r in range(rows):
        row = output_data[0][0][r]
        confidence = row[4]
        if confidence >= 0.45:

            classes_scores = row[5:]
            class_id = np.argmax(classes_scores)
             
            if (classes_scores[class_id] > .5):

                confidences.append(confidence)

                class_ids.append(class_id)

                x, y, w, h = row[0], row[1], row[2], row[3]
                left = int((x - w/2) * x_factor)
                top = int((y - h/2) * y_factor)
                width = int(w * x_factor)
                height = int(h * y_factor)
                box = np.array([left, top, width, height])
                boxes.append(box)

            
            

    return class_ids, confidences, boxes

outputs = preprocess(img, net)

classes, conf_, bound_ = fetch_detection(img, outputs)

indexes = cv.dnn.NMSBoxes(bound_, conf_, 0.25, 0.45) 

blob_class = []
blob_conf = []
blob_boxes = []
class_list = ["riped tomato", "unriped tomato", "diseased"]
colors = [(255, 255, 0), (0, 255, 0), (0, 255, 255), (255, 0, 0)]
for i in indexes:
    box = bound_[i]
    left = box[0]
    top = box[1]
    width = box[2]
    height = box[3] 
    blob_conf.append(conf_[i])
    blob_class.append(classes[i])
    blob_boxes.append(bound_[i])

    label = "{}:{:.2f}".format(classes[blob_class[i]], blob_conf[i])       

    draw_label(img, label, left, top, colors)


cv.imshow("pic", img)
if cv.waitKey(0)&0xFF == ord('x'):
    cv.destroyAllWindows()


