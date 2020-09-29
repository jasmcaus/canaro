import cv2 as cv
import numpy as np

net = cv.dnn.readNetFromTensorflow(r'C:\Users\aus\Documents\Code Projects\PyPi\kangeras\model_pb/saved_model.pb')

img = cv.imread(r'C:\Users\aus\Downloads\bart.jpg')
img = cv.resize(img, (80,80))
img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

blob = cv.dnn.blobFromImage(img, scalefactor=1.0)

net.setInput(blob)
preds = net.forward()

classes = ['Homer simpson',
 'Ned flanders',
 'Moe szyslak',
 'Lisa simpson',
 'Bart simpson',
 'Marge simpson',
 'Krusty the clown',
 'Principal skinner',
 'Charles montgomery burns',
 'Milhouse van houten']

idx = np.argsort(preds[0])[::-1][0]
text = "Label: {}, {:.2f}%".format(classes[idx],
	preds[0][idx] * 100)
cv2.putText(image, text, (5, 25),  cv2.FONT_HERSHEY_SIMPLEX,
	0.7, (0, 0, 255), 2)

# show the output image
cv2.imshow("Image", image)
cv2.waitKey(0)




