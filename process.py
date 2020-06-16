import os
import numpy as np
import cv2
from PIL import Image
import shutil
from collections import OrderedDict
import dlib
import imutils
import sys

FACIAL_LANDMARKS_68_IDXS = OrderedDict([
    ("mouth", (48, 68)),
    ("right_eyebrow", (17, 22)),
    ("left_eyebrow", (22, 27)),
    ("right_eye", (36, 42)),
    ("left_eye", (42, 48)),
    ("nose", (27, 36)),
    ("jaw", (0, 17))
])


def shape_to_np(shape, dtype="int"):
    coords = np.zeros((68, 2), dtype=dtype)
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords


def rect_to_bb(rect):
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y
    return (x, y, w, h)


class FaceAligner:
    def __init__(self, predictor, desiredFaceWidth, desiredFaceHeight, desiredLeftEye=(0.35, 0.35)):
        # store the facial landmark predictor, desired output left
        # eye position, and desired output face width + height
        self.predictor = predictor
        self.desiredLeftEye = desiredLeftEye
        self.desiredFaceWidth = desiredFaceWidth
        self.desiredFaceHeight = desiredFaceHeight
        # if the desired face height is None, set it to be the
        # desired face width (normal behavior)
        if self.desiredFaceHeight is None:
            self.desiredFaceHeight = self.desiredFaceWidth

    def align(self, image, gray, rect):
            # convert the landmark (x, y)-coordinates to a NumPy array
        shape = self.predictor(gray, rect)
        shape = shape_to_np(shape)
        # extract the left and right eye (x, y)-coordinates
        (lStart, lEnd) = FACIAL_LANDMARKS_68_IDXS["left_eye"]
        (rStart, rEnd) = FACIAL_LANDMARKS_68_IDXS["right_eye"]
        leftEyePts = shape[lStart:lEnd]
        rightEyePts = shape[rStart:rEnd]
        # compute the center of mass for each eye
        leftEyeCenter = leftEyePts.mean(axis=0).astype("int")
        rightEyeCenter = rightEyePts.mean(axis=0).astype("int")
        # compute the angle between the eye centroids
        dY = rightEyeCenter[1] - leftEyeCenter[1]
        dX = rightEyeCenter[0] - leftEyeCenter[0]
        angle = np.degrees(np.arctan2(dY, dX)) - 180
        # compute the desired right eye x-coordinate based on the
        # desired x-coordinate of the left eye
        desiredRightEyeX = 1.0 - self.desiredLeftEye[0]
        # determine the scale of the new resulting image by taking
        # the ratio of the distance between eyes in the *current*
        # image to the ratio of distance between eyes in the
        # *desired* image
        dist = np.sqrt((dX ** 2) + (dY ** 2))
        desiredDist = (desiredRightEyeX - self.desiredLeftEye[0])
        desiredDist *= self.desiredFaceWidth
        scale = desiredDist / (dist*1.3)
        # compute center (x, y)-coordinates (i.e., the median point)
        # between the two eyes in the input image
        eyesCenter = ((leftEyeCenter[0] + rightEyeCenter[0]) // 2,
                      (leftEyeCenter[1] + rightEyeCenter[1]) // 2)
        # grab the rotation matrix for rotating and scaling the face
        M = cv2.getRotationMatrix2D(eyesCenter, angle, scale)
        # update the translation component of the matrix
        tX = self.desiredFaceWidth*0.5
        tY = 1.2*self.desiredFaceHeight * self.desiredLeftEye[1]
        M[0, 2] += (tX - eyesCenter[0])
        M[1, 2] += (tY - eyesCenter[1])

        # apply the affine transformation
        (w, h) = (self.desiredFaceWidth, self.desiredFaceHeight)
        output = cv2.warpAffine(image, M, (w, h))
        # return the aligned face
        return output


def crop(image_path, coords, outputpath):
    orig_image_obj = Image.open(image_path)
    for index, (x, y, w, h) in enumerate(coords):
        image_obj = orig_image_obj.crop(
            (x-0.25*w, y-0.25*h, x+w+0.25*w, y+h+0.6*h))
        print(outputpath+str(index)+'_'+image_path)
        image_obj.save(outputpath.replace(".jpg", '_'+str(index)+'_.jpg'))


def cropandalign(image_path, rects, outputpath, fa, gray):
    image = cv2.imread(image_path)
    b, g, r = cv2.split(image)
    image = cv2.merge([r, g, b])
    # loop over the face detections
    i = 0
    for rect in rects:
        faceAligned = fa.align(image, gray, rect)
        # save the output images
        alignedImage = Image.fromarray(faceAligned)
        alignedImage.save(outputpath.replace(".jpg", '_'+str(i)+'_.jpg'))
        i += 1
        print(outputpath)


detector = dlib.get_frontal_face_detector()

predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
fa = FaceAligner(predictor, desiredFaceWidth=512,
                 desiredFaceHeight=512)

rootdir = os.getcwd()+'\\test_browse\\downloads2\\'
outputpath = os.getcwd()+'\\test_browse\\output\\'

for subdir, dirs, files in os.walk(rootdir):
    structure = os.path.join(outputpath, subdir[len(rootdir):])
    if not os.path.isdir(structure):
        os.mkdir(structure)

for subdir, dirs, files in os.walk(rootdir):
    for file in files:
        try:
            image = cv2.imread(os.path.join(subdir, file))
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            rects = detector(gray, 2)
            cropandalign(str(os.path.join(subdir, file)), rects, str(os.path.join(
                subdir.replace(rootdir, outputpath), file)), fa, gray)
        except:
            # manage excetions here
            path = os.path.join(subdir, file)
            shutil.copy(path, path.replace(rootdir, outputpath))
