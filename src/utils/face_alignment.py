import argparse
import bob.io.base
import bob.io.base.test_utils
import bob.io.image
import bob.ip.facedetect
from bob.ip.skincolorfilter import SkinColorFilter
import cv2
import numpy as np
from PIL import Image
import dlib
import matplotlib.pyplot as plt

def plot_image(img):
    plt.axis("off")
    plt.imshow(img, origin='upper')
    plt.show()

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', type=int, default=0,
                        help='0 if opencv mode else dlib ')
    parser.add_argument('--path_to_load', type=str, required=False, help='path to load original image from')
    parser.add_argument('--path_to_save',type=str,required=False,help='path to save rotated image to')
    parser.add_argument('--show',type=bool,default=False, help='show result or not')
    parser.add_argument('--rotation_mode',type=int,default=1,help='0 if Pillow rotation , 1 if opencv')
    args = parser.parse_args()
    return args


def load_img(path):
    img = cv2.imread(path)
    return img


def draw_predict(frame, left, top, right, bottom):
    cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 3)


def get_eyes_nose_dlib(shape):
    nose = shape[4][1]
    left_eye_x = int(shape[3][1][0] + shape[2][1][0]) // 2
    left_eye_y = int(shape[3][1][1] + shape[2][1][1]) // 2
    right_eyes_x = int(shape[1][1][0] + shape[0][1][0]) // 2
    right_eyes_y = int(shape[1][1][1] + shape[0][1][1]) // 2
    return nose, (left_eye_x, left_eye_y), (right_eyes_x, right_eyes_y)


def get_eyes_nose(eyes, nose):
    left_eye_x = int(eyes[0][0] + eyes[0][2] / 2)
    left_eye_y = int(eyes[0][1] + eyes[0][3] / 2)
    right_eye_x = int(eyes[1][0] + eyes[1][2] / 2)
    right_eye_y = int(eyes[1][1] + eyes[1][3] / 2)
    nose_x = int(nose[0][0] + nose[0][2] / 2)
    nose_y = int(nose[0][1] + nose[0][3] / 2)

    return (nose_x, nose_y), (right_eye_x, right_eye_y), (left_eye_x, left_eye_y)


def rotate_point(origin, point, angle):
    ox, oy = origin
    px, py = point

    qx = ox + np.cos(angle) * (px - ox) - np.sin(angle) * (py - oy)
    qy = oy + np.sin(angle) * (px - ox) + np.cos(angle) * (py - oy)
    return qx, qy


def is_between(point1, point2, point3, extra_point):
    c1 = (point2[0] - point1[0]) * (extra_point[1] - point1[1]) - (point2[1] - point1[1]) * (extra_point[0] - point1[0])
    c2 = (point3[0] - point2[0]) * (extra_point[1] - point2[1]) - (point3[1] - point2[1]) * (extra_point[0] - point2[0])
    c3 = (point1[0] - point3[0]) * (extra_point[1] - point3[1]) - (point1[1] - point3[1]) * (extra_point[0] - point3[0])
    if (c1 < 0 and c2 < 0 and c3 < 0) or (c1 > 0 and c2 > 0 and c3 > 0):
        return True
    else:
        return False


def distance(a, b):
    return np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def cosine_formula(length_line1, length_line2, length_line3):
    cos_a = -(length_line3 ** 2 - length_line2 ** 2 - length_line1 ** 2) / (2 * length_line2 * length_line1)
    return cos_a


def show_img(img):
    while True:
        cv2.imshow('face_alignment_app', img)
        c = cv2.waitKey(1)
        if c == 27:
            break
    cv2.destroyAllWindows()


def shape_to_normal(shape):
    shape_normal = []
    for i in range(0, 5):
        shape_normal.append((i, (shape.part(i).x, shape.part(i).y)))
    return shape_normal


def rotate_opencv(img, nose_center, angle):
    M = cv2.getRotationMatrix2D(nose_center, angle, 1)
    rotated = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]), flags=cv2.INTER_CUBIC)
    return rotated


def rotation_detection_dlib(img, mode, show=False):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('shape_predictor_5_face_landmarks.dat')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 0)
    if len(rects) > 0:
        for rect in rects:
            x = rect.left()
            y = rect.top()
            w = rect.right()
            h = rect.bottom()
            shape = predictor(gray, rect)
            shape = shape_to_normal(shape)
            nose, left_eye, right_eye = get_eyes_nose_dlib(shape)
            center_of_forehead = ((left_eye[0] + right_eye[0]) // 2, (left_eye[1] + right_eye[1]) // 2)
            center_pred = (int((x + w) / 2), int((y + y) / 2))
            length_line1 = distance(center_of_forehead, nose)
            length_line2 = distance(center_pred, nose)
            length_line3 = distance(center_pred, center_of_forehead)
            cos_a = cosine_formula(length_line1, length_line2, length_line3)
            angle = np.arccos(cos_a)
            rotated_point = rotate_point(nose, center_of_forehead, angle)
            rotated_point = (int(rotated_point[0]), int(rotated_point[1]))
            if is_between(nose, center_of_forehead, center_pred, rotated_point):
                angle = np.degrees(-angle)
            else:
                angle = np.degrees(angle)

            if mode:
                img = rotate_opencv(img, nose, angle)
            else:
                img = Image.fromarray(img)
                img = np.array(img.rotate(angle))
        if show:
            show_img(img)
        return img
    else:
        return img


def rotation_detection_opencv(img, mode, show=False):
    nose_cascade = cv2.CascadeClassifier('./haarcascade_mcs_nose.xml')
    eyes_cascade = cv2.CascadeClassifier('./haarcascade_eye.xml')
    fase_cascade = cv2.CascadeClassifier('./haarcascade_frontalface_alt2.xml')
    # gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray = img
    nose_rects = nose_cascade.detectMultiScale(gray, 1.3, 5)
    eyes_rects = eyes_cascade.detectMultiScale(gray, 1.3, 5)
    face_rects = fase_cascade.detectMultiScale(gray, 1.3, 5)
    length_eyes = len(eyes_rects)

    if length_eyes == 2 and len(nose_rects) != 0 and len(face_rects) != 0:
        nose, right_eye, left_eye = get_eyes_nose(eyes_rects, nose_rects)
    else:
        print("Couldn't determine eyes/nose")
        return img
    center_of_forehead = (int((right_eye[0] + left_eye[0]) / 2), int((right_eye[1] + left_eye[1]) / 2))
    center_pred = (int((face_rects[0][0] + face_rects[0][2]) / 2), int((face_rects[0][1] + face_rects[0][1]) / 2))
    length_line1 = distance(center_of_forehead, nose)
    length_line2 = distance(center_pred, nose)
    length_line3 = distance(center_pred, center_of_forehead)
    cos_a = cosine_formula(length_line1, length_line2, length_line3)
    angle = np.arccos(cos_a)
    rotated_point = rotate_point(nose, center_of_forehead, angle)
    rotated_point = (int(rotated_point[0]), int(rotated_point[1]))
    if is_between(nose, center_of_forehead, center_pred, rotated_point):
        angle = np.degrees(-angle)
    else:
        angle = np.degrees(angle)
    if mode:
        img = rotate_opencv(img, nose, angle)
    else:
        img = Image.fromarray(img)
        img = np.array(img.rotate(angle))
    if show:
        show_img(img)
    return img


def save_img(path, img):
    cv2.imwrite(path, img)


# def align_face(img, args):
#     # img = load_img(args.path_to_load)
#     if args.mode == 0:
#         img = rotation_detection_opencv(img, args.rotation_mode, args.show)
#     else:
#         img = rotation_detection_dlib(img, args.rotation_mode,args.show)
#
#     return img
#     # save_img(args.path_to_save, img)
from collections import OrderedDict
#For dlib’s 68-point facial landmark detector:

FACIAL_LANDMARKS_68_IDXS = OrderedDict([
	("mouth", (48, 68)),
	("inner_mouth", (60, 68)),
	("right_eyebrow", (17, 22)),
	("left_eyebrow", (22, 27)),
	("right_eye", (36, 42)),
	("left_eye", (42, 48)),
	("nose", (27, 36)),
	("jaw", (0, 17)),
    ("left_cheek", (0, 17)),
    ("right_cheek", (0, 17))
])

def shape_to_np(shape, dtype="int"):
    # initialize the list of (x, y)-coordinates
    coords = np.zeros((shape.num_parts, 2), dtype=dtype)

    # loop over all facial landmarks and convert them
    # to a 2-tuple of (x, y)-coordinates
    for i in range(0, shape.num_parts):
        coords[i] = (shape.part(i).x, shape.part(i).y)

    # return the list of (x, y)-coordinates
    return coords


def align_face(img, args):
    # simple hack ;)
    # detector = dlib.get_frontal_face_detector()
    desiredLeftEye=(0.35, 0.35)
    desiredFaceWidth = 180
    desiredFaceHeight = 180
    predictor = dlib.shape_predictor("./shape_predictor_68_face_landmarks.dat")
    shape = predictor(img, dlib.rectangle(0, 0, img.shape[0], img.shape[1]))
    shape = shape_to_np(shape)
    # if (len(shape) == 68):
    # extract the left and right eye (x, y)-coordinates
    (lStart, lEnd) = FACIAL_LANDMARKS_68_IDXS["left_eye"]
    (rStart, rEnd) = FACIAL_LANDMARKS_68_IDXS["right_eye"]
    # else:
    #     (lStart, lEnd) = FACIAL_LANDMARKS_5_IDXS["left_eye"]
    #     (rStart, rEnd) = FACIAL_LANDMARKS_5_IDXS["right_eye"]

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
    desiredRightEyeX = 1.0 - desiredLeftEye[0]

    # determine the scale of the new resulting image by taking
    # the ratio of the distance between eyes in the *current*
    # image to the ratio of distance between eyes in the
    # *desired* image
    dist = np.sqrt((dX ** 2) + (dY ** 2))
    desiredDist = (desiredRightEyeX - desiredLeftEye[0])
    desiredDist *= desiredFaceWidth
    scale = desiredDist / dist

    # compute center (x, y)-coordinates (i.e., the median point)
    # between the two eyes in the input image
    eyesCenter = ((leftEyeCenter[0] + rightEyeCenter[0]) // 2,
                  (leftEyeCenter[1] + rightEyeCenter[1]) // 2)

    # grab the rotation matrix for rotating and scaling the face
    M = cv2.getRotationMatrix2D(eyesCenter, angle, scale)

    # update the translation component of the matrix
    tX = desiredFaceWidth * 0.5
    tY = desiredFaceHeight * desiredLeftEye[1]
    M[0, 2] += (tX - eyesCenter[0])
    M[1, 2] += (tY - eyesCenter[1])

    # apply the affine transformation
    (w, h) = (desiredFaceWidth, desiredFaceHeight)
    output = cv2.warpAffine(img, M, (w, h),
                            flags=cv2.INTER_CUBIC)

    fase_cascade = cv2.CascadeClassifier('./haarcascade_frontalface_alt2.xml')
    face_rects = fase_cascade.detectMultiScale(output, 1.3, 5)
    try:
        (x, y, w, h) = face_rects[0]
        output = output[y:(y + h), x:(x + w)]
    except:
        output = output
        # cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 2)
    # return the aligned face
    return output


def get_skin_segmentation_mask(image):
    # face_image = image.transpose(2, 0, 1)
    face_image = image.reshape((3, image.shape[0], image.shape[1]))
    # face_image = cv2.imread("./detect_skin_pixels_00.png")
    # face_image = face_image.reshape(3, face_image.shape[0], face_image.shape[1])
    # detection = bob.ip.facedetect.detect_single_face(face_image)
    bounding_box, quality = bob.ip.facedetect.detect_single_face(face_image)
    face = face_image[:, bounding_box.top:bounding_box.bottom, bounding_box.left:bounding_box.right]
    skin_filter = SkinColorFilter()
    skin_filter.estimate_gaussian_parameters(face)
    for thresh in [0.1, 0.2, 0.3, 0.4, 0.5]:
        skin_mask = skin_filter.get_skin_mask(face_image, thresh)
        skin_image = np.copy(face_image)
        skin_image[:, np.logical_not(skin_mask)] = 0
        skin_image_cropped = skin_image[:, bounding_box.top:bounding_box.bottom, bounding_box.left:bounding_box.right]
        plot_image(skin_image.reshape((image.shape[0],image.shape[1],3)))
    return skin_image_cropped
