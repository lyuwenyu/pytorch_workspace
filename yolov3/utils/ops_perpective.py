'''
by lyuwenyu
2018/9/7
'''

from PIL import Image
import numpy as np
import random
from math import floor, ceil
import numpy
import cv2


def find_coeffs(pa, pb):
    matrix = []
    for p1, p2 in zip(pa, pb):
        matrix.append([p1[0], p1[1], 1, 0, 0, 0, -p2[0]*p1[0], -p2[0]*p1[1]])
        matrix.append([0, 0, 0, p1[0], p1[1], 1, -p2[1]*p1[0], -p2[1]*p1[1]])

    A = numpy.matrix(matrix, dtype=numpy.float)
    B = numpy.array(pb).reshape(8)

    res = numpy.dot(numpy.linalg.inv(A.T * A) * A.T, B)
    return numpy.array(res).reshape(8)


def perspective_operation(images, magnitude, skew_type):

    if not isinstance(images, list):
        images = [images]
        
    w, h = images[0].size

    x1 = 0
    x2 = h
    y1 = 0
    y2 = w

    original_plane = [(y1, x1), (y2, x1), (y2, x2), (y1, x2)]

    max_skew_amount = max(w, h)
    max_skew_amount = int(ceil(max_skew_amount * magnitude))
    skew_amount = random.randint(1, max_skew_amount)


    if skew_type == "RANDOM":
        skew = random.choice(["TILT", "TILT_LEFT_RIGHT", "TILT_TOP_BOTTOM", "CORNER"])
    else:
        skew = skew_type

    # We have two choices now: we tilt in one of four directions
    # or we skew a corner.

    if skew == "TILT" or skew == "TILT_LEFT_RIGHT" or skew == "TILT_TOP_BOTTOM":

        if skew == "TILT":
            skew_direction = random.randint(0, 3)
        elif skew == "TILT_LEFT_RIGHT":
            skew_direction = random.randint(0, 1)
        elif skew == "TILT_TOP_BOTTOM":
            skew_direction = random.randint(2, 3)

        # skew_direction = 0

        if skew_direction == 0:
            # Left Tilt
            new_plane = [(y1, x1 - skew_amount),  # Top Left
                            (y2, x1),                # Top Right
                            (y2, x2),                # Bottom Right
                            (y1, x2 + skew_amount)]  # Bottom Left

        elif skew_direction == 1:
            # Right Tilt
            new_plane = [(y1, x1),                # Top Left
                            (y2, x1 - skew_amount),  # Top Right
                            (y2, x2 + skew_amount),  # Bottom Right
                            (y1, x2)]                # Bottom Left
        elif skew_direction == 2:
            # Forward Tilt
            new_plane = [(y1 - skew_amount, x1),  # Top Left
                            (y2 + skew_amount, x1),  # Top Right
                            (y2, x2),                # Bottom Right
                            (y1, x2)]                # Bottom Left
        elif skew_direction == 3:
            # Backward Tilt
            new_plane = [(y1, x1),                # Top Left
                            (y2, x1),                # Top Right
                            (y2 + skew_amount, x2),  # Bottom Right
                            (y1 - skew_amount, x2)]  # Bottom Left

    if skew == "CORNER":

        skew_direction = random.randint(0, 7)

        if skew_direction == 0:
            # Skew possibility 0
            new_plane = [(y1 - skew_amount, x1), (y2, x1), (y2, x2), (y1, x2)]
        elif skew_direction == 1:
            # Skew possibility 1
            new_plane = [(y1, x1 - skew_amount), (y2, x1), (y2, x2), (y1, x2)]
        elif skew_direction == 2:
            # Skew possibility 2
            new_plane = [(y1, x1), (y2 + skew_amount, x1), (y2, x2), (y1, x2)]
        elif skew_direction == 3:
            # Skew possibility 3
            new_plane = [(y1, x1), (y2, x1 - skew_amount), (y2, x2), (y1, x2)]
        elif skew_direction == 4:
            # Skew possibility 4
            new_plane = [(y1, x1), (y2, x1), (y2 + skew_amount, x2), (y1, x2)]
        elif skew_direction == 5:
            # Skew possibility 5
            new_plane = [(y1, x1), (y2, x1), (y2, x2 + skew_amount), (y1, x2)]
        elif skew_direction == 6:
            # Skew possibility 6
            new_plane = [(y1, x1), (y2, x1), (y2, x2), (y1 - skew_amount, x2)]
        elif skew_direction == 7:
            # Skew possibility 7
            new_plane = [(y1, x1), (y2, x1), (y2, x2), (y1, x2 + skew_amount)]

    if skew_type == "ALL":

        corners = dict()
        corners["top_left"] = (y1 - random.randint(1, skew_amount), x1 - random.randint(1, skew_amount))
        corners["top_right"] = (y2 + random.randint(1, skew_amount), x1 - random.randint(1, skew_amount))
        corners["bottom_right"] = (y2 + random.randint(1, skew_amount), x2 + random.randint(1, skew_amount))
        corners["bottom_left"] = (y1 - random.randint(1, skew_amount), x2 + random.randint(1, skew_amount))

        new_plane = [corners["top_left"], corners["top_right"], corners["bottom_right"], corners["bottom_left"]]


    # matrix = []
    # for p1, p2 in zip(new_plane, original_plane):
    #     matrix.append([p1[0], p1[1], 1, 0, 0, 0, -p2[0] * p1[0], -p2[0] * p1[1]])
    #     matrix.append([0, 0, 0, p1[0], p1[1], 1, -p2[1] * p1[0], -p2[1] * p1[1]])

    # A = np.matrix(matrix, dtype=np.float)
    # B = np.array(original_plane).reshape(8)
    # perspective_skew_coefficients_matrix = np.dot(np.linalg.pinv(A), B)
    # perspective_skew_coefficients_matrix = np.array(perspective_skew_coefficients_matrix).reshape(8)

    ## ---
    # matrix_reverse = []
    # for p1, p2 in zip(reverse_plane, original_plane):
    #     matrix_reverse.append([p1[0], p1[1], 1, 0, 0, 0, -p2[0] * p1[0], -p2[0] * p1[1]])
    #     matrix_reverse.append([0, 0, 0, p1[0], p1[1], 1, -p2[1] * p1[0], -p2[1] * p1[1]])

    M = cv2.getPerspectiveTransform(np.array(original_plane).astype(np.float32), np.array(new_plane).astype(np.float32))
    M_reverse = np.linalg.pinv(M)

    def do(image):
        # return image.transform(image.size,
        #                         Image.PERSPECTIVE,
        #                         perspective_skew_coefficients_matrix,
        #                         resample=Image.BICUBIC)
        corrected_image = cv2.warpPerspective(np.array(image), M, (0, 0))
        return Image.fromarray(corrected_image)

    augmented_images = []

    for image in images:
        augmented_images.append(do(image))

    ###---
    # A = np.matrix(matrix, dtype=np.float)
    # B = np.array(original_plane).reshape(8)
    # perspective_skew_coefficients_matrix = np.dot(np.linalg.pinv(A), B)
    # perspective_skew_coefficients_matrix = np.array(perspective_skew_coefficients_matrix).reshape(8)

    # tmp = do(augmented_images[0])
    # tmp.show()
    # print('A :', A)
    # print('B :', B)
    # print('origin plane: ', original_plane)
    # print('new plane: ', new_plane)
    # print('matrix: ', perspective_skew_coefficients_matrix)

    # M = cv2.getPerspectiveTransform(np.array(original_plane).astype(np.float32), np.array(new_plane).astype(np.float32))
    # corrected_image = cv2.warpPerspective(np.array(images[0]), M, (0, 0))
    # _tmp = Image.fromarray(corrected_image)
    # _tmp.show()

    # M_revs = np.linalg.pinv(M)
    # corrected_image = cv2.warpPerspective(np.array(corrected_image), M_revs, (0, 0))
    # _tmp = Image.fromarray(corrected_image)
    # _tmp.show()

    return augmented_images[0], M, M_reverse


def NMS(dets, threshold):
    
    # dets = np.array(dets)

    # res = dets[:, 1:]
    # y1 = dets[:, 2]
    # x1 = dets[:, 3]
    # y2 = dets[:, 4]
    # x2 = dets[:, 5]
    x1 = dets[:, 2]
    y1 = dets[:, 3]
    x2 = dets[:, 4]
    y2 = dets[:, 5]
    scores = dets[:, 1]

    areas = (x1 - x2 + 1) * (y1 - y2 + 1)
    order = np.argsort(scores)[::-1]
    keep = []
    while len(order) > 0:
        i = order[0]
        keep += [i]
        
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        overlap = w * h
        ovr = overlap / (areas[i] + areas[order[1:]] - overlap)

        inds = np.where(ovr <= threshold)[0]
        order = order[inds + 1]

    return list(map(int, keep))
