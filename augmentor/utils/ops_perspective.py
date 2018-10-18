from PIL import Image, ImageDraw
import numpy as np
import random
from math import floor, ceil
import numpy
import cv2
from ops_parse_json import load_json
from shapely import geometry

def find_coeffs(pa, pb):
    matrix = []
    for p1, p2 in zip(pa, pb):
        matrix.append([p1[0], p1[1], 1, 0, 0, 0, -p2[0]*p1[0], -p2[0]*p1[1]])
        matrix.append([0, 0, 0, p1[0], p1[1], 1, -p2[1]*p1[0], -p2[1]*p1[1]])

    A = numpy.matrix(matrix, dtype=numpy.float)
    B = numpy.array(pb).reshape(8)

    res = numpy.dot(numpy.linalg.inv(A.T * A) * A.T, B)
    return numpy.array(res).reshape(8)


def perspective_operation(image, bboxes=None, magnitude=2.0, skew_type='RANDOM'):

    # if not isinstance(image, list):
    #     images = [images]
    w, h = image.size

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
    # M_reverse = np.linalg.pinv(M)

    def do(image, bbox=None):
        # return image.transform(image.size,
        #                         Image.PERSPECTIVE,
        #                         perspective_skew_coefficients_matrix,
        #                         resample=Image.BICUBIC)
        _img = cv2.warpPerspective(np.array(image), M, (0, 0))
        _img = Image.fromarray(_img)
        bbox = np.array(bbox)
        points = bbox.reshape(-1, 2)
        points = np.hstack([points, np.ones((len(points), 1), dtype=np.float32)])
        # print(points.shape)

        points = np.dot(points, M.T)
        points[:, 0] = points[:, 0] / (points[:, -1] + 1e-10)
        points[:, 1] = points[:, 1] / (points[:, -1] + 1e-10)
        points = points[:, :2]
        print(points.shape)
        # points = np.hstack([points[:int(len(points)/2), :2], points[int(len(points)/2):, :2]])
        # points[:, 0] = np.minimum(np.maximum(0, points[:, 0]), _img.size[0] - 1)
        # points[:, 1] = np.minimum(np.maximum(0, points[:, 1]), _img.size[1] - 1)
        # points[:, 2] = np.minimum(np.maximum(0, points[:, 2]), _img.size[0] - 1)
        # points[:, 3] = np.minimum(np.maximum(0, points[:, 3]), _img.size[1] - 1)
        print(points[list(range(0, len(points), 2)), 0])

        points[:, 0] = np.minimum(np.maximum(points[:, 0], 0), _img.size[0] - 1)
        points[:, 1] = np.minimum(np.maximum(points[:, 1], 0), _img.size[1] - 1)
        points = points[:, :2].reshape(bbox.shape)
        

        for i, bbx in enumerate(points):
            polygon = geometry.Polygon([(x,y) for x, y in zip(bbx[::2], bbx[1::2])])
            if polygon.area < 100:
                print(i)
                pass
        
        # areas = (points[:, 3] - points[:, 1]) * (points[:, 2] - points[:, 0])
        # index = np.where(areas > 100)[0]
        # if len(index) == 0:
        #     print('no objects...')
        #     return _img, None
        # return _img, [list(lin) for lin in points[index]]
        return _img, points

    # augmented_images = []
    # for image in images:
    #     augmented_images.append(do(image))
    
    return do(image, bbox=bboxes)



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


if __name__ == '__main__':

    blob = load_json('./00001.json')
    # print(blob)
    
    img = blob['imageData']
    bbox = np.array(blob['points'])
    draw = ImageDraw.Draw(img)
    for lin in bbox:
        draw.polygon(tuple(lin), outline='red')
    img.show()

    # bbox = bbox.reshape(-1, 2)
    img, points = perspective_operation(img, bbox)
    draw = ImageDraw.Draw(img)

    points = np.array(points).reshape(-1, 2)
    for pt in points:
        draw.ellipse((pt[0]-3, pt[1]-3, pt[0]+3, pt[1]+3), fill='red')
    img.show()

    # print(points)