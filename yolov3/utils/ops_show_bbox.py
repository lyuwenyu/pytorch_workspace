from PIL import Image
from PIL import ImageDraw


def show_bbox(img, bbox=None, names=None):

    draw = ImageDraw.Draw(img)
    print(bbox)
    for bb in bbox:
        draw.rectangle(tuple(bb), outline='red')
    img.show()