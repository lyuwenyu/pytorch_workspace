from collections import OrderedDict


label_color = OrderedDict([
    # name: (r, g, b)
    ['c1', (0, 0, 0)],
    ['c2', (128, 0, 0)],
    ['c3', (0, 128, 0)],
    ['c4', (0, 0, 128)],
    ['c5', (128, 128, 0)],

])


num_classes = len(label_color)
label_id_map = {k: i for i, (k, _) in enumerate(label_color.items())}
label_id_color = { v: label_color[k] for k, v in label_id_map.items()}


if __name__ == '__main__':
    print(num_classes)
    print(label_id_map)
    print(label_id_color)