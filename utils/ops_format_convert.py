'''
detection annotation format
by wenyu
'''
import labelme
import json
import numpy as np
from xml.etree import cElementTree as ET
from PIL import Image
import os, sys
import copy
# import xmltodict

__all__ = [
    'encode_labelme',
    'decode_labelme',
    'decode_pascalvoc',
    'encode_pascalvoc',
    'labelme2pascalvoc',
]

def decode_labelme(filename=''):
    '''
    labelme json
    '''
    assert os.path.exists(filename), ''
    assert filename[-4:] == 'json', ''

    with open(filename, 'rb') as f:
        raw_data = json.load(f)
        # image = Image.fromarray(labelme.utils.img_b64_to_arr(raw_data['imageData']))
        imagePath = raw_data['imagePath']
        labels = [shape['label'] for shape in raw_data['shapes']]
        points = [shape['points'] for shape in raw_data['shapes']]

    bboxes = []
    for pts in points:
        _points = np.array(pts)
        _xmin = np.min(_points[:, 0])
        _xmax = np.max(_points[:, 0])
        _ymin = np.min(_points[:, 1])
        _ymax = np.max(_points[:, 1])
        bboxes += [[_xmin, _ymin, _xmax, _ymax], ]

    blob = {
        # 'image': image,
        'imagePath': imagePath,
        'filename': imagePath,
        'labels': labels,
        'points': points,
        'bboxes': bboxes
    }

    # blob['width'] = blob['image'].size[0]
    # blob['height'] = blob['image'].size[1]
    # blob['depth'] = len(blob['image'].mode)
    blob['classes'] = blob['labels']

    return blob


def decode_pascalvoc(filename=''):
    '''
    '''
    assert filename[-3:] == 'xml', ''

    blob = {'filename': '',
            'classes': [], 
            'bboxes': [],
            'height': 0,
            'width': 0
    }

    with open(filename, 'r') as f:
        data = ET.fromstring(f.read())

        for c in data.getchildren():

            if c.tag == 'filename':
                blob['filename'] = c.text

            elif c.tag == 'size':
                blob['height'] = int(c.find('height').text)
                blob['width'] = int(c.find('width').text)

            elif c.tag == 'object':
                
                name = c.find('name').text
                bbox =[float(x.text) for x in c.find('bndbox').getchildren()]

                blob['classes'] += [name]
                blob['bboxes'] += [bbox]
    
    assert len(blob['classes']) == len(blob['bboxes']), ''
    assert len(blob['classes']) > 0, ''
    blob['labels'] = blob['classes']

    return blob


def encode_pascalvoc(blob, file_path='', classes_map=dict):
    '''
    blob = {
        'folder': '',
        'filename': '',
        'path': '',
        
    }
    '''
    assert isinstance(classes_map, dict) or classes_map is None, ''

    new_xml = ET.Element('annotation')

    folder = ET.SubElement(new_xml, 'folder')
    folder.text = 'image'

    filename = ET.SubElement(new_xml, 'filename')
    filename.text = blob['filename']

    path = ET.SubElement(new_xml, 'path')
    path.text = blob['filename']

    source = ET.SubElement(new_xml, 'source')
    database = ET.SubElement(source, 'database')
    database.text = 'Unknown'

    size = ET.SubElement(new_xml, 'size')
    width = ET.SubElement(size, 'width')
    width.text = str(blob['width'])
    height = ET.SubElement(size, 'height')
    height.text = str(blob['height'])
    depth = ET.SubElement(size, 'depth')
    depth.text = str(blob['depth'])

    segmented = ET.SubElement(new_xml, 'segmented')
    segmented.text = '0'

    for i, bbx in enumerate(blob['bboxes']):

        objectx = ET.SubElement(new_xml, 'object')
        name = ET.SubElement(objectx, 'name')
        if classes_map is not None:
            name.text = classes_map.get(blob['classes'][i], blob['classes'][i])
        else:
            name.text = blob['classes'][i]
            
        pose = ET.SubElement(objectx, 'pose')
        pose.text = 'Unspecified'
        truncated = ET.SubElement(objectx, 'truncated')
        truncated.text = str(0)
        difficult = ET.SubElement(objectx, 'difficult')
        difficult.text = str(0)
        bndbox = ET.SubElement(objectx, 'bndbox')
        xmin = ET.SubElement(bndbox, 'xmin')
        xmin.text = str(bbx[0])
        ymin = ET.SubElement(bndbox, 'ymin')
        ymin.text = str(bbx[1])
        xmax = ET.SubElement(bndbox, 'xmax')
        xmax.text = str(bbx[2])
        ymax = ET.SubElement(bndbox, 'ymax')
        ymax.text = str(bbx[3])  

    et = ET.ElementTree(new_xml)
    et.write(file_path, encoding='utf-8', xml_declaration=True)
    # ET.dump(new_xml)


def labelme2pascalvoc(filename='', save_dir='.', classes_map=None, save_img=True):
    '''
    labelme json to pascalvoc xml
    '''

    blob = decode_labelme(filename)

    basename = os.path.basename(filename).replace('.json', '.xml')
    xmlname = os.path.join(save_dir, basename)

    if save_img:
        jpgname = os.path.join(save_dir, xmlname.replace('.xml', '.jpg'))
        blob['image'].save(jpgname)

    encode_pascalvoc(blob, xmlname, classes_map)


def encode_labelme(blob, path):
    '''
    '''
    filename = os.path.join(os.path.dirname(os.path.abspath(__file__)),'001.json')
    with open(filename, 'r') as f:
        raw_data = json.load(f)        

    shapes = []
    for i in range(len(blob['bboxes'])):
        shapes_template = copy.deepcopy(raw_data['shapes'][0])
        shapes_template['points'] = [[232.7, 301.2], [276, 301], [276.3, 334.5], [232, 334]]
        shapes_template['label'] = blob['labels'][i]
        shapes += [shapes_template]

    raw_data['shapes'] = shapes
    raw_data['imageData'] = None
    raw_data['imagePath'] = blob['path']

    with open(path, 'w') as f:
        json.dump(raw_data, f)



if __name__ == '__main__':


    filename = os.path.join(os.path.dirname(os.path.abspath(__file__)),'001.json')
    with open(filename, 'r') as f:
        raw_data = json.load(f)        