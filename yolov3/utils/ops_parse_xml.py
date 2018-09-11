from xml.etree import cElementTree as ET 
import os

def parse_xml(path):
    
    blob = {'filename': '',
            'names': [], 
            'bboxes': [],
            'height': 0,
            'width': 0
    }

    with open(path, 'r') as f:
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

                blob['names'] += [name]
                blob['bboxes'] += [bbox]

    return blob