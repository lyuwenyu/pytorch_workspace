from xml.etree import cElementTree as ET 
from collections import OrderedDict

with open('1.xml', 'r') as f:
    raw_data = ET.fromstring(f.read())
    # raw_data = ET.parse('1.xml')
    # raw_data = raw_data.getroot()
    # print(dir(raw_data))
    # print(len(raw_data))

    data = []
    
    for c in raw_data.getchildren():

        if c.tag != 'object':
            continue

        bbox = c.find('bndbox')
        name = c.find('name').text

        obj = {}
        obj[name] = {}

        for i in bbox.getchildren():
            obj[name][i.tag] = int(i.text)

        data += [obj]

    print(data)