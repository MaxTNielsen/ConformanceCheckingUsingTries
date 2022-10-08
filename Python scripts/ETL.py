import xml.etree.ElementTree as ET
from datetime import datetime

FILE_NAME = "logs/M1_sim.xes"

def read_from_file():
    tree = ET.parse(FILE_NAME)
    root = tree.getroot()
    traces_dict = {}
    
    for trace in root.findall('trace'):
        for val in trace:
        #traces_dict[trace.find('{http://www.xes-standard.org/concept.xesext}concept:name').get('value')] = []
            print(val.tag, val.attrib)
    #     for event in trace.findall('{http://www.xes-standard.org/}event'):
    #         e = {}
    #         for attributes in event:
    #             if attributes.attrib['key'] == 'time:timestamp':
    #                 e[attributes.attrib['key']] = datetime.fromisoformat(attributes.attrib['value']).replace(tzinfo=None)
    #                 continue
    #             if attributes.attrib['key'] == 'cost':
    #                 e[attributes.attrib['key']] = int(attributes.attrib['value'])
    #             else:
    #                 e[attributes.attrib['key']] = attributes.attrib['value']
    #         traces_dict[trace.find('{http://www.xes-standard.org/}string').get('value')].append(e)
    # return traces_dict


print(read_from_file())