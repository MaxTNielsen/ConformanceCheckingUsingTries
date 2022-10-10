import xml.etree.ElementTree as ET

FILE_NAME = "logs/M1.xes"

def get_traces():
    traces_dict = {}
    tree = ET.parse(FILE_NAME)
    root = tree.getroot()
    
    for trace in root.findall('trace'):
        trace_id = trace.find("string").attrib['value']
        traces_dict[trace_id] = []
        for attribute in trace.findall(".//event/string[2]"):
            traces_dict[trace_id].append(attribute.attrib['value'])
    return traces_dict

trace_id_min = 'instance_291'
trace_id_max = 'instance_383'

traces_dict = get_traces()
trace_idx_last = len(list(traces_dict.keys()))

def make_log(trace_id, file_name, title):
    tree = ET.parse(FILE_NAME)
    root = tree.getroot()
    s = ".//*[@value='instance_{}']..".format(trace_id)
    trace_xml = root.findall(s)[0]

    for i in range(0,round(trace_idx_last/4)):
        xml_root = ET.Element(trace_xml.tag)
        xml_string = ET.SubElement(xml_root, 'string', key='concept:name', value="instance_"+str(trace_idx_last+i))
        xml_string = ET.SubElement(xml_root, 'string', key='LogType', value='MXML.EnactmentLog')
        xml_events = trace_xml.findall("./event")
        for event in xml_events:
            xml_event = ET.SubElement(xml_root, 'event')
            for val in event:
                xml_attrib = ET.SubElement(xml_event, val.tag, key=val.attrib['key'], value=val.attrib['value'])
        
        root.append(xml_root)

    root.findall("./*[@key='concept:name']")[0].set('value', title)

    tree.write('logs/'+file_name, encoding="UTF-8")

make_log(int(trace_id_min[9:]), 'M1_sim_short.xes', "short trace log")
make_log(int(trace_id_max[9:]), 'M1_sim_long.xes', "long trace log")

# tree = ET.parse("logs/M1_sim_long.xes")
# root = tree.getroot()
# len_traces_long = len(get_traces())
# print(len_traces_long)

# tree = ET.parse("logs/M1_sim_short.xes")
# root = tree.getroot()
# len_traces_short = len(get_traces())
# print(len_traces_short)