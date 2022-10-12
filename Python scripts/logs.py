import xml.etree.ElementTree as ET

FILE_NAME = "logs/M1.xes"


def get_traces(filename=FILE_NAME):
    traces_dict = {}
    tree = ET.parse(filename)
    root = tree.getroot()

    for trace in root.findall('trace'):
        trace_id = trace.find("string").attrib['value']
        traces_dict[trace_id] = []
        for attribute in trace.findall(".//event/string[2]"):
            traces_dict[trace_id].append(attribute.attrib['value'])
    return traces_dict


def make_log(trace_id, file_name, file_name_title):
    traces_dict = get_traces()
    trace_idx_last = len(list(traces_dict.keys()))

    tree = ET.parse(file_name)
    root = tree.getroot()
    s = ".//*[@value='instance_{}']..".format(trace_id)
    trace_xml = root.findall(s)
    traces_xml = root.findall("./trace")

    for i in range(0, round(trace_idx_last/4)):
        xml_root = ET.Element(trace_xml[0].tag)
        xml_string = ET.SubElement(
            xml_root, 'string', key='concept:name', value="instance_"+str(trace_idx_last+i))
        xml_string = ET.SubElement(
            xml_root, 'string', key='LogType', value='MXML.EnactmentLog')
        xml_events = trace_xml[0].findall("./event")
        for event in xml_events:
            xml_event = ET.SubElement(xml_root, 'event')
            for val in event:
                xml_attrib = ET.SubElement(
                    xml_event, val.tag, key=val.attrib['key'], value=val.attrib['value'])

        root.append(xml_root)
        root.remove(traces_xml[i])

    root.findall("./*[@key='concept:name']")[0].set('value', file_name_title)

    tree.write('logs/'+file_name_title, encoding="UTF-8")

trace_id_min = 'instance_291'
trace_id_max = 'instance_383'

make_log(int(trace_id_min[9:]),FILE_NAME,
             'M1_simulated_short.xes')
make_log(int(trace_id_max[9:]), FILE_NAME, 'M1_simulated_long.xes')

if __name__ == 'main':
    trace_id_min = 'instance_291'
    trace_id_max = 'instance_383'

    make_log(int(trace_id_min[9:]),
             'M1_simulated_short.xes', "short trace log")
    make_log(int(trace_id_max[9:]), 'M1_simulated_long.xes', "long trace log")

    # tree = ET.parse("logs/M1_sim_long.xes")
    # root = tree.getroot()
    # len_traces_long = len(get_traces())
    # print(len_traces_long)

    # tree = ET.parse("logs/M1_sim_short.xes")
    # root = tree.getroot()
    # len_traces_short = len(get_traces())
    # print(len_traces_short)
