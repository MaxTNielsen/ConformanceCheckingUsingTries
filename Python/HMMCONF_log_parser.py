import pandas as pd
import os
import sys
import csv
from pandas.api.types import CategoricalDtype
import xml.etree.ElementTree as ET
import re


FIELDS = ['T:concept:name', 'E:concept:name', 'id']
BASE_PATH = os.path.join('output', 'hmmconf')
INDIR = os.path.join(BASE_PATH, 'csv_logs')
OUTDIR = os.path.join(BASE_PATH, 'hmmconf_logs')


def preprocess_logs(filep: str, log_type: str):
    for fp in os.listdir('./'+filep):
        traces = list()
        tree = ET.parse(filep+fp)
        root = tree.getroot()

        if log_type == 'M-model':
            m = re.search(r'completeness|confidence', fp)
            tag = '{http://www.xes-standard.org/}' if m else ''
            traces = get_traces(root=root, nspace=tag)
        else:
            tag = '{http://www.xes-standard.org/}'
            traces = get_traces(root=root, nspace=tag)

        fp_out = INDIR + fp.split('.')[0] + '.csv'

        with open(fp_out, 'w') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(FIELDS)
            for trace in traces:
                csvwriter.writerows(trace)


def get_traces(root: object, nspace: str) -> list:
    event_id = 0.0
    traces = list()
    for trace_id, trace in enumerate(root.findall(nspace+'trace')):
        trace_ = list()
        for attribute in trace.findall(".//"+nspace+"event/*[@key='concept:name']"):
            event_id += 1.0
            trace_.append([trace_id, attribute.attrib['value'], event_id])
        traces.append(trace_)
    return traces


if __name__ == '__main__':
    preprocess_logs("input/trie-stream/M-models/", 'M-model')
    preprocess_logs("input/trie-stream/BPI-logs/", 'BPI-logs')
    if not os.path.isdir(OUTDIR):
        os.mkdir(OUTDIR)

    for fp in os.listdir(INDIR):
        outfp = os.path.join(INDIR, fp)
        print('Processing {}'.format(fp))
        df = pd.read_csv(outfp, sep=',')

        n_cases = df['T:concept:name'].unique().shape[0]
        case_length_stats = df[['T:concept:name', 'id']].groupby(
            'T:concept:name').count().describe()

        # rename
        df.columns = ['caseid', 'activity', 'id']

        # add activity id
        ordered_acts = sorted(list(set(df['activity'])))
        activity_cat_type = CategoricalDtype(
            categories=ordered_acts, ordered=True)
        df['activity_id'] = df.activity.astype(activity_cat_type).cat.codes
        df = df[['id', 'caseid', 'activity', 'activity_id']]

        out_fp = os.path.join(OUTDIR, fp)
        df.to_csv(out_fp, index=None)