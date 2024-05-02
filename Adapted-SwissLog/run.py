# -*- coding: utf-8 -*-
#!/usr/bin/env python
from concurrent.futures import ThreadPoolExecutor

from layers.file_output_layer import FileOutputLayer
from layers.knowledge_layer import KnowledgeGroupLayer
from layers.mask_layer import MaskLayer
from layers.tokenize_group_layer import TokenizeGroupLayer
from layers.dict_group_layer import DictGroupLayer


import sys
import os
import re
import string
import hashlib
from datetime import datetime
from tqdm import tqdm
import pandas as pd
import argparse

input_dir = '../datasets' # The input directory of log file
output_dir = './LogParserResult/' # The output directory of parsing results


def load_logs(log_file, regex, headers):
    """ Function to transform log file to dataframe
    """
    log_messages = dict()
    linecount = 0
    with open(log_file, 'r',  encoding="utf8", errors='ignore') as fin:
        for line in tqdm(fin.readlines(), desc='load data'):
            try:
                linecount += 1
                match = regex.search(line.strip())
                message = dict()
                for header in headers:
                    message[header] = match.group(header)
                message['LineId'] = linecount
                log_messages[linecount] = message
            except Exception as e:
                pass
    return log_messages

def generate_logformat_regex(logformat):
    """ Function to generate regular expression to split log messages
    """
    headers = []
    splitters = re.split(r'(<[^<>]+>)', logformat)
    regex = ''
    for k in range(len(splitters)):
        if k % 2 == 0:
            splitter = re.sub(' +', '\\\s+', splitters[k])
            regex += splitter
        else:
            header = splitters[k].strip('<').strip('>')
            regex += '(?P<%s>.*?)' % header
            headers.append(header)
    regex = re.compile('^' + regex + '$')
    return headers, regex

benchmark_settings = {

#    'OpenStack': {
#        'log_file': 'OpenStack/openstack_concat.log',
#        'log_format': '<Logrecord> <Date> <Time> <Pid> <Level> <Component> \[<ADDR>\] <Content>',
#        'regex': [r'((\d+\.){3}\d+,?)+', r'/.+?\s', r'\s\d+\s']
#    },

 #   'SSH': {
 #       'log_file': 'SSH/SSH.log',
 #       'log_format': '<Date> <Day> <Time> <Component> sshd\[<Pid>\]: <Content>',
 #       'regex': [r'([\w-]+\.){2,}[\w-]+', r'(\d+\.){3}\d+']
 #   },


 #   'BGL': {
 #       'log_file': 'BGL/BGL.log',
 #       'log_format': '<Label> <Timestamp> <Date> <Node> <Time> <NodeRepeat> <Type> <Component> <Level> <Content>',
 #       'regex': [r'core\.\d+']
 #   },

 #   'Linux': {
 #       'log_file': 'Linux/Linux.log',
 #       'log_format': '<Month> <Date> <Time> <Level> <Component>(\[<PID>\])?: <Content>',
 #       'regex': [r'(\d+\.){3}\d+', r'\d{2}:\d{2}:\d{2}']
 #   },

 #   'Zookeeper': {
 #       'log_file': 'Zookeeper/Zookeeper.log',
 #       'log_format': '<Date> <Time> - <Level>  \[<Node>:<Component>@<Id>\] - <Content>',
 #       'regex': [r'(/|)(\d+\.){3}\d+(:\d+)?']
 #   },

   # 'Hadoop': {
   #     'log_file': 'Hadoop/Hadoop_concat.log',
   #     'log_format': '<Date> <Time> <Level> \[<Process>\] <Component>: <Content>',
   #     'regex': [r'(\d+\.){3}\d+']
   # },

   # 'Android': {
   #     'log_file': 'Android_v1/Android.log',
   #     'log_format': '<Date> <Time>  <Pid>  <Tid> <Level> <Component>: <Content>',
   #     'regex': [r'(/[\w-]+)+', r'([\w-]+\.){2,}[\w-]+', r'\b(\-?\+?\d+)\b|\b0[Xx][a-fA-F\d]+\b|\b[a-fA-F\d]{4,}\b']
   # },

   # 'HealthApp': {
   #     'log_file': 'HealthApp/HealthApp.log',
   #     'log_format': '<Time>\|<Component>\|<Pid>\|<Content>',
   #     'regex': []
   # },

  #  'Apache': {
  #      'log_file': 'Apache/Apache.log',
  #      'log_format': '\[<Time>\] \[<Level>\] <Content>',
  #      'regex': [r'(\d+\.){3}\d+']
  #  },

 #   'HPC': {
 #       'log_file': 'HPC/HPC.log',
 #       'log_format': '<LogId> <Node> <Component> <State> <Time> <Flag> <Content>',
 #       'regex': [r'=\d+']
 #   },


#    'Proxifier': {
#        'log_file': 'Proxifier/Proxifier.log',
#        'log_format': '\[<Time>\] <Program> - <Content>',
#        'regex': [r'<\d+\s?sec', r'([\w-]+\.)+[\w-]+(:\d+)?', r'\d{2}:\d{2}(:\d{2})*', r'[KGTM]B']
#    },

# End batch 1

   # 'HDFS': {
   #      'log_file': 'HDFS_v1/HDFS.log',
   #      'log_format': '<Date> <Time> <Pid> <Level> <Component>: <Content>',
   #      'regex': [r'blk_-?\d+', r'(\d+\.){3}\d+(:\d+)?']
   #      },
   #
   #   'Mac': {
   #       'log_file': 'Mac/Mac_clean.log',
   #       'log_format': '<Month>  <Date> <Time> <User> <Component>\[<PID>\]( \(<Address>\))?: <Content>',
   #       'regex': [r'([\w-]+\.){2,}[\w-]+']
   #   },

# End batch 2

    # 'Spark': {
    #     'log_file': 'Spark/Spark_concat.log',
    #     'log_format': '<Date> <Time> <Level> <Component>: <Content>',
    #     'regex': [r'(\d+\.){3}\d+', r'\b[KGTM]?B\b', r'([\w-]+\.){2,}[\w-]+']
    # },

    'Thunderbird': {
        'log_file': 'Thunderbird/Thunderbird.log',
        'log_format': '<Label> <Timestamp> <Date> <User> <Month> <Day> <Time> <Location> <Component>(\[<PID>\])?: <Content>',
        'regex': [r'(\d+\.){3}\d+']
        },
    #
    # 'Windows': {
    #     'log_file': 'Windows/Windows.log',
    #     'log_format': '<Date> <Time>, <Level>                  <Component>    <Content>',
    #     'regex': [r'0x.*?\s']
    #     },
    #

# End batch 3

}

def parse_logs(dataset, setting):
    print('\n=== Evaluation on %s ===' % dataset)
    indir = os.path.join(input_dir, os.path.dirname(setting['log_file']))
    indir = os.path.join(input_dir, os.path.dirname(setting['log_file']))
    outdir = os.path.join(output_dir, os.path.dirname(setting['log_file']))
    log_file = os.path.basename(setting['log_file'])

    filepath = os.path.join(indir, log_file)
    print('Parsing file: ' + filepath)
    starttime = datetime.now()
    headers, regex = generate_logformat_regex(setting['log_format'])
    log_messages = load_logs(filepath, regex, headers)
    # preprocess layer
    log_messages = KnowledgeGroupLayer(log_messages).run()
    # tokenize layer
    log_messages = TokenizeGroupLayer(log_messages, rex=setting['regex']).run()
    # dictionarize layer and cluster by wordset
    dict_group_result = DictGroupLayer(log_messages, corpus).run()
    # apply LCS and prefix tree
    results, templates = MaskLayer(dict_group_result).run()
    output_file = os.path.join(outdir, log_file)
    # output parsing results
    FileOutputLayer(log_messages, output_file, templates, ['LineId'] + headers).run()
    print('Parsing done. [Time taken: {!s}]'.format(datetime.now() - starttime))

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--dictionary', default='../EngCorpus.pkl', type=str)
    args = parser.parse_args()
    corpus = args.dictionary

    #pool = ThreadPoolExecutor(max_workers=11)

    for dataset, setting in benchmark_settings.items():
        parse_logs(dataset, setting)
        #pool.submit(parse_logs, dataset, setting)


        # F1_measure, accuracy = evaluator.evaluate(
        #                     groundtruth=os.path.join(indir, log_file + '_structured.csv'),
        #                     parsedresult=os.path.join(outdir, log_file + '_structured.csv')
        #                     )
        # benchmark_result.append([dataset, F1_measure, accuracy])

    # print('\n=== Overall evaluation results ===')
    # avg_accr = 0
    # for i in range(len(benchmark_result)):
    #     avg_accr += benchmark_result[i][2]
    # avg_accr /= len(benchmark_result)
    # pd_result = pd.DataFrame(benchmark_result, columns={'dataset', 'F1_measure', 'Accuracy'})
    # print(pd_result)
    # print('avarage accuracy is {}'.format(avg_accr))
    # pd_result.to_csv('benchmark_result.csv', index=False)

