# encoding: utf-8

"""
@author: huayuxiu

Transform xml files to csv files.
The original traffic data are from GEANT, https://totem.info.ucl.ac.be/dataset.html
Please refer to data.pdf for the description of the traffic data
"""

import xml.etree.cElementTree as et
import numpy as np
import os

# path of xml files
xml_path = './data/traffic-matrices/'

# path of csv files
csv_path = './data/csv/'

xml_count = 0
csv_count = 0

for filename in os.listdir(xml_path):
    xml_count += 1
    tree= et.parse(os.path.join(xml_path, filename))
    root = tree.getroot()
    IntraTM = root.getchildren()[1]
    srcs = IntraTM.getchildren()

    # get traffic matrix from xml file 
    traffic_matrix = np.zeros([23, 23])
    for src in srcs:
        src_id = int(src.attrib['id'])
        for dst in src:
            if 'id' in dst.attrib:
                dst_id = int(dst.attrib['id'])
            bandwidth = float(dst.text)
            traffic_matrix[src_id-1, dst_id-1] = bandwidth

    # store traffic matrix to csv file
    portion = os.path.splitext(filename)
    if portion[1] == '.xml':
        newname = portion[0]+'.csv'
        np.savetxt(os.path.join(csv_path, newname), traffic_matrix, delimiter=',')
        csv_count += 1
        print(newname, 'has been saved')
    del(traffic_matrix)

print('the number of xml files', xml_count)
print('the number of csv files', csv_count)

assert xml_count == csv_count
