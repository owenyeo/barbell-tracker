import os
import pandas as pd
import xml.etree.ElementTree as ET

def xml_to_csv(path):
    xml_list = []
    for xml_file in os.listdir(path):
        if xml_file.endswith('.xml'):
            tree = ET.parse(os.path.join(path, xml_file))
            root = tree.getroot()
            for member in root.findall('object'):
                value = (root.find('filename').text,
                         int(root.find('size/width').text),
                         int(root.find('size/height').text),
                         member.find('name').text,
                         int(member.find('bndbox/xmin').text),
                         int(member.find('bndbox/ymin').text),
                         int(member.find('bndbox/xmax').text),
                         int(member.find('bndbox/ymax').text))
                xml_list.append(value)
    column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    return xml_df

def main():
    xml_path = 'img'
    xml_df = xml_to_csv(xml_path)
    xml_df.to_csv('annotations.csv', index=False)

if __name__ == '__main__':
    main()