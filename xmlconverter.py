import os
import xml.etree.ElementTree as ET

def convert_voc_to_yolo(voc_dir, yolo_dir, classes):
    if not os.path.exists(yolo_dir):
        os.makedirs(yolo_dir)
    
    for filename in os.listdir(voc_dir):
        if not filename.endswith('.xml'):
            continue
        
        # Parse the XML file
        tree = ET.parse(os.path.join(voc_dir, filename))
        root = tree.getroot()
        
        image_width = int(root.find('size/width').text)
        image_height = int(root.find('size/height').text)
        
        yolo_annotations = []
        
        for obj in root.findall('object'):
            class_name = obj.find('name').text
            if class_name not in classes:
                continue
            class_id = classes.index(class_name)
            
            bndbox = obj.find('bndbox')
            xmin = int(bndbox.find('xmin').text)
            ymin = int(bndbox.find('ymin').text)
            xmax = int(bndbox.find('xmax').text)
            ymax = int(bndbox.find('ymax').text)
            
            # Convert to YOLO format
            x_center = (xmin + xmax) / 2.0 / image_width
            y_center = (ymin + ymax) / 2.0 / image_height
            width = (xmax - xmin) / image_width
            height = (ymax - ymin) / image_height
            
            yolo_annotations.append(f"{class_id} {x_center} {y_center} {width} {height}")
        
        # Save YOLO annotations to file
        yolo_filename = os.path.join(yolo_dir, filename.replace('.xml', '.txt'))
        with open(yolo_filename, 'w') as yolo_file:
            yolo_file.write('\n'.join(yolo_annotations))

# Define classes
classes = ['barbell']

# Define directories
voc_dir = 'img'
yolo_dir = 'data'

# Convert VOC to YOLO format
convert_voc_to_yolo(voc_dir, yolo_dir, classes)
