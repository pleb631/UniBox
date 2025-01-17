import numpy as np
from unibox import Dataset,Bbox
from typing import Dict
import xml.etree.ElementTree as ET
import cv2
import os



class VOC:
    @staticmethod
    def import_set(dset: Dataset, in_stream,**kwargs):
        dset.clear()
        tree = ET.parse(in_stream)
        root = tree.getroot()
        size = root.find('size')
        w = int(size.find('width').text)
        h = int(size.find('height').text)
        key = ["difficult","pose","truncated"]
        for obj in root.iter('object'):
            label = obj.find('name').text
            info = {k:obj.find(k).text for k in key}
            info["label"] = label
            
            
            xmlbox = obj.find('bndbox')
            bb = [float(xmlbox.find(x).text) for x in ('xmin','ymin','xmax','ymax')]
            box = Bbox(bb, "ltrb", True, [w,h],info)
            dset.append(box)

    @staticmethod
    def export_set(dset:Dataset,mapping:Dict=None,**kwargs):


        if dset["img_shape"] is None:
            img_wh = dset.label[0].img_wh()
            if img_wh is None:
                if dset.img_path is None:
                    raise ValueError("Image shape is not defined.")
                img = cv2.imdecode(np.fromfile(dset.img_path, np.uint8), 1)
                dset["img_shape"] = [img.shape[1], img.shape[0]]
                img_wh = dset["img_shape"]
            else:
                dset["img_shape"] = img_wh

        img_wh = dset["img_shape"]

        xml_str = '<annotation>\n' + '<folder>VOC2007</folder>\n'
        xml_str+=f'<filename>{os.path.basename(dset.img_path)}</filename>\n'
        xml_str+='<size>\n'
        xml_str+=f'<width>{img_wh[0]}</width>\n'
        xml_str+=f'<height>{img_wh[1]}</height>\n'
        xml_str+='<depth>3</depth>\n'
        xml_str+='</size>\n'



        for bbox in dset.label:

            x1, y1, x2, y2 = bbox.ltrb(is_pixel_distance=True,img_shape = img_wh).tolist()
            xml_str+='<object>\n'

            label = bbox.info.get("label",None)
            if label is None:
                label = '0'
            if mapping is not None:
                label = mapping[label]

            
            truncated = bbox.info.get("truncated",0)
            difficult = bbox.info.get("difficult",0)
            pose = bbox.info.get("pose","Unspecified")
            xml_str+=f'<name>{label}</name>\n'
            xml_str+=f'<pose>{pose}</pose>\n'
            xml_str+=f'<truncated>{truncated}</truncated>\n'
            xml_str+=f'<difficult>{difficult}</difficult>\n'
            xml_str+='<bndbox>\n'
            xml_str+=f'<xmin>{int(round(x1))}</xmin>\n'
            xml_str+=f'<ymin>{int(round(y1))}</ymin>\n'
            xml_str+=f'<xmax>{int(round(x2))}</xmax>\n'
            xml_str+=f'<ymax>{int(round(y2))}</ymax>\n'
            xml_str+='</bndbox>\n'
            xml_str+='</object>\n'

        xml_str+='</annotation>'
        
        
        return xml_str