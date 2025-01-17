import json
import numpy as np
import cv2
from typing import Dict
import os

from unibox import Dataset,Bbox

class Labelme:
    SHAPE_TEMPLATE = """\
    {{ \
      "label": "{label}", \
      "points": {point}, \
      "group_id": null, \
      "description": "", \
      "shape_type": "rectangle", \
      "flags": {{}}, \
      "mask": null \
    }}"""
    
    TABLE_TEMPLATE = """\
    {{
    "version": "5.6.0",
    "flags": {{}}, \
    "shapes": [{shape}], \
    "imagePath": "{imagePath}", \
    "imageData": null, \
    "imageHeight": {imageHeight}, \
    "imageWidth": {imageWidth} \
    }}"""

    @staticmethod
    def import_set(dset:Dataset, in_stream,base64=False,**kwargs):
        
        """Returns dataset from JSON stream."""
        dset.clear()


        json_data = json.loads(in_stream.read().decode('utf-8'))
        
        shapes = json_data["shapes"]
        imageHeight = json_data["imageHeight"]
        imageWidth = json_data["imageWidth"]
        if dset.img_path is None:
            dset.img_path = json_data["imagePath"]

        dset["img_shape"] = [imageWidth, imageHeight]

        for shape in shapes:
            if shape["shape_type"] == "rectangle":
                label = shape["label"]
                points = np.array(shape["points"]).reshape(-1, 2)
                
 
                x1, y1 = points.min(axis=0)
                x2, y2 = points.max(axis=0)
                
                bbox = Bbox(
                    [x1, y1, x2, y2],
                    "ltrb",
                    is_pixel_distance=True,
                    img_shape=[imageWidth, imageHeight],
                    info={"label": label}
                )
                dset.append(bbox)

    @staticmethod
    def export_set(dset:Dataset,mapping:Dict=None,**kwargs):
        """Writes dataset to JSON stream."""
        shape = []
        
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
      
        

        for bbox in dset.label:
            x1, y1, x2, y2 = bbox.ltrb(is_pixel_distance=True,img_shape = img_wh).tolist()
            l = bbox.info.get('label', None)
            if l is None:
                l = "0"
            elif mapping is not None:
                l = mapping[l]
            point = [[x1, y1], [x2, y2]]
            
            shape.append(Labelme.SHAPE_TEMPLATE.format(point=point, label=l))
        
        img_path = os.path.basename(dset.img_path)
        if img_path is None:
            raise ValueError("Image path is not defined.")
        
        
        result = Labelme.TABLE_TEMPLATE.format(
            shape=",".join(shape), 
            imageHeight=img_wh[1], 
            imageWidth=img_wh[0],
            imagePath=img_path
        )
        return json.dumps(json.loads(result), ensure_ascii=False, indent=4)
