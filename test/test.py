from unibox import Dataset, Bbox

if __name__ == '__main__':
    txt_path = r'.asset/bus.txt'
    img_path = r'.asset/bus.jpg'

    data = Dataset().load('yolo', lb_path=txt_path)
    data.img_path = img_path
    
    # Create a new bounding box
    bbox = Bbox([50, 50, 200, 200],"ltrb",True, info={"label": "1"})
    data.append(bbox)
    

    mapping = {'0': 'person', '1': 'bus'}
    mapping1 = {'person': '0', 'bus': '1'}


    xml_path = r'.asset/bus.xml'
    json_path = r'.asset/bus.json'
    txt_path1 = r'.asset/bus1.txt'
    txt_path2 = r'.asset/bus2.txt'

    data.save(xml_path, format='voc', mapping=mapping)
    data.save(json_path, format='labelme', mapping=mapping)
    data.save(txt_path1, format='yolo')
    
    
    data1 = Dataset(img_path).load('labelme', lb_path=json_path)
    data1.save(txt_path2, format='yolo', mapping=mapping1)
    print(data1.dump(format='yolo', mapping=mapping1))