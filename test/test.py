from unibox import Dataset


if __name__=='__main__':
    path = r'test/rgb.txt'
    data = Dataset().load('yolo', lb_path=path)
    data.img_path = r'test/rgb.jpg'
    dst = r'test/test.json'
    with open(dst, 'wb') as file:
        data.dump(file, format='labelme')
