
from unibox.bbox import Bbox

import unittest
import numpy as np

class TestBbox(unittest.TestCase):

    def test_init(self):
        # Test initialization with different formats and is_pixel_distance
        box = [10, 20, 30, 40]
        bbox = Bbox(box, "ltrb", True)
        self.assertEqual(bbox.ltrb().tolist(), box)

        box = [10, 20, 20, 20]
        bbox = Bbox(box, "xywh", True)
        self.assertEqual(bbox.ltrb().tolist(), [0, 10, 20, 30])

        box = [10, 20, 30, 40]
        bbox = Bbox(box, "ltwh", True)
        self.assertEqual(bbox.ltrb().tolist(), [10, 20, 40, 60])

        # box = [0.1, 0.2, 0.3, 0.4]
        # bbox = Bbox(box, "ltrb", False)
        # self.assertEqual(bbox.ltrb().tolist(), box)

    def test_norm2pixel(self):
        # Test conversion from normalized to pixel coordinates
        box = [0.1, 0.2, 0.3, 0.4]
        img_shape = [100, 200]
        expected = [10, 40, 30, 80]
        self.assertEqual(Bbox.norm2pixel(box, img_shape).tolist(), expected)

    def test_pixel2norm(self):
        # Test conversion from pixel to normalized coordinates
        box = [10, 40, 30, 80]
        img_shape = [100, 200]
        expected = [0.1, 0.2, 0.3, 0.4]
        self.assertEqual(Bbox.pixel2norm(box, img_shape).tolist(), expected)

    def test_convert(self):
        # Test conversion between different formats
        box = [10, 20, 30, 40]
        expected = [20, 30, 20, 20]
        self.assertEqual(Bbox.convert(np.array(box), "ltrb", "xywh").tolist(), expected)

        box = [20, 30, 20, 20]
        expected = [10, 20, 30, 40]
        self.assertEqual(Bbox.convert(np.array(box), "xywh", "ltrb").tolist(), expected)

        box = [10, 20, 30, 40]
        expected = [10, 20, 20, 20]
        self.assertEqual(Bbox.convert(np.array(box), "ltrb", "ltwh").tolist(), expected)

        box = [10, 20, 20, 20]
        expected = [10, 20, 30, 40]
        self.assertEqual(Bbox.convert(np.array(box), "ltwh", "ltrb").tolist(), expected)

    def test_get_safe_box(self):
        # Test getting safe box with different formats and is_pixel_distance
        box = [0.1, 0.2, 0.3, 0.4]
        img_shape = [100, 200]
        expected = [10, 40, 30, 80]
        self.assertEqual(Bbox.get_safe_box(box, "ltrb", "ltrb", img_shape, False, True).tolist(), expected)

        box = [10, 20, 30, 40]
        expected = [0.1, 0.1, 0.3, 0.2]
        self.assertEqual(Bbox.get_safe_box(box, "ltrb", "ltrb", img_shape, True, False).tolist(), expected)
        
        
        box = [90, 180, 30, 40]
        expected = [75, 160, 100, 200]
        self.assertEqual(Bbox.get_safe_box(box, "xywh", "ltrb", img_shape, True, True).tolist(), expected)

    def test_input_validation(self):
        # Test input validation
        with self.assertRaises(ValueError):
            Bbox([-10, 20, 30, 40], "ltrb", True)

        with self.assertRaises(ValueError):
            Bbox([10, 20, 30], "ltrb", True)

        with self.assertRaises(ValueError):
            Bbox([10, 20, 30, 40], "invalid", True)



if __name__ == '__main__':
    
    unittest.main()
