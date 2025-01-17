import unittest
from pathlib import Path
from unibox import Bbox,Dataset


class TestDataset(unittest.TestCase):

    def setUp(self):
        self.dataset = Dataset()

    def test_init(self):
        self.assertIsNone(self.dataset.img_path)
        self.assertEqual(self.dataset.label, [])
        self.assertEqual(self.dataset["label_path"], None)

    def test_append(self):
        bbox = Bbox([0, 0, 10, 10], "ltrb", True)
        self.dataset.append(bbox)
        self.assertEqual(len(self.dataset), 1)
        self.assertEqual(self.dataset.label[0], bbox)

    def test_img_path_property(self):
        path = "/path/to/image.jpg"
        self.dataset.img_path = path
        self.assertEqual(self.dataset.img_path, path)

    def test_img_path_setter_none(self):
        with self.assertRaises(ValueError):
            self.dataset.img_path = None

    def test_label_property(self):
        bbox = Bbox([0, 0, 10, 10], "ltrb", True)
        self.dataset.append(bbox)
        self.assertEqual(len(self.dataset.label), 1)
        self.assertEqual(self.dataset.label[0], bbox)

    def test_remove_label(self):
        bbox = Bbox([0, 0, 10, 10], "ltrb", True)
        self.dataset.append(bbox)
        self.dataset.remove_label(0)
        self.assertEqual(len(self.dataset), 0)

    def test_set_label(self):
        bbox1 = Bbox([0, 0, 10, 10], "ltrb", True)
        bbox2 = Bbox([10, 10, 20, 20], "ltrb", True)
        self.dataset.append(bbox1)
        self.dataset.set_label(0, bbox2)
        self.assertEqual(self.dataset.label[0], bbox2)

    def test_set_label_index_out_of_range(self):
        bbox = Bbox([0, 0, 10, 10], "ltrb", True)
        self.dataset.append(bbox)
        with self.assertRaises(IndexError):
            self.dataset.set_label(1, bbox)

    def test_clear(self):
        bbox = Bbox([0, 0, 10, 10], "ltrb", True)
        self.dataset.append(bbox)
        self.dataset.clear()
        self.assertEqual(len(self.dataset), 0)
        self.assertEqual(self.dataset.img_path, None)
        self.assertEqual(self.dataset["label_path"], None)

    def test_getitem(self):
        self.dataset["label_path"] = "/path/to/label.txt"
        self.assertEqual(self.dataset["label_path"], "/path/to/label.txt")

    def test_setitem(self):
        self.dataset["label_path"] = "/path/to/label.txt"
        self.assertEqual(self.dataset["label_path"], "/path/to/label.txt")

    def test_delitem(self):
        self.dataset["label_path"] = "/path/to/label.txt"
        del self.dataset["label_path"]
        self.assertIsNone(self.dataset["label_path"])

    def test_update(self):
        self.dataset.update(label_path="/path/to/label.txt")
        self.assertEqual(self.dataset["label_path"], "/path/to/label.txt")

    def test_len(self):
        bbox = Bbox([0, 0, 10, 10], "ltrb", True)
        self.dataset.append(bbox)
        self.assertEqual(len(self.dataset), 1)

    def test_load_no_input(self):
        with self.assertRaises(ValueError):
            self.dataset.load("format")

    def test_load_invalid_format(self):
        with self.assertRaises(FileNotFoundError):
            self.dataset.load("invalid_format", lb_path="path/to/label.txt")

    def test_dump_invalid_format(self):
        with self.assertRaises(ImportError):
            self.dataset.dump("invalid_format")

    def test_save(self):
        bbox = Bbox([0, 0, 10, 10], "ltrb", True,[30,30])
        self.dataset.append(bbox)
        path = Path("test_output.txt")
        self.dataset.save(path, "yolo")
        self.assertTrue(path.exists())
        path.unlink()

if __name__ == "__main__":
    unittest.main()
