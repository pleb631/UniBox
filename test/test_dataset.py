import unittest
from io import BytesIO
from unibox.formats import registry
from unibox.utils import normalize_input
from unibox.dataset import Dataset


class TestDataset(unittest.TestCase):

    def setUp(self):
        self.dataset = Dataset()

    def test_append_and_label(self):
        self.dataset.append("label1")
        self.dataset.append("label2")
        self.assertEqual(self.dataset.label, ["label1", "label2"])

    def test_remove_label(self):
        self.dataset.append("label1")
        self.dataset.append("label2")
        self.dataset.remove_label(0)
        self.assertEqual(self.dataset.label, ["label2"])

    def test_set_label(self):
        self.dataset.append("label1")
        self.dataset.append("label2")
        self.dataset.set_label(1, "new_label")
        self.assertEqual(self.dataset.label, ["label1", "new_label"])

    def test_set_label_out_of_range(self):
        self.dataset.append("label1")
        self.dataset.append("label2")
        with self.assertRaises(IndexError):
            self.dataset.set_label(5, "new_label")

    def test_clear(self):
        self.dataset.append("label1")
        self.dataset.append("label2")
        self.dataset.clear()
        self.assertEqual(self.dataset.label, [])
        self.assertEqual(self.dataset._data["info"], {})

    def test_get_set_item(self):
        self.dataset["key"] = "value"
        self.assertEqual(self.dataset["key"], "value")

    def test_del_item(self):
        self.dataset["key"] = "value"
        del self.dataset["key"]
        with self.assertRaises(KeyError):
            _ = self.dataset["key"]

    def test_update(self):
        self.dataset.update(key1="value1", key2="value2")
        self.assertEqual(self.dataset._data["info"], {"key1": "value1", "key2": "value2"})

    def test_len(self):
        self.dataset.append("label1")
        self.dataset.append("label2")
        self.assertEqual(len(self.dataset), 2)

    # def test_load(self):
    #     # Mocking the import_set function
    #     class MockFormat:
    #         @staticmethod
    #         def import_set(dataset, stream, **kwargs):
    #             dataset.append("mock_label")

    #     registry.register_format("mock", MockFormat)
    #     stream = BytesIO(b"mock_data")
    #     self.dataset.load(stream, format="mock")
    #     self.assertEqual(self.dataset.label, ["mock_label"])

    # def test_dump(self):
    #     # Mocking the export_set function
    #     class MockFormat:
    #         @staticmethod
    #         def export_set(dataset, **kwargs):
    #             return "mock_data"

    #     registry.register_format("mock", MockFormat)
    #     stream = BytesIO()
    #     self.dataset.append("label1")
    #     self.dataset.dump(stream, format="mock")
    #     self.assertEqual(stream.getvalue(), b"mock_data")

    def test_repr(self):
        self.dataset.append("label1")
        self.dataset["key"] = "value"
        self.assertEqual(repr(self.dataset), "{'data': ['label1'], 'info': {'key': 'value'}}")

if __name__ == "__main__":
    unittest.main()
