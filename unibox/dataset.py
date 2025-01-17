from typing import List, Any, Dict
from pathlib import Path

from unibox import Bbox
from unibox.formats import registry
from unibox.utils import normalize_input



class Dataset:
    def __init__(self, img_path: str | Path = None, flag: str = None) -> None:
        self._img_path: str = str(img_path) if img_path is not None else None
        self._data: dict = {
            "data": List[Bbox],
            "info": {
                "label_path": str | Path | None,
            },
        }

        self.flag = flag
        # optional: img_shape = [w,h]

    def append(self, label: Bbox):
        self._data["data"].append(label)

    @property
    def img_path(self) -> str:
        return self._img_path

    @img_path.setter
    def img_path(self, img_path: str | Path):
        if img_path is None:
            raise ValueError("img_path cannot be None.")
        self._img_path = str(img_path)

    @property
    def label(self) -> List[Bbox]:
        return self._data["data"].copy()

    def remove_label(self, index: int):
        self._data["data"].pop(index)

    def set_label(self, index: int, label: Bbox):
        if index >= len(self):
            raise IndexError("index out of range")
        self._data["data"][index] = label

    def clear(
        self,
    ):
        self._data = dict(data=[], info={"img_path": None, "label_path": None})

    def __getitem__(self, key: str):
        return self._data["info"].get(key, None)

    def __setitem__(self, key: str, value: Any):
        self._data["info"][key] = value

    def __delitem__(self, key: str):
        del self._data["info"][key]

    def update(self, **kwargs):
        self._data["info"].update(kwargs)

    def __len__(self):
        return len(self._data["data"])

    def load(self, format: str, in_stream=None, lb_path=None, **kwargs):
        """
        Load the dataset from a file or input stream.

        Args:
            format (str): The format of the dataset.
            in_stream (file-like object, optional): The input stream containing the dataset.
            lb_path (str, optional): The path to the file containing the dataset.
            **kwargs: Additional keyword arguments to be passed to the format-specific import function..

        Raises:
            ValueError: If neither lb_path nor in_stream is provided.
            ImportError: If the specified format cannot be imported.
        """
        if lb_path is None and in_stream is None:
            raise ValueError("Either lb_path or in_stream must be provided.")
        if in_stream is None:
            with open(lb_path, "rb") as file:
                in_stream = file.read()

        if lb_path is not None:
            self["label_path"] = lb_path

        stream = normalize_input(in_stream)

        fmt = registry.get_format(format)
        if not hasattr(fmt, "import_set"):
            raise ImportError(f"Format {format} cannot be imported.")

        fmt.import_set(self, stream, **kwargs)
        return self

    def dump(self, format: str, **kwargs):
        """
            Export the dataset in the specified format.

            Args:
                format (str): The format to export the dataset in.
                **kwargs: Additional keyword arguments to pass to the export function.

            Returns:
                bytes: The exported dataset as bytes.
        """
        fmt = registry.get_format(format)
        if not hasattr(fmt, "export_set"):
            raise ImportError(f"Format {format} cannot be exported.")

        result = fmt.export_set(self, **kwargs)
        if isinstance(result, str):
            result = result.encode("utf-8")

        return result

    def save(self, outfile: str | Path, format: str, **kwargs):
        """
        Save the dataset to a file.

        Args:
            outfile (str | Path): The path to the output file.
            format (str): The format in which to save the dataset.
            **kwargs: Additional keyword arguments to be passed to the dump method.
        """       
        with open(outfile, "wb") as out_stream:
            stream = normalize_input(out_stream)
            result = self.dump(out_stream, format, **kwargs)
            stream.write(result)

    def __repr__(self) -> str:
        return f"{self._data}"


registry.register_builtins()

if __name__ == "__main__":
    da = Dataset()
