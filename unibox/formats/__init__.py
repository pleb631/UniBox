
from importlib import import_module


def load_format_class(dotted_path):
    try:
        module_path, class_name = dotted_path.rsplit('.', 1)
        return getattr(import_module(module_path), class_name)
    except (ValueError, AttributeError) as err:
        raise ImportError(
            f"Unable to load format class '{dotted_path}' ({err})"
        ) from err

class Registry:
    _formats = {}

    def register(self, key, format_or_path):

        self._formats[key] = format_or_path

    def register_builtins(self):
        # Registration ordering matters for autodetection.
        self.register('labelme', "unibox.formats.labelme.Labelme")
        self.register('yolo', "unibox.formats.yolo.Yolo")
        self.register('voc', "unibox.formats.voc.VOC")

 

    def formats(self):
        for key, frm in self._formats.items():
            if isinstance(frm, str):
                self._formats[key] = load_format_class(frm)
            yield self._formats[key]

    def get_format(self, key):
        if key not in self._formats:
            raise ImportError(f"has no format '{key}' or it is not registered.")
        if isinstance(self._formats[key], str):
            self._formats[key] = load_format_class(self._formats[key])
        return self._formats[key]


registry = Registry()


