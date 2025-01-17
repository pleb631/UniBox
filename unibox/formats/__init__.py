
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

 

    def formats(self):
        for key, frm in self._formats.items():
            if isinstance(frm, str):
                self._formats[key] = load_format_class(frm)
            yield self._formats[key]

    def get_format(self, key):
        if key not in self._formats:
            raise NotImplementedError(f"has no format '{key}' or it is not registered.")
        if isinstance(self._formats[key], str):
            self._formats[key] = load_format_class(self._formats[key])
        return self._formats[key]


registry = Registry()


# import_set
# 在读入lb数据时，应该把box转换为Bbox并存入，其他必要的信息按字典形式存入self._data
# 部分数据集没有图片路径，需要在读入时存入
# 部分数据集的图片是存放在lb中，要进行区分
# 不要在读入lb数据时读入图片
# 在读入lb数据时，应该标注好来源，方便导出时的相关映射

# export_set
# 在需要shape时，应先读dset.img_shape,再读box里的shape，最后读入img
# 在需要其他字段时，应先使用dataset.getitem,最后再读入img，如果没有，就报错
# 在label的分类标签转换时，需要额外定义映射关系