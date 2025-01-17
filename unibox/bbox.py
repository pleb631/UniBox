import numpy as np


class Bbox:
    """
    Bbox class represents a bounding box with various formats and conversions.

    Args:
        box (list | np.ndarray): The bounding box coordinates.
        format (str): The format of the bounding box coordinates. Must be one of ["ltrb", "xywh", "ltwh"].
        is_pixel_distance (bool, optional): Whether the bounding box coordinates are in pixel distance. Defaults to True.
        img_shape (list | np.ndarray | None, optional): The shape of the image. Defaults to None.
        info (dict | None, optional): Additional information about the bounding box. Defaults to None.

    Attributes:
        _formats (list): The list of supported bounding box formats.

    Methods:
        ltrb: Converts the bounding box coordinates to the "ltrb" format.
        norm2pixel: Converts the bounding box coordinates from normalized format to pixel format.
        pixel2norm: Converts the bounding box coordinates from pixel format to normalized format.
        info: Returns the additional information about the bounding box.

    Static Methods:
        convert: Converts the bounding box coordinates from one format to another.
        ltrb2xywh: Converts the bounding box coordinates from "ltrb" format to "xywh" format.
        xywh2ltrb: Converts the bounding box coordinates from "xywh" format to "ltrb" format.
        ltrb2ltwh: Converts the bounding box coordinates from "ltrb" format to "ltwh" format.
        ltwh2ltrb: Converts the bounding box coordinates from "ltwh" format to "ltrb" format.
        get_safe_box: Converts the bounding box coordinates from one format to another with clipping.


    Usage:
        box = Bbox([0.1, 0.2, 0.3, 0.4], "ltrb", is_pixel_distance=True, img_shape=[100, 100], info={"label": "car"})
        box.ltrb(is_pixel_distance=True)

    """

    _formats: list = ["ltrb", "xywh", "ltwh"]

    def __init__(
        self,
        box: list | np.ndarray,
        format: str,
        is_pixel_distance: bool = True,
        label: str|int = None,
        img_shape: list | np.ndarray | None = None,  # [w,h]
        info: dict | None = None,
    ) -> None:
        """
        Initialize a Bbox object.

        Args:
            box (list | np.ndarray): The bounding box coordinates.
            format (str): The format of the bounding box coordinates.
            is_pixel_distance (bool, optional): Whether the box coordinates are in pixel distance. Defaults to True.
            img_shape (list | np.ndarray | None, optional): The shape of the image. Defaults to None.
            info (dict | None, optional): Additional information about the bounding box. Defaults to None.
        """
        box, img_shape = self._check_input_corrcetness(
            box, is_pixel_distance, img_shape, format
        )

        bbox = Bbox.convert(box, format, "ltrb")
        x1, y1, x2, y2 = bbox
        if x1 > x2 or y1 > y2:
            raise ValueError(f"Invalid bounding box format,with {bbox}")
        self._bbox = bbox

        self._img_shape = img_shape
        self._is_pixel_distance = is_pixel_distance

        if label is None:
            self._label = "0"
        elif isinstance(label, str):
            self._label = label
        elif isinstance(label, int):
                self._label = str(label)
        else:
            raise ValueError("Label must be a string")
            
        self._info = info

    def _check_input_corrcetness(
        self,
        bbox: list | np.ndarray,
        is_pixel_distance: bool,
        img_shape: list | np.ndarray | None,
        format: str = "ltrb",
    ):

        if isinstance(bbox, list):
            bbox = np.array(bbox)
        bbox = bbox.flatten()

        if np.any(bbox < 0):
            raise ValueError("Bounding box must have non-negative values")

        if bbox.size != 4:
            raise ValueError("Bounding box must have 4 elements")

        if img_shape is not None:
            if isinstance(img_shape, list):
                img_shape = np.array(img_shape, dtype=np.int32)
            img_shape = img_shape.flatten()

            if len(img_shape) != 2:
                raise ValueError("img_shape must have 2 elements")
            if not isinstance(img_shape[0], np.int32) or not isinstance(
                img_shape[1], np.int32
            ):
                raise ValueError("img_shape elements must be integers")

        if format not in Bbox._formats:
            raise ValueError(
                f"Invalid bounding box format: {format}, format must be one of {Bbox._formats}"
            )

        if not is_pixel_distance:
            if np.any(bbox > 1.0):
                raise ValueError(
                    f"is_pixel_distance is {is_pixel_distance}, Bounding box must have values between 0 and 1,but {bbox}"
                )
        elif np.all(bbox < 1.0):
            raise ValueError(
                f"is_pixel_distance is {is_pixel_distance}, Bounding box is not corrent pixel format:{bbox}"
            )

        return bbox, img_shape

    def ltrb(self, is_pixel_distance: bool = True, img_shape=None) -> np.ndarray:
        if is_pixel_distance == self._is_pixel_distance:
            return self._bbox
        shape = img_shape if self._img_shape is None else self._img_shape
        if shape is not None:

            return (
                self.norm2pixel(self._bbox, shape)
                if is_pixel_distance
                else self.pixel2norm(self._bbox, shape)
            )
        else:
            raise ValueError(
                "img_shape is not provided, cannot convert box between normalized and pixel"
            )

    def xywh(self, is_pixel_distance: bool = True, img_shape=None) -> np.ndarray:
        box = self.ltrb(is_pixel_distance, img_shape)
        box = Bbox.convert(box, "ltrb", "xywh")
        return box

    # @property
    def img_wh(self) -> list | np.ndarray:
        return self._img_shape

    @property
    def label(self):
        return self._label

    @label.setter
    def setlabel(self, x: str) -> str:
        self._label = x

    @staticmethod
    def norm2pixel(bbox: list | np.ndarray, img_shape: list | np.ndarray) -> np.ndarray:
        if isinstance(bbox, list):
            bbox = np.array(bbox)
        bbox = bbox.flatten()
        if np.any(bbox > 1.0) and np.any(bbox < 0.0):
            raise ValueError(
                f"Bounding box must have values between 0 and 1,but {bbox}"
            )
        if img_shape is None:
            raise ValueError(
                "img_shape is not provided, cannot convert normalized to pixel"
            )

        if isinstance(bbox, list):
            bbox = np.array(bbox)
        bbox = bbox.flatten()

        return bbox * [img_shape[0], img_shape[1], img_shape[0], img_shape[1]]

    @staticmethod
    def pixel2norm(bbox: list | np.ndarray, img_shape: list | np.ndarray) -> np.ndarray:
        if isinstance(bbox, list):
            bbox = np.array(bbox)
        bbox = bbox.flatten()
        if np.all(bbox < 1.0) or np.any(bbox < 0.0):
            raise ValueError(f"Bounding box is not corrent pixel format: {bbox}")
        if img_shape is None:
            raise ValueError(
                "img_shape is not provided, cannot convert normalized to pixel"
            )

        if isinstance(bbox, list):
            bbox = np.array(bbox)
        bbox = bbox.flatten()

        return bbox / [img_shape[0], img_shape[1], img_shape[0], img_shape[1]]

    @property
    def info(self) -> dict:
        return self._info

    @staticmethod
    def convert(box: np.ndarray, srcf: str, dstf: str):
        if srcf not in Bbox._formats:
            raise ValueError(
                f"Invalid bounding box format: {srcf}, format must be one of {Bbox._formats}"
            )
        if dstf not in Bbox._formats:
            raise ValueError(
                f"Invalid bounding box format: {dstf}, format must be one of {Bbox._formats}"
            )

        if srcf != dstf:
            fun_str = f"Bbox.{srcf}2{dstf}"
            try:
                func: np.ndarray = eval(fun_str)
            except:
                raise NotImplementedError(
                    f"Conversion from {srcf} to {dstf} is not implemented"
                )

            box = func(box)

        return box

    @staticmethod
    def ltrb2xywh(x: np.ndarray) -> np.ndarray:
        y = np.copy(x)
        y[..., 0] = (x[..., 0] + x[..., 2]) / 2  # center x
        y[..., 1] = (x[..., 1] + x[..., 3]) / 2  # center y
        y[..., 2] = x[..., 2] - x[..., 0]
        y[..., 3] = x[..., 3] - x[..., 1]
        return y

    @staticmethod
    def xywh2ltrb(x: np.ndarray):
        y = np.empty_like(x)  # faster than clone/copy
        dw = x[..., 2] / 2  # half-width
        dh = x[..., 3] / 2  # half-height
        y[..., 0] = x[..., 0] - dw  # top left x
        y[..., 1] = x[..., 1] - dh  # top left y
        y[..., 2] = y[..., 0] + x[..., 2]  # bottom right x
        y[..., 3] = y[..., 1] + x[..., 3]  # bottom right y
        return y

    @staticmethod
    def ltrb2ltwh(x: np.ndarray):
        y = np.copy(x)
        y[..., 2] = x[..., 2] - x[..., 0]  # width
        y[..., 3] = x[..., 3] - x[..., 1]  # height
        return y

    @staticmethod
    def ltwh2ltrb(x: np.ndarray):
        y = np.copy(x)
        y[..., 2] = x[..., 2] + x[..., 0]  # width
        y[..., 3] = x[..., 3] + x[..., 1]  # height
        return y

    @staticmethod
    def get_safe_box(
        box: list | np.ndarray,
        src_format: str,
        dst_format: str,
        img_shape: list | np.ndarray,
        is_pixel_distance: bool,
        is_dst_pixel_distance: bool,
    ):
        box:Bbox = Bbox(box, src_format, is_pixel_distance, img_shape=img_shape)
        box = box.ltrb(is_dst_pixel_distance)
        box = Bbox.convert(box, "ltrb", dst_format)

        if not is_dst_pixel_distance:
            box = np.clip(box, 0, 1)
        else:
            if img_shape is not None:
                box[0] = np.clip(box[0], 0, img_shape[0])
                box[2] = np.clip(box[2], 0, img_shape[0])
                box[1] = np.clip(box[1], 0, img_shape[1])
                box[3] = np.clip(box[3], 0, img_shape[1])
            else:
                raise ValueError(
                    "img_shape is not provided, cannot clip box to image boundaries"
                )
        return box

    def __repr__(self) -> str:
        return f"xywh=[{self._bbox[0]:.2f},{self._bbox[1]:.2f},{self._bbox[2]:.2f},{self._bbox[3]:.2f}], [w,h]={self._img_shape}, info={self._info}\n"
