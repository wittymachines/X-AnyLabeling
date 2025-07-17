import logging
import os
from pathlib import Path

import cv2
import numpy as np
from mmdeploy.apis.utils import build_task_processor
from mmdeploy.utils import get_input_shape
from mmengine import Config
from PyQt5 import QtCore
from PyQt5.QtCore import QCoreApplication

from anylabeling.app_info import __preferred_device__
from anylabeling.views.labeling.shape import Shape
from anylabeling.views.labeling.utils.opencv import qt_img_to_rgb_cv_img

from .model import Model
from .types import AutoLabelingResult
from PyQt5.QtGui import QImage

class WittyOpenEggCarton(Model):
    """Model for Witty Open Egg Carton using RTMDet."""
    class Meta:
        required_config_names = [
            "type",
            "name",
            "display_name",
            "model_path",
            "classes",
        ]
        widgets = ["button_run", "output_select_combobox", "edit_conf"]
        output_modes = {
            "rectangle": QCoreApplication.translate("Model", "Rectangle"),
        }
        default_output_mode = "rectangle"

    def __init__(self, model_config, on_message) -> None:
        # Run the parent class's init method
        super().__init__(model_config, on_message)
        model_name = self.config["type"]
        model_abs_path = Path(self.get_model_abs_path(self.config, "model_path"))
        if not model_abs_path or not os.path.isdir(model_abs_path):
            raise FileNotFoundError(
                QCoreApplication.translate(
                    "Model",
                    f"Could not download or initialize {model_name} model.",
                )
            )
        deploy_cfg = Config.fromfile(str(model_abs_path / "deploy_cfg.py"))
        model_cfg = Config.fromfile(str(model_abs_path / "model_cfg.py"))
        onnx_path = str(model_abs_path / deploy_cfg.onnx_config.save_file)
        backend_files = [
            onnx_path,
        ]
        self.device = __preferred_device__.lower()
        self.task_processor = build_task_processor(model_cfg, deploy_cfg, self.device)
        self.input_shape = get_input_shape(deploy_cfg)
        self.model = self.task_processor.build_backend_model(
            backend_files,
            data_preprocessor_updater=self.task_processor.update_data_preprocessor,
        )
        self.classes = self.config["classes"]
        self.confidence_threshold = self.config.get("conf_threshold", 0.5)

    def predict_shapes(self, image: QImage, image_path: str=None) -> AutoLabelingResult | list:
        """
        Predict shapes from image
        """
        if image is None:
            return []
        try:
            image = qt_img_to_rgb_cv_img(image, image_path)
        except Exception as e:  # noqa
            logging.warning("Could not inference model")
            logging.warning(e)
            return []

        inference_frame, scale = self.preprocess(image)
        model_inputs, _ = self.task_processor.create_input(
            inference_frame, self.input_shape
        )
        outputs = self.model.test_step(model_inputs)[0]
        results = self.postprocess(outputs, scale)
        shapes = []
        for bbox, score, cls in results:
            # Make sure to close
            shape = Shape(flags={}, shape_type=self.output_mode)
            for point in bbox:
                shape.add_point(QtCore.QPointF(point[0], point[1]))
            shape.fill_color = "#000000"
            shape.line_color = "#000000"
            shape.line_width = 1
            shape.selected = True
            shape.label = self.classes[cls]  # Assuming single class for Witty Product Segmentation
            shape.score = score
            shapes.append(shape)

        result = AutoLabelingResult(shapes, replace=True)
        return result

    def unload(self):
        del self.model

    def preprocess(self, bgr_frame: np.ndarray) -> tuple[np.ndarray, float]:
        """
        Preprocess the input frame for the model.
        """
        return bgr_frame, 1.0

    def postprocess(self, result: dict, scale: float) -> list[tuple[list[float], float, int]]:
        """
        Postprocess the model output to extract masks and bounding boxes.
        The model outputs masks and bounding boxes, which we need to normalize
        to the original frame size.
        """
        scores = result.pred_instances.scores
        selected_indices = scores > self.confidence_threshold

        bboxes = result.pred_instances.bboxes[selected_indices].numpy()
        scores = result.pred_instances.scores[selected_indices].numpy().tolist()
        classes = result.pred_instances.labels[selected_indices].numpy().tolist()
        # labels = [LABEL_FROM_CLASS_ID[c] for c in classes]

        bboxes_list = []
        for bbox in bboxes:
            x1, y1, x2, y2 = (bbox.astype(np.float32)*scale + 0.5).astype(np.int32)
            bboxes_list.append([[x1, y1], [x1, y2], [x2, y2], [x2, y1],])

        assert len(bboxes_list) == len(scores) == len(classes), (
            f"We must have the same number of bounding boxes ({len(bboxes_list)}) "
            f"as scores {len(scores)} and classes {len(classes)}."
        )

        results = [(bbox, score, cls) for bbox, score, cls in zip(bboxes_list, scores, classes)]

        return results

    def set_auto_labeling_conf(self, value: float) -> None:
        """set auto labeling confidence threshold"""
        if 0 < value < 1:
            self.confidence_threshold = value