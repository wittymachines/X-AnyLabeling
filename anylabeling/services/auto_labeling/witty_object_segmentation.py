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

class WittyObjectSegmentation(Model):
    """Model for Witty Object Segmentation using RTMDet."""
    class Meta:
        required_config_names = [
            "type",
            "name",
            "display_name",
            "model_path",
            "classes",
            "task"
        ]
        widgets = ["button_run", "output_select_combobox", "edit_conf"]
        output_modes = {
            "polygon": QCoreApplication.translate("Model", "Polygon"),
            "rotation": QCoreApplication.translate("Model", "Rotation"),
            "rectangle": QCoreApplication.translate("Model", "Rectangle"),
        }
        default_output_mode = "rotation"

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
        self.task = self.config["task"]

        self.available_tasks = ("egg_carton_top",)
        self.preprocess_functions = {"egg_carton_top": self.preprocess_egg_carton_top}
        self.postprocess_functions = {
            "egg_carton_top": self.postprocess_egg_carton_top,
        }
        
        if self.task not in self.available_tasks:
            raise ValueError(
                QCoreApplication.translate(
                    "Model",
                    f"Task {self.task} is not supported by {model_name} model.",
                )
            )


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

        inference_frame, scale = self.preprocess_functions[self.task](image)
        model_inputs, _ = self.task_processor.create_input(
            inference_frame, self.input_shape
        )
        outputs = self.model.test_step(model_inputs)[0]
        results = self.postprocess_functions[self.task](outputs, scale)
        shapes = []
        for contour, score, class_id in results:
            # Make sure to close
            contour.append(contour[0])
            shape = Shape(flags={}, shape_type=type)
            for point in contour:
                shape.add_point(QtCore.QPointF(point[0], point[1]))
            shape.closed = True
            shape.fill_color = "#000000"
            shape.line_color = "#000000"
            shape.line_width = 1
            shape.label = self.classes[class_id]
            shape.score = score
            shape.selected = False
            shapes.append(shape)

        result = AutoLabelingResult(shapes, replace=True)
        return result

    def unload(self):
        del self.model

    def preprocess_egg_carton_top(self, bgr_frame: np.ndarray) -> tuple[np.ndarray, float]:
        """
        Preprocess the input frame for the model.
        The model expects a 640x640 input size, so we resize the frame accordingly.
        """
        # Internally the model uses 640x640 as input size.
        # They pad the image internally, so we just make sure that the longest side is 640
        # and the other side is scaled accordingly.
        MAX_SIZE = 640
        original_height, original_width = bgr_frame.shape[:2]
        scale = MAX_SIZE / max(original_height, original_width)
        height = int(original_height * scale)
        width = int(original_width * scale)
        inference_frame = cv2.resize(bgr_frame, dsize=(width, height))
        return inference_frame, 1.0 / scale

    def postprocess_egg_carton_top(self, result: dict, scale: float) -> list[float, list[float]]:
        """
        Postprocess the model output to extract masks and bounding boxes.
        The model outputs masks and bounding boxes, which we need to normalize
        to the original frame size.
        """
        scores = result.pred_instances.scores
        selected_indices = scores > self.confidence_threshold

        scores = result.pred_instances.scores[selected_indices].numpy().tolist()
        classes = result.pred_instances.labels[selected_indices].numpy().tolist()
        masks = result.pred_instances.masks[selected_indices]
        bboxes = result.pred_instances.bboxes[selected_indices].numpy()

        if self.output_mode in ["polygon", "rotation"]:
            contours_list = []
            for i, mask in enumerate(masks):
                # Convert to NumPy and then to uint8 (0 or 255)
                mask_np = mask.numpy().astype(np.uint8) * 255
                # Resize mask to original size
                mask_np = cv2.resize(mask_np, None, fx=scale, fy=scale)

                # Find contours. cv2.findContours may return different values based on your OpenCV version.
                contours, _ = cv2.findContours(
                    mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )

                if len(contours) > 0:
                    # take the contour with the largest area
                    contours = sorted(contours, key=cv2.contourArea, reverse=True)
                    chosen_contour = contours[0][:, 0, :]
                    contours_list.append((chosen_contour.tolist(), scores[i], classes[i]))

        if self.output_mode == "rotation":
            # Get oriented bounding boxes from contours
            oriented_bboxes = []
            for contour, score, class_id in contours_list:
                # Convert contour to bounding box
                rect = cv2.minAreaRect(np.array(contour))
                box = cv2.boxPoints(rect).astype(np.int32)
                oriented_bboxes.append((box.tolist(), score, class_id))
            return oriented_bboxes
        elif self.output_mode == "rectangle":
            bboxes_list = []
            for bbox, score, class_id in zip(bboxes, scores, classes):
                x1, y1, x2, y2 = (bbox.astype(np.float32)*scale + 0.5).astype(np.int32)
                bboxes_list.append(([[x1, y1], [x1, y2], [x2, y2], [x2, y1],]), score, class_id)
            return bboxes_list
        else:
            return contours_list  # Return contours for polygon output mode

    def set_auto_labeling_conf(self, value: float) -> None:
        """set auto labeling confidence threshold"""
        if 0 < value < 1:
            self.confidence_threshold = value