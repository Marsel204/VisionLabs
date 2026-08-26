"""Dense multi-scale vehicle proposal inference."""

from __future__ import annotations

from dataclasses import dataclass, field

from app.services.annotation.domain import (
    TARGET_CLASSES,
    Annotation,
    AnnotationDocument,
    AnnotationSource,
    BoundingBox,
)
from app.services.inference.grounding import grounding_class, prompt_class, prompt_variants


@dataclass(frozen=True, slots=True)
class ClassRule:
    """Per-class acceptance rules for raw detector proposals."""

    min_confidence: float
    min_area: float
    max_area: float
    min_aspect: float
    max_aspect: float


DEFAULT_CLASS_RULES: dict[str, ClassRule] = {
    # Motorcycles in dense traffic are small and partially occluded; recall matters.
    "motorcycle": ClassRule(0.05, 0.00005, 0.75, 0.10, 4.0),
    "car": ClassRule(0.10, 0.0005, 0.85, 0.20, 4.0),
    # Buses and trucks are large; small or weak detections are almost always
    # misclassified cars/vans or background clutter.
    "bus": ClassRule(0.45, 0.02, 0.85, 0.30, 3.5),
    "truck": ClassRule(0.40, 0.02, 0.85, 0.30, 3.5),
}


@dataclass(frozen=True, slots=True)
class DenseInferenceConfig:
    """High-recall settings for crowded traffic images."""

    tile_sizes: tuple[int, ...] = (512, 320)
    tile_overlap: float = 0.25
    yolo_image_size: int = 1280
    yolo_confidence: float = 0.05
    yolo_iou: float = 0.90
    dino_box_threshold: float = 0.12
    dino_text_threshold: float = 0.18
    nms_iou: float = 0.30
    max_vehicle_area: float = 0.85
    enabled_classes: frozenset[str] = frozenset(TARGET_CLASSES)
    class_rules: dict[str, ClassRule] = field(
        default_factory=lambda: dict(DEFAULT_CLASS_RULES)
    )


class DenseMotorcycleInference:
    """Run recall-oriented YOLO and Grounding DINO proposals on dense scenes."""

    def __init__(
        self,
        grounding_model,
        grounding_processor,
        yolo_model,
        config: DenseInferenceConfig | None = None,
    ) -> None:  # type: ignore[no-untyped-def]
        self._grounding_model = grounding_model
        self._grounding_processor = grounding_processor
        self._yolo_model = yolo_model
        self.config = config or DenseInferenceConfig()

    def predict(
        self,
        document: AnnotationDocument,
        prompt: str,
        use_yolo: bool = True,
    ) -> tuple[list[Annotation], list[Annotation]]:
        """Return ``(yolo_predictions, dino_predictions)`` for one image."""
        from PIL import Image

        image = Image.open(document.image_path).convert("RGB")
        yolo = self._yolo_predictions(image, document) if use_yolo else []
        dino = self._dino_predictions(image, document, prompt, motorcycle_only=use_yolo)
        return yolo, dino

    def _yolo_predictions(self, image, document: AnnotationDocument) -> list[Annotation]:  # type: ignore[no-untyped-def]
        if self._yolo_model is None:
            return []
        from app.core.runtime import detect_gpu

        device = 0 if detect_gpu().device == "cuda" else "cpu"
        predictions: list[Annotation] = []
        for crop, offset_x, offset_y, _is_tile in self._crops(image):
            result = self._yolo_model(
                crop,
                device=device,
                imgsz=self.config.yolo_image_size,
                conf=self.config.yolo_confidence,
                iou=self.config.yolo_iou,
                verbose=False,
            )[0]
            names = self._yolo_model.names
            for box in result.boxes:
                class_name = str(names[int(box.cls[0])])
                if class_name not in TARGET_CLASSES:
                    continue
                left, top, right, bottom = box.xyxy[0].tolist()
                annotation = self._annotation(
                    class_name,
                    left + offset_x,
                    top + offset_y,
                    right + offset_x,
                    bottom + offset_y,
                    float(box.conf[0]),
                    AnnotationSource.YOLO,
                    document,
                )
                if annotation is not None and self._accept(annotation):
                    predictions.append(annotation)
        return self._nms(predictions)

    def _dino_predictions(
        self,
        image,
        document: AnnotationDocument,
        prompt: str,
        motorcycle_only: bool = False,
    ) -> list[Annotation]:
        import torch

        if self._grounding_model is None or self._grounding_processor is None:
            return []
        predictions: list[Annotation] = []
        variants = [
            variant
            for variant in prompt_variants(prompt)
            if prompt_class(variant) in self.config.enabled_classes
        ]
        if motorcycle_only:
            variants = [
                variant
                for variant in variants
                if prompt_class(variant) == "motorcycle"
            ]
        compound_prompt = " . ".join(dict.fromkeys(v.strip(" .") for v in variants)) + " ."
        if not compound_prompt.strip(" ."):
            return []

        for crop, offset_x, offset_y, is_tile in self._crops(image):
            inputs = self._grounding_processor(
                images=crop,
                text=compound_prompt,
                return_tensors="pt",
            )
            device = next(self._grounding_model.parameters()).device
            inputs = {
                key: value.to(device) if hasattr(value, "to") else value
                for key, value in inputs.items()
            }
            with torch.no_grad():
                outputs = self._grounding_model(**inputs)
            result = self._post_process(outputs, inputs["input_ids"], crop)
            labels = result.get("text_labels", result.get("labels", ()))
            for index, (box, score) in enumerate(
                zip(result["boxes"], result["scores"], strict=True)
            ):
                if index >= len(labels):
                    continue
                class_name = grounding_class(str(labels[index]))
                if class_name is None or class_name not in self.config.enabled_classes:
                    continue
                if motorcycle_only and class_name != "motorcycle":
                    continue
                score_value = float(score)
                left, top, right, bottom = box.tolist()
                if is_tile and self._tile_artifact(left, top, right, bottom, crop):
                    continue
                annotation = self._annotation(
                    class_name,
                    left + offset_x,
                    top + offset_y,
                    right + offset_x,
                    bottom + offset_y,
                    score_value,
                    AnnotationSource.GROUNDING_DINO,
                    document,
                )
                if annotation is not None and self._accept(annotation):
                    predictions.append(annotation)
        return self._nms(predictions)

    def _accept(self, annotation: Annotation) -> bool:
        """Apply per-class enablement, confidence, area, and aspect rules."""
        if annotation.class_name not in self.config.enabled_classes:
            return False
        rule = self.config.class_rules.get(annotation.class_name)
        if rule is None:
            return True
        confidence = annotation.confidence if annotation.confidence is not None else 0.0
        if confidence < rule.min_confidence:
            return False
        box = annotation.box
        if not rule.min_area <= box.area <= rule.max_area:
            return False
        aspect = box.width / box.height
        return rule.min_aspect <= aspect <= rule.max_aspect

    def _post_process(self, outputs, input_ids, crop):  # type: ignore[no-untyped-def]
        kwargs = {
            "text_threshold": self.config.dino_text_threshold,
            "target_sizes": [(crop.height, crop.width)],
        }
        try:
            return self._grounding_processor.post_process_grounded_object_detection(
                outputs,
                input_ids,
                threshold=self.config.dino_box_threshold,
                **kwargs,
            )[0]
        except TypeError:
            return self._grounding_processor.post_process_grounded_object_detection(
                outputs,
                input_ids,
                box_threshold=self.config.dino_box_threshold,
                **kwargs,
            )[0]

    @staticmethod
    def _label_class(label: str, expected_class: str) -> str | None:
        normalized = label.lower().strip(" .")
        detected = grounding_class(normalized)
        if detected == expected_class:
            return detected
        return None

    def _crops(self, image):  # type: ignore[no-untyped-def]
        crops = [(image, 0, 0, False)]
        for tile_size in self.config.tile_sizes:
            if image.width <= tile_size and image.height <= tile_size:
                continue
            stride = max(1, int(tile_size * (1.0 - self.config.tile_overlap)))
            for top in self._positions(image.height, tile_size, stride):
                for left in self._positions(image.width, tile_size, stride):
                    right = min(image.width, left + tile_size)
                    bottom = min(image.height, top + tile_size)
                    crops.append((image.crop((left, top, right, bottom)), left, top, True))
        return crops

    @staticmethod
    def _positions(length: int, tile_size: int, stride: int) -> list[int]:
        positions = list(range(0, max(1, length - tile_size + 1), stride))
        final = max(0, length - tile_size)
        if not positions or positions[-1] != final:
            positions.append(final)
        return positions

    @staticmethod
    def _tile_artifact(left: float, top: float, right: float, bottom: float, crop) -> bool:  # type: ignore[no-untyped-def]
        width = right - left
        height = bottom - top
        return (
            left <= 2
            or top <= 2
            or right >= crop.width - 2
            or bottom >= crop.height - 2
        ) and (width >= crop.width * 0.8 or height >= crop.height * 0.8)

    @staticmethod
    def _annotation(
        class_name: str,
        left: float,
        top: float,
        right: float,
        bottom: float,
        confidence: float,
        source: AnnotationSource,
        document: AnnotationDocument,
    ) -> Annotation | None:
        left = max(0.0, min(float(left), document.image_width))
        top = max(0.0, min(float(top), document.image_height))
        right = max(0.0, min(float(right), document.image_width))
        bottom = max(0.0, min(float(bottom), document.image_height))
        if left >= right or top >= bottom:
            return None
        return Annotation(
            class_name,
            BoundingBox(
                left / document.image_width,
                top / document.image_height,
                right / document.image_width,
                bottom / document.image_height,
            ),
            confidence=confidence,
            source=source,
        )

    def _nms(self, predictions: list[Annotation]) -> list[Annotation]:
        kept: list[Annotation] = []
        for prediction in sorted(
            predictions,
            key=lambda item: item.confidence if item.confidence is not None else 0.0,
            reverse=True,
        ):
            if any(
                item.class_name == prediction.class_name
                and self._iou(item.box, prediction.box) >= self.config.nms_iou
                for item in kept
            ):
                continue
            kept.append(prediction)
        return kept

    @staticmethod
    def _iou(first: BoundingBox, second: BoundingBox) -> float:
        left = max(first.left, second.left)
        top = max(first.top, second.top)
        right = min(first.right, second.right)
        bottom = min(first.bottom, second.bottom)
        intersection = max(0.0, right - left) * max(0.0, bottom - top)
        union = first.area + second.area - intersection
        return intersection / union if union else 0.0
