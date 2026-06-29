# Ultralytics YOLO 🚀, AGPL-3.0 license

from ultralytics.engine.model import Model
from ultralytics.nn.tasks import ClassificationModel, DetectionModel, OBBModel, PoseModel, SegmentationModel


class YOLO(Model):
    """YOLO (You Only Look Once) object detection model."""

    def __init__(self, model="yolov8n.pt", task=None, verbose=False):
        """Initialize a standard YOLO model."""
        super().__init__(model=model, task=task, verbose=verbose)

    def _smart_load(self, key: str):
        """Load model classes eagerly and train/predict helpers only when requested."""
        model_map = {
            "classify": ClassificationModel,
            "detect": DetectionModel,
            "segment": SegmentationModel,
            "pose": PoseModel,
            "obb": OBBModel,
        }
        if key == "model":
            return model_map[self.task]
        if self.task == "detect":
            from ultralytics.models.yolo.detect import DetectionPredictor, DetectionTrainer, DetectionValidator

            return {"predictor": DetectionPredictor, "trainer": DetectionTrainer, "validator": DetectionValidator}[key]
        if self.task == "segment":
            from ultralytics.models.yolo.segment import SegmentationPredictor, SegmentationTrainer, SegmentationValidator

            return {"predictor": SegmentationPredictor, "trainer": SegmentationTrainer, "validator": SegmentationValidator}[key]
        if self.task == "classify":
            from ultralytics.models.yolo.classify import ClassificationPredictor, ClassificationTrainer, ClassificationValidator

            return {"predictor": ClassificationPredictor, "trainer": ClassificationTrainer, "validator": ClassificationValidator}[key]
        if self.task == "pose":
            from ultralytics.models.yolo.pose import PosePredictor, PoseTrainer, PoseValidator

            return {"predictor": PosePredictor, "trainer": PoseTrainer, "validator": PoseValidator}[key]
        if self.task == "obb":
            from ultralytics.models.yolo.obb import OBBPredictor, OBBTrainer, OBBValidator

            return {"predictor": OBBPredictor, "trainer": OBBTrainer, "validator": OBBValidator}[key]
        return super()._smart_load(key)

    @property
    def task_map(self):
        """Map head to model, trainer, validator, and predictor classes."""
        from ultralytics.models.yolo.classify import ClassificationPredictor, ClassificationTrainer, ClassificationValidator
        from ultralytics.models.yolo.detect import DetectionPredictor, DetectionTrainer, DetectionValidator
        from ultralytics.models.yolo.obb import OBBPredictor, OBBTrainer, OBBValidator
        from ultralytics.models.yolo.pose import PosePredictor, PoseTrainer, PoseValidator
        from ultralytics.models.yolo.segment import SegmentationPredictor, SegmentationTrainer, SegmentationValidator

        return {
            "classify": {
                "model": ClassificationModel,
                "trainer": ClassificationTrainer,
                "validator": ClassificationValidator,
                "predictor": ClassificationPredictor,
            },
            "detect": {
                "model": DetectionModel,
                "trainer": DetectionTrainer,
                "validator": DetectionValidator,
                "predictor": DetectionPredictor,
            },
            "segment": {
                "model": SegmentationModel,
                "trainer": SegmentationTrainer,
                "validator": SegmentationValidator,
                "predictor": SegmentationPredictor,
            },
            "pose": {
                "model": PoseModel,
                "trainer": PoseTrainer,
                "validator": PoseValidator,
                "predictor": PosePredictor,
            },
            "obb": {
                "model": OBBModel,
                "trainer": OBBTrainer,
                "validator": OBBValidator,
                "predictor": OBBPredictor,
            },
        }
