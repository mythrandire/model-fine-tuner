import sys
import os
from datetime import datetime
import logging
import torch
import fiftyone as fo
import fiftyone.operators as foo
import fiftyone.operators.types as types
import fiftyone.core.storage as fos

logger = logging.getLogger("fiftyone.core.collections")


""" 
This plugin is a simple example of how to finetune YOLOv8 object detection
model using FiftyOne Teams.
"""

# Set to writable working dir in teams-do pods
TRAIN_ROOT = "/tmp/yolo/"
MODEL_ROOT = os.path.join(TRAIN_ROOT, "models")
DATA_ROOT = os.path.join(TRAIN_ROOT, "data")
PROJECT_ROOT = os.path.join(TRAIN_ROOT, "projects")


class ModelFineTuner2(foo.Operator):
    @property
    def config(self):
        """
        Defines how the FiftyOne App should display this operator (name,
        label, whether it shows in the operator browser, etc).
        """
        return foo.OperatorConfig(
            name="model-fine-tuner-2",  # Must match what's in fiftyone.yml
            label="Finetune models like YOLOv8",
            description="Finetune a YOLOv8 model on the current view",
            icon="build_circle",  # Material UI icon, or path to custom icon
            allow_immediate_execution=True,  # AL: maybe should be false
            allow_delegated_execution=True,
            default_choice_to_delegated=True,  # AL: prob should be true
        )

    def resolve_placement(self, ctx):
        """
        Optional convenience: place a button in the App so the user can
        click to open this operator's input form.
        """
        return types.Placement(
            types.Places.SAMPLES_GRID_SECONDARY_ACTIONS,
            types.Button(
                label="Finetune YOLOv8",
                icon="build_circle",
                prompt=True,  # always show the operator's input prompt
            ),
        )

    def resolve_input(self, ctx):
        """
        Collect the inputs we need from the user. This defines the form
        that appears in the FiftyOne App when the operator is invoked.
        """
        inputs = types.Object()

        dataset = ctx.dataset
        schema = dataset.get_field_schema(
            ftype=fo.EmbeddedDocumentField, embedded_doc_type=fo.Detections
        )
        fields = schema.keys()
        field_choices = types.DropdownView()
        for field_name in fields:
            field_choices.add_choice(field_name, label=field_name)

        inputs.enum(
            "det_field",
            field_choices.values(),
            required=True,
            label="detections field",
            view=field_choices,
        )

        # 1) Path to existing YOLOv8 model weights (local or cloud)
        inputs.str(
            "weights_path",
            default="gs://voxel51-demo-fiftyone-ai/yolo/yolov8n.pt",
            required=True,
            description="S3 or GCS path (or local) to initial YOLOv8 *.pt weights",
            label="Initial weights input URI",
        )

        # 2) Choice to export to CoreML
        inputs.bool(
            "to_coreml",
            label="Check to export as a CoreML model",
            view=types.CheckboxView(),
            default=False,
        )

        # 3) Path to store the new finetuned model weights (local or cloud)
        inputs.str(
            "export_uri",
            default="gs://voxel51-demo-fiftyone-ai/yolo/yolov8n_finetuned.pt",
            required=True,
            description="S3 or GCS path (or local) to save finetuned weights",
            label="Finetuned weights output URI",
        )

        # 4) Path to store CoreML model
        inputs.str(
            "core_ml_export_uri",
            default="gs://voxel51-demo-fiftyone-ai/yolo/yolov8n_finetuned.mlpackage",
            required=False,
            description="S3 or GCS path (or local) to save finetuned CoreML weights",
            label="(Optional) Finetuned weights (CoreML format) output URI",
        )

        # 5) Can add more hyperparameters here
        inputs.int(
            "epochs",
            default=1,
            description="Number of epochs to train",
            label="Training epochs",
        )

        # 6) CUDA target device
        inputs.int(
            "target_device_index",
            default=0,
            required=False,
            description="CUDA Device number to train on. Optional, defaults to device cuda:0",
            label="Target CUDA device number",
        )

        return types.Property(
            inputs,
            view=types.View(label="Finetune YOLOv8"),
        )

    def execute(self, ctx):
        """
        Main logic that:
        - Checks that `weights_path` is a YOLOv8 model
        - Exports the current view to YOLO dataset format
        - Runs YOLOv8 training
        - Saves best weights to user-supplied `export_uri`
        """

        from ultralytics import YOLO

        det_field = ctx.params["det_field"]
        weights_path = ctx.params["weights_path"]
        export_uri = ctx.params["export_uri"]
        epochs = ctx.params["epochs"]
        target_device_index = ctx.params["target_device_index"]
        to_coreml = ctx.params.get("to_coreml", False)
        core_ml_export_uri = ctx.params["core_ml_export_uri"]

        dataset = ctx.dataset
        det_label_field = f"{det_field}.detections.label"
        classes = dataset.distinct(det_label_field)

        # --- Step 1: Verify the weights_path is YOLOv8 ---
        local_weights_path = os.path.join(MODEL_ROOT, os.path.basename(weights_path))
        fos.copy_file(weights_path, local_weights_path)
        # model = self._try_load_model(local_weights_path)
        str = f"Model downloaded to: {local_weights_path}"
        logger.warning(str)

        dataset_root = os.path.join(DATA_ROOT, dataset.name)
        now_time = datetime.now().strftime("%Y%m%dT%H%M%S")
        export_dir = os.path.join(dataset_root, now_time)
        str = f"Exporting to: {export_dir}"
        logger.warning(str)

        export_yolo_data(
            ctx.dataset,
            export_dir,
            classes=classes,
            label_field=det_field,
            split=["train", "val"],
        )

        # The dataset.yaml that YOLO wants is typically `export_dir/dataset.yaml`
        data_yaml = f"{export_dir}/dataset.yaml"
        assert fos.exists(data_yaml), f"Failed to export dataset to {data_yaml}"

        logger.warning("Starting training")
        ctx.set_progress(progress=0.1, label="Dataset exported. Starting training...")

        # --- Step 3: Finetune YOLOv8 model with ultralytics ---
        # We'll re-init the model with the user-provided weights
        model = YOLO(local_weights_path)

        cuda_device_count = torch.cuda.device_count()
        logger.warning(f"Number of CUDA devices found: {cuda_device_count}")

        if cuda_device_count > 1 and target_device_index <= cuda_device_count:
            target_device = f"cuda:{target_device_index}"
            model.to(target_device)
        else:
            model.to("cuda:0")

        # `train(...)` expects various kwargs. Adjust as you prefer.
        # `epochs=epochs` might require more GPU time if large
        results = model.train(
            data=data_yaml,
            epochs=epochs,
            imgsz=640,
            name="finetuned",
            project=PROJECT_ROOT,
            exist_ok=True,
        )

        # Once training is done, the best weights are typically in
        #   runs/detect/finetuned/weights/best.pt
        # or
        #   runs/detect/train/weights/best.pt
        # `results.save_dir` is the folder YOLO used for the last run
        best_weights = os.path.join(results.save_dir, "weights", "best.pt")

        logger.warning("Ending training")

        ctx.set_progress(
            progress=0.9, label="Training complete. Saving final weights..."
        )

        # --- Step 4: Save to user-supplied path (export_uri) ---
        # If `export_uri` is on local disk, we can just copy. If it’s S3 or GCS,
        # you might need a custom storage library. For simplicity, below is a naive local example:
        fos.copy_file(best_weights, export_uri)
        logger.warning(f"Saved finetuned weights to {export_uri}")

        if to_coreml:
            coreml_path = os.path.join(MODEL_ROOT, "yolov8n.mlpackage")
            try:
                model.export(format="coreml", nms=True)
                fos.copy_file(coreml_path, core_ml_export_uri)
            except ModuleNotFoundError as MNFE:
                ctx.log(f"{MNFE}")

        # ctx.set_progress(progress=1.0, label="Done!")
        return {
            "finetuned_weights_path": export_uri,
            "status": "success",
            "cuda_device_count": cuda_device_count,
        }

    def resolve_output(self, ctx):
        """
        Display any final outputs in the App after training completes.
        """
        outputs = types.Object()
        outputs.str(
            "finetuned_weights_path",
            label="Finetuned weights saved to",
        )
        outputs.str(
            "status",
            label="Finetuning status",
        )

        outputs.str("cuda_device_count", label="Number of CUDA devices")
        return types.Property(
            outputs,
            view=types.View(label="Finetune Results"),
        )


class ApplyRemoteModel2(foo.Operator):
    @property
    def config(self):
        """
        Defines how the FiftyOne App should display this operator (name,
        label, whether it shows in the operator browser, etc).
        """
        return foo.OperatorConfig(
            name="apply-remote-model-2",  # Must match what's in fiftyone.yml
            label="Run YOLO model with cloud-backed weights",
            description="Run inference with a YOLOv8 model on the current view using remotely stored weights",
            icon="input",  # Material UI icon, or path to custom icon
            allow_immediate_execution=True,
            allow_delegated_execution=True,
            default_choice_to_delegated=True,
        )

    def resolve_placement(self, ctx):
        """
        Optional convenience: place a button in the App so the user can
        click to open this operator's input form.
        """
        return types.Placement(
            types.Places.SAMPLES_GRID_SECONDARY_ACTIONS,
            types.Button(
                label="Apply YOLO model",
                icon="input",
                prompt=True,  # always show the operator's input prompt
            ),
        )

    def resolve_input(self, ctx):
        """
        Collect the inputs we need from the user. This defines the form
        that appears in the FiftyOne App when the operator is invoked.
        """
        inputs = types.Object()

        inputs.str(
            "det_field",
            required=True,
            label="Detections field",
        )

        # 1) Local filepath to existing YOLOv8 model weights
        inputs.str(
            "weights_path",
            default="gs://voxel51-demo-fiftyone-ai/yolo/yolov8n_finetuned.pt",
            required=True,
            description="Filepath to the YOLOv8 *.pt weights file",
            label="YOLOv8 weights",
        )

        # 2) CUDA target device
        inputs.int(
            "target_device_index",
            default=0,
            required=False,
            description="CUDA Device number to train on. Optional, defaults to device cuda:0",
            label="Target CUDA device number",
        )

        return types.Property(
            inputs,
            view=types.View(label="Run inference YOLOv8"),
        )

    def execute(self, ctx):
        """ """

        from ultralytics import YOLO

        det_field = ctx.params["det_field"]
        weights_path = ctx.params["weights_path"]
        target_device_index = ctx.params["target_device_index"]

        dataset = ctx.dataset

        # --- Step 1: Verify the weights_path is YOLOv8 ---
        local_weights_path = os.path.join(MODEL_ROOT, os.path.basename(weights_path))
        fos.copy_file(weights_path, local_weights_path)
        # model = self._try_load_model(local_weights_path)
        str = f"Model downloaded to: {local_weights_path}"
        logger.warning(str)

        cuda_device_count = torch.cuda.device_count()
        logger.warning(f"Number of CUDA devices found: {cuda_device_count}")

        model = YOLO(local_weights_path)

        if cuda_device_count > 1 and target_device_index <= cuda_device_count:
            target_device = f"cuda:{target_device_index}"
            model.to(target_device)
        else:
            model.to("cuda:0")

        ctx.dataset.apply_model(model, label_field=det_field)

        logger.warning("Ending inference")

        return {"status": "success", "cuda_device_count": cuda_device_count}

    def resolve_output(self, ctx):
        """
        Display any final outputs in the App after training completes.
        """
        outputs = types.Object()
        outputs.str(
            "status",
            label="Finetuning status",
        )

        outputs.str("cuda_device_count", label="Number of CUDA devices")
        return types.Property(
            outputs,
            view=types.View(label="Finetune Results"),
        )


def register(plugin):
    """
    Called by FiftyOne to discover and register your plugin’s operators/panels.
    """
    plugin.register(ModelFineTuner2)
    plugin.register(ApplyRemoteModel2)


#
# Helper functions referenced above.
#


def export_yolo_data(
    samples, export_dir, classes, label_field="ground_truth", split=None
):

    if type(split) == list:
        splits = split
        for split in splits:
            export_yolo_data(samples, export_dir, classes, label_field, split)
    else:
        if split is None:
            split_view = samples
            split = "val"
        else:
            split_view = samples.match_tags(split)

        split_view.export(
            export_dir=export_dir,
            dataset_type=fo.types.YOLOv5Dataset,
            label_field=label_field,
            classes=classes,
            split=split,
        )
