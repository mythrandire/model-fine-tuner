import os
from datetime import datetime
import fiftyone as fo
import fiftyone.operators as foo
import fiftyone.operators.types as types
import fiftyone.core.storage as fos

#from ultralytics import YOLO


TRAIN_ROOT = '/tmp/yolo_train'
MODEL_ROOT = os.path.join(TRAIN_ROOT, 'models')
DATA_ROOT = os.path.join(TRAIN_ROOT, 'data')
PROJECT_ROOT = os.path.join(TRAIN_ROOT, 'projects')



#
# We assume you have ultralytics installed
#
#try:
#    from ultralytics import YOLO
#except ImportError:
#    raise ImportError(
#        "You must install ultralytics to use this plugin. "
#        "Add `ultralytics` to your plugin's requirements.txt."
#    )


class ModelFineTuner(foo.Operator):
    @property
    def config(self):
        """
        Defines how the FiftyOne App should display this operator (name,
        label, whether it shows in the operator browser, etc).
        """
        return foo.OperatorConfig(
            name="model-fine-tuner",  # Must match what's in fiftyone.yml
            label="Finetune models like YOLOv8",
            description="Finetune a YOLOv8 model on the current view",
            icon="build_circle",           # Material UI icon, or path to custom icon
            allow_immediate_execution=True,
            allow_delegated_execution=True,
            default_choice_to_delegated=False,
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
        schema = dataset.get_field_schema(ftype=fo.EmbeddedDocumentField,
                                          embedded_doc_type=fo.Detections)
        fields = schema.keys()
        field_choices = types.DropdownView()
        for field_name in fields:
            field_choices.add_choice(field_name, label=field_name)

        inputs.enum(
            'det_field',
            field_choices.values(),
            required=True,
            label='detections field',
            view=field_choices,
        )

        # 1) Local filepath to existing YOLOv8 model weights
        inputs.str(
            "weights_path",
            default='gs://voxel51-test/al/yolo/yolov8n.pt',
            required=True,
            description="Local filepath to the YOLOv8 *.pt weights file",
            label="Local YOLOv8 weights",
        )

        # 2) Path to store the new finetuned model weights (S3/GCS/whatever)
        inputs.str(
            "export_uri",
            default='gs://voxel51-test/al/yolo/yolov8n_finetuned.pt',
            required=True,
            description="S3 or GCS path (or local) to save finetuned weights, e.g. s3://mybucket/finetuned.pt",
            label="Finetuned weights output URI",
        )

        # 3) Can add more hyperparameters here
        inputs.int(
            "epochs",
            default=1,
            description="Number of epochs to train",
            label="Training epochs",
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
        
        dataset = ctx.dataset
        det_label_field = f'{det_field}.detections.label'
        classes = dataset.distinct(det_label_field)

        # --- Step 1: Verify the weights_path is YOLOv8 ---
        local_weights_path = os.path.join(MODEL_ROOT, os.path.basename(weights_path))
        fos.copy_file(weights_path, local_weights_path)
        #model = self._try_load_model(local_weights_path)
        str = f'Model downloaded to: {local_weights_path}'
        ctx.log(str)
        print(str)

        dataset_root = os.path.join(DATA_ROOT, dataset.name)
        fos.ensure_dir(dataset_root)
        str = f'Exporting to: {dataset_root}'
        ctx.log(str)
        print(str)
       
        export_yolo_data(ctx.dataset, 
                         dataset_root,
                         classes=classes, 
                         label_field=det_field,
                         split=["train", "val"])

        # The dataset.yaml that YOLO wants is typically `export_dir/dataset.yaml`
        data_yaml = f"{dataset_root}/dataset.yaml"
        assert fos.exists(data_yaml), f"Failed to export dataset to {data_yaml}"

        print("Starting training")
        #ctx.set_progress(progress=0.1, label="Dataset exported. Starting training...")

        # --- Step 3: Finetune YOLOv8 model with ultralytics ---
        # We'll re-init the model with the user-provided weights
        model = YOLO(local_weights_path)
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

        print("Endg training")

        #ctx.set_progress(progress=0.9, label="Training complete. Saving final weights...")

        # --- Step 4: Save to user-supplied path (export_uri) ---
        # If `export_uri` is on local disk, we can just copy. If it’s S3 or GCS,
        # you might need a custom storage library. For simplicity, below is a naive local example:
        fos.copy_file(best_weights, export_uri)
        
        print(f"Saved finetuned weights to {export_uri}")

        #ctx.set_progress(progress=1.0, label="Done!")
        return {
            "finetuned_weights_path": export_uri,
            "status": "success",
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
        return types.Property(
            outputs,
            view=types.View(label="Finetune Results"),
        )

    '''
    def _try_load_model(self, weights_path):
        """
        Helper that attempts to load YOLO weights using ultralytics.
        If load fails or if the model is not YOLOv8, we raise an error.
        """
        try:
            # If `YOLO(...)` can parse it, we’ll assume it’s YOLOv8
            _ = YOLO(weights_path)
        except Exception as e:
            raise ValueError(
                f"Failed to load model from {weights_path}. "
                "Ensure this is a valid YOLOv8 .pt file. "
                f"Error: {e}"
            )
        return True
    '''


def register(plugin):
    """
    Called by FiftyOne to discover and register your plugin’s operators/panels.
    """
    plugin.register(ModelFineTuner)


#
#Helper functions referenced above. 
# 

def _export_view_to_yolo(view, label_field, export_dir):
    """
    Example minimal YOLO export using FiftyOne's built-in YOLO v5/v8 exporter.
    The user is responsible for ensuring that the dataset has bounding boxes
    or relevant labels that YOLO supports.
    """
    # We must specify label_field(s) containing the object detections
    # and the classes (if relevant).
    #
    # If your dataset has multiple detection fields, or if you want
    # specific classes only, adapt as you wish.
    #
    # For brevity, let’s assume your ground-truth bounding boxes are
    # in a field named "ground_truth", and all classes are relevant:
    #label_field = "ground_truth"

    view.export(
        export_dir=export_dir,
        dataset_type=fo.types.YOLOv5Dataset,  # also works for YOLOv8
        label_field=label_field,
        overwrite=True,
    )


'''
def _copy_local_file(src, dst):
    """
    Minimal local file copy. If your `dst` is a cloud path, you must
    implement your own upload logic via boto3, GCS library, etc.
    """
    import shutil

    # If the user typed something like s3://..., you'd handle that differently
    if dst.startswith("s3://") or dst.startswith("gs://"):
        raise NotImplementedError(
            "For S3/GCS uploads, implement your desired storage logic here!"
        )

    shutil.copy(src, dst)
'''

def export_yolo_data(
    samples, 
    export_dir, 
    classes, 
    label_field = "ground_truth", 
    split = None
    ):

    if type(split) == list:
        splits = split
        for split in splits:
            export_yolo_data(
                samples, 
                export_dir, 
                classes, 
                label_field, 
                split
            )   
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
            split=split
        )
