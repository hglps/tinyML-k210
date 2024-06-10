# Usage:

Run ```python kmodel_manager.py --type [model type] --target [model path] --output [output path]``` for kmodel file creation from other models.

## model type(string)
Select between the options
* 'onnx'
* 'uint8_tflite'
* 'float32_preprocessing'
* 'float32_tflite'

## model path(string)
String of the path where the model is located.

```*.onnx``` and ```*.tflite``` files only

e.g. "Documents/models/model.tflite" or "Documents/models/model.onnx"

## output path(string)
String of the path where the output model will be saved. Should be a ```.kmodel``` file. For example, "output_model.kmodel"
