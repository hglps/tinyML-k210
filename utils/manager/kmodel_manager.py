import argparse
from kmodel_generator import KModelGenerator
from kmodel_approaches import *

class KModelManager:
    def __init__(self) -> None:
        pass
    
    def run(self) -> None:
        parser = argparse.ArgumentParser()
        parser.add_argument('--type', type=str, required=True, choices=['onnx', 'uint8_tflite', 'float32_preprocessing', 'float32_tflite'], help="model file type, e.g. onnx, uint8_tflite, int8_tflite, float32_preprocessing, float32_tflite")
        parser.add_argument('--target', type=str, required=True, help="model filename")
        parser.add_argument('--output', required=True, type=str, help="output kmodel filename")
        args = parser.parse_args()

        model_type = args.type
        model_file = args.target
        output_filename = args.output

        approach = ['onnx', 'uint8_tflite', 'float32_preprocessing', 'float32_tflite']
        
        approach = {
            'onnx': ONNX_Simplified(),
            'uint8_tflite': TFLite_uint8(),
            'float32_preprocessing': TFLite_float32_preprocessing(),
            'float32_tflite' : TFLite_float32()
        }
        
        approach_choice = approach[model_type]
        
        manager = KModelGenerator(approach_choice, model_file=model_file, output_filename=output_filename)
        print("Approach set to ", " ".join(model_type.split('_')), " model")
        manager.run()


if __name__ == "__main__":
    kmodel_manager = KModelManager()
    kmodel_manager.run()
