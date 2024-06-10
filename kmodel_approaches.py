import os
import onnxsim
import onnx
import numpy as np
import nncase
from kmodel_generator import Strategy
from nncase_tools import NNCaseTools

class ONNX_Simplified(Strategy):
    
    def __init__(self) -> None:
        super().__init__()
        self.tools = NNCaseTools()
    
    def parse_model_input_output(self, model_file: str):
        onnx_model = onnx.load(model_file)
        input_all = [node.name for node in onnx_model.graph.input]
        input_initializer = [node.name for node in onnx_model.graph.initializer]
        input_names = list(set(input_all) - set(input_initializer))
        input_tensors = [node for node in onnx_model.graph.input if node.name in input_names]

        # input
        inputs= []
        for _, e in enumerate(input_tensors):
            onnx_type = e.type.tensor_type
            input_dict = {}
            input_dict['name'] = e.name
            input_dict['dtype'] = onnx.mapping.TENSOR_TYPE_TO_NP_TYPE[onnx_type.elem_type]
            input_dict['shape'] = [(i.dim_value if i.dim_value != 0 else d) for i, d in zip(
                onnx_type.shape.dim, [1, 3, 224, 224])]
            inputs.append(input_dict)


        return onnx_model, inputs

    def onnx_simplify(self, model_file: str):
        onnx_model, inputs = self.parse_model_input_output(model_file)
        onnx_model = onnx.shape_inference.infer_shapes(onnx_model)
        input_shapes = {}
        for input in inputs:
            input_shapes[input['name']] = input['shape']
    
        onnx_model, check = onnxsim.simplify(onnx_model, overwrite_input_shapes=input_shapes)
        # onnx_model, check = onnxsim.simplify(onnx_model, input_shapes=input_shapes)
        
        assert check, "Simplified ONNX model could not be validated"
    
        model_file = os.path.join(os.path.dirname(model_file), 'simplified.onnx')
        onnx.save_model(onnx_model, model_file)
        return model_file
    
    def generate_kmodel(self, model_file: str, output_filename: str) -> None:
        target = 'k210'
        # onnx simplify
        model_file = self.onnx_simplify(model_file)

        # compile_options
        print('compiling opts')
        compile_options = nncase.CompileOptions()
        compile_options.target = target
        compile_options.dump_ir = True
        compile_options.dump_asm = True
        compile_options.dump_dir = 'tmp'
        # extra
        compile_options.input_type = 'uint8'
        compile_options.input_layout = 'BCHW'
        compile_options.input_shape = [1, 3, 224, 224] # estava 640, 640
        compile_options.quant_type = 'uint8'  # uint8 or int8

        # compiler
        print('compiler')
        compiler = nncase.Compiler(compile_options)

        # import_options
        print('import_options')
        import_options = nncase.ImportOptions()

        # import
        print('import onnx')
        model_content = self.tools.read_model_file(model_file)
        
        compiler.import_onnx(model_content, import_options)

        # compile
        print('compiler.compile')
        compiler.compile()
        
        self.tools.write_kmodel_file(output_filename, compiler)


class TFLite_uint8(Strategy):
    
    def __init__(self) -> None:
        super().__init__()
        self.tools = NNCaseTools()

    def generate_data(self, shape, batch):
        shape[0] *= batch
        data = np.random.rand(*shape).astype(np.float32)
        return data

    def generate_kmodel(self, model_file: str, output_filename: str) -> None:
        input_shape = [1,224,224,3]
        target = 'k210'

        # compile_options
        compile_options = nncase.CompileOptions()
        compile_options.target = target
        compile_options.input_type = 'float32'
        compile_options.input_layout = 'NHWC'
        compile_options.output_layout = 'NHWC'
        compile_options.dump_ir = True
        compile_options.dump_asm = True
        compile_options.dump_dir = 'tmp'

        # compiler
        compiler = nncase.Compiler(compile_options)

        # import_options
        import_options = nncase.ImportOptions()

        # quantize model
        compile_options.quant_type = 'uint8' # or 'int8' 'int16'

        # # ptq_options
        # ptq_options = nncase.PTQTensorOptions()
        # ptq_options.samples_count = 10
        # ptq_options.set_tensor_data(self.generate_data(input_shape, ptq_options.samples_count).tobytes())

        # import
        model_content = self.tools.read_model_file(model_file=model_file)
        compiler.import_tflite(model_content, import_options)

        # compile
        # compiler.use_ptq(ptq_options)
        compiler.compile()

        self.tools.write_kmodel_file(output_filename, compiler)


class TFLite_float32(Strategy):
    def __init__(self) -> None:
        super().__init__()
        self.tools = NNCaseTools()
    
    def generate_kmodel(self, model_file: str, output_filename: str) -> None:
        target = 'k210'

        # compile_options
        compile_options = nncase.CompileOptions()
        compile_options.target = target
        compile_options.input_type = 'float32'  # or 'uint8' 'int8'
        compile_options.dump_ir = True
        compile_options.dump_asm = True
        compile_options.dump_dir = 'tmp'

        # compiler
        compiler = nncase.Compiler(compile_options)

        # import_options
        import_options = nncase.ImportOptions()

        # import
        model_content = self.tools.read_model_file(model_file)
        compiler.import_tflite(model_content, import_options)

        # compile
        compiler.compile()

        self.tools.write_kmodel_file(output_filename, compiler)


class TFLite_float32_preprocessing(Strategy):
    def __init__(self) -> None:
        super().__init__()
        self.tools = NNCaseTools()
    
    def generate_kmodel(self, model_file: str, output_filename: str) -> None:
        target = 'k210'

        # compile_options
        compile_options = nncase.CompileOptions()
        compile_options.target = target
        compile_options.input_type = 'float32'  # or 'uint8' 'int8'
        compile_options.preprocess = True # if False, the args below will unworked
        compile_options.swapRB = True
        compile_options.input_shape = [1,224,224,3] # keep layout same as input layout
        compile_options.input_layout = 'NHWC'
        compile_options.output_layout = 'NHWC'
        compile_options.mean = [0,0,0]
        compile_options.std = [1,1,1]
        compile_options.input_range = [0,1]
        compile_options.letterbox_value = 114. # pad what you want
        compile_options.dump_ir = True
        compile_options.dump_asm = True
        compile_options.dump_dir = 'tmp'

        # compiler
        compiler = nncase.Compiler(compile_options)

        # import_options
        import_options = nncase.ImportOptions()

        # import
        model_content = self.tools.read_model_file(model_file)
        compiler.import_tflite(model_content, import_options)

        # compile
        compiler.compile()

        self.tools.write_kmodel_file(output_filename, compiler)
