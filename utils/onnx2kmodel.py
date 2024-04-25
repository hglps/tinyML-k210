# Src: https://github.com/Watch-Later/nncase/blob/7413ea6a2748bad372f17079d6fd43a0ecd4794d/docs/USAGE_EN.md
# Compile float32 model for onnx
import os
import onnxsim
import onnx
import nncase
import argparse

def parse_model_input_output(model_file):
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

def onnx_simplify(model_file):
    onnx_model, inputs = parse_model_input_output(model_file)
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


def read_model_file(model_file):
    with open(model_file, 'rb') as f:
        model_content = f.read()
    return model_content


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--onnx', type=str, required=True)
    args = parser.parse_args()

    model_file = args.onnx
    
    target = 'k210'

    # onnx simplify
    model_file = onnx_simplify(model_file)

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
    compile_options.input_shape = [1, 3, 640, 640]
    compile_options.quant_type = 'uint8'  # uint8 or int8

    # compiler
    print('compiler')
    compiler = nncase.Compiler(compile_options)

    # import_options
    print('import_options')
    import_options = nncase.ImportOptions()

    # import
    print('import onnx')
    model_content = read_model_file(model_file)
    compiler.import_onnx(model_content, import_options)

    # compile
    compiler.compile()

    # kmodel
    kmodel = compiler.gencode_tobytes()
    with open('test.kmodel', 'wb') as f:
        print('writing kmodel')
        f.write(kmodel)

if __name__ == '__main__':
    main()
