class NNCaseTools:
    def write_model_file(self, model_file: str, model_content):
        with open(model_file, 'wb') as f:
            f.write(model_content)
            
    def read_model_file(self, model_file: str):
        with open(model_file, 'rb') as f:
            model_content = f.read()
        return model_content
    
    def write_kmodel_file(self, output_filename: str, compiler):
        kmodel = compiler.gencode_tobytes()
        with open(output_filename, 'wb') as f:
            print('writing kmodel...')
            f.write(kmodel)
