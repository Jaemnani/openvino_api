from openvino.runtime import Core
from abc import ABC, abstractmethod

class OPENVINO_MODEL:
    def __init__(self, model_xml, device="CPU"):
        self.model_xml = model_xml
        self.model_bin = model_xml.replace(".xml", ".bin")
        self.device = device
        
        core = Core()
        model = core.read_model(model=self.model_xml, weights=self.model_bin)
        self.compiled_model = core.compile_model(model=model, device_name=device)
        
        self.infer_request = self.compiled_model.create_infer_request()
        # self.input_name = self.compiled_model.input().any_name

        self.last_output = None
        self.input_image = None
        
        self.input_count = len(self.compiled_model.inputs)
        self.output_count = len(self.compiled_model.outputs)
        
        self.input_shape = [ list(inp.shape) for inp in self.compiled_model.inputs ]
        self.output_shape = [ list(out.shape) for out in self.compiled_model.outputs ]
        
        self.input_name = [ inp.any_name for inp in self.compiled_model.inputs ]
        self.output_name = [ out.any_name for out in self.compiled_model.outputs ]
        
        self.input_type = [ inp.element_type.to_dtype().name for inp in self.compiled_model.inputs ]
        self.output_type = [ out.element_type.to_dtype().name for out in self.compiled_model.outputs ]
    
    def __call__(self, input_image):
        results = self.inference(input_image)
        return results
        
    @abstractmethod
    def inference(self, input_image):
        pass

    @abstractmethod
    def preprocessing(self, input_image):
        
        pass

    @abstractmethod
    def postprocessing(self, output):
        pass
