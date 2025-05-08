class TritonBaseClient:
    """
        model request triton-inference-server with gRPC
    """
    def __init__(self,
                triton_host="localhost:port",
                trion_model_name="model_name",
                connection="GRPC", 
                verbose=False, 
                ssl=False, 
                root_certificates=None, 
                private_key=None, 
                certificate_chain=None):
        
        assert connection in ['GRPC', 'HTTP'], "Current support only connection type GRPC or HTTP"
        print('Init connection from Triton-inference-server')
        print('- Host: {}'.format(triton_host))
        print('- Connection: {}'.format(connection))
        self.triton_host = triton_host
        self.model_name = trion_model_name
        self.connection = connection
        if self.connection == 'GRPC':
            import tritonclient.grpc as grpcclient
            self.model = grpcclient.InferenceServerClient(url = self.triton_host,
                                                verbose=verbose,
                                                ssl=ssl,
                                                root_certificates=root_certificates,
                                                private_key=private_key,
                                                certificate_chain=certificate_chain)
        else:
            import tritonclient.http as httpclient
            self.model = httpclient.InferenceServerClient(url = self.triton_host)
        self.host_is_live()

    def host_is_live(self):
        # check host is connected or disconnected
        if not self.model.is_server_live():
            raise ValueError

    @staticmethod
    def mapping_label(classes_name_file):
        # mapping int -> values
        MAPPING_LABEL = {}
        try:
            with open(classes_name_file, "r") as f:
                str_car_brand = f.read()
                if ";" in str_car_brand:
                    list_car_brand = str_car_brand.split(";")
                elif "," in str_car_brand:
                    list_car_brand = str_car_brand.split(",")
                else:
                    list_car_brand = str_car_brand.split("\n")
                list_car_brand = [brand.strip() for brand in list_car_brand if brand.strip()]
                for idx, car_brand in enumerate(list_car_brand):
                    MAPPING_LABEL[idx] = car_brand

        except Exception as err:
            MAPPING_LABEL = {}
        return MAPPING_LABEL
    
    def preprocess_image(self):
        """
            Preprocess image
            Input: List image RGB
            Output: List image RGB normalization
        """
        raise NotImplementedError("The 'preprocess' method must be implemented by subclasses.")

    def postprocess_image(self):
        """
            Postprocessing image
        """
        raise NotImplementedError("The 'postprocess' method must be implemented by subclasses.")

    def inference(self):
        """
            Inference image
        """
        raise NotImplementedError("The 'inference' method must be implemented by subclasses.")
