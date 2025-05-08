import cv2
import os
import json
import requests
import numpy as np
import tritonclient.grpc as grpcclient
from PIL import Image
from io import BytesIO
from triton_base import TritonBaseClient

class TritonCarBrandClient(TritonBaseClient):

    def __init__(self,
                triton_host="localhost:port",
                triton_model_name="model_name",
                connection="GRPC",
                preprocess_input_size=(256, 256),
                max_batch_size=8,
                classes_name_file="",
                **kwargs):
        
        super().__init__(triton_host, triton_model_name, connection)
        self.triton_model_name = triton_model_name
        self.preprocess_input_size = preprocess_input_size
        self.max_batch_size = max_batch_size
        self.kwargs = kwargs
        self.labels = self.mapping_label(classes_name_file)
        self.std = kwargs.get("std", None)
        self.mean = kwargs.get("mean", None)
        

    def preprocess_image(self, images):
        """
            Preprocess image car brand
        """
        batch_images = []
        for i, image in enumerate(images):
            if isinstance(image, str):
                if os.path.exists(image):
                    image = cv2.imread(image)
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                elif image.startswith("http"):
                    response = requests.get(image)
                    if response.status_code != 200:
                        continue
                    else:
                        image_data = BytesIO(response.content)
                        image = Image.open(image_data)
                        image = np.array(image)
            elif isinstance(image, np.ndarray):
                image = image
            else:
                raise ValueError
            image = cv2.resize(image, self.preprocess_input_size)
            image = image.astype(np.float32)
            if self.mean is not None:
                image -= self.mean
            if self.std is not None:
                image /= self.std
            image /= 255.0
            batch_images.append(image)
        batch_images = np.array(batch_images, dtype=np.float32)
        batch_images = batch_images.transpose((0, 3, 1, 2))
        return batch_images
    
    def postprocess_image(self, output):
        """
            Postprocess image (use output from model to process continue)
        """
        idx_output = list(np.argmax(output, axis=1)) 
        confs = list(np.max(output, axis=1))
        confs = list(map(str, confs))
        labels = [self.labels.get(item, "Unknown") for item in idx_output]
        return labels, confs 
    
    def inference(self, 
            images,
            meta_inputs = [('input', 'FP32')],
            meta_outputs = [('output', 'FP32')]):
        """
            Predict batch image
        """
        total_images = len(images)
        total_batch = int(total_images/self.max_batch_size) if total_images % self.max_batch_size == 0 else int(total_images/self.max_batch_size) + 1
        predict_labels, predict_confs = [], []
        for iter_batch in range(total_batch):
            inputs = []
            outputs = []
            lower = iter_batch * self.max_batch_size
            higher = min((iter_batch + 1) * self.max_batch_size, total_images)
            batch_preprocess = self.preprocess_image(images[lower:higher])
            if self.connection == "GRPC":
                for ix, input_tuple in enumerate(meta_inputs):
                    inputs.append(grpcclient.InferInput(input_tuple[0], batch_preprocess.shape, input_tuple[1])) # <name, shape, dtype>
                    inputs[ix].set_data_from_numpy(batch_preprocess)
                for ix, output_tuple in enumerate(meta_outputs):
                    outputs.append(grpcclient.InferRequestedOutput(output_tuple[0]))
            results = self.model.infer(
                model_name=self.triton_model_name,
                inputs=inputs,
                outputs=outputs,
                client_timeout=None
            )

            results = results.as_numpy(output_tuple[0])
            labels, confs = self.postprocess_image(results)
            predict_labels += labels
            predict_confs += confs
        results_final = {key: (val1, val2) for key, val1, val2 in zip(images, predict_labels, predict_confs)}            
        return results_final
    

if __name__ == '__main__':
    path_folder = "/home1/data/hoangle/deepstream-hawkice-v2/test"
    images = []
    for path_image in os.listdir(path_folder):
        images.append(os.path.join(path_folder, path_image))
    model = TritonCarBrandClient(
        triton_host = 'localhost:1000',
        triton_model_name = "car_brand_convnext_v9",
        classes_name_file = "/home1/data/hoangle/hawkice_weights/triton/car_brand_convnext_v9/1/classes.txt"
    )
    netouts = model.inference(images)
    print(netouts)