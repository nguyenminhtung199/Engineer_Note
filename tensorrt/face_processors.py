import os
import sys
import cv2
import numpy as np
# import torch
import time
# import backbones
import sklearn.preprocessing
# from exec_backends.trt_backend import alloc_buf, inference
import onnxruntime

def flip_batch(batch_data):
    assert len(batch_data.shape) == 4
    assert batch_data.shape[-1] == 3
    return np.flip(batch_data, 2)

class FaceEmbedding(object):
    def __init__(self, model_path, gpu_id = 0, network = 'iresnet124', data_shape = (3, 112, 112)):
        image_size = (112, 112)
        self.image_size = image_size
        self.device = torch.device('cuda:{}'.format(gpu_id))
        weight = torch.load(model_path)
        self.model = eval("backbones.{}".format(network))(False)
        self.model.load_state_dict(weight)
        self.model.to(self.device)
        self.model.eval()

    def check_assertion(self, feats, eps = 1e-6):
        '''
            Make sure that face embedding model (or code) work normally
        '''
        for ix, feat in enumerate(feats):
            assert np.fabs(np.linalg.norm(feat) - 1.0) < eps, print(ix, np.linalg.norm(feat))


    def get_features(self, inp, batch_size = 32, feat_dim = 512, flip = 0):
        '''
            Input: List of RGB image BxHxWx3
        '''
        batch_data = inp.copy()
        if flip:
            batch_data = flip_batch(batch_data)
        # Divide to batch
        n_img = len(batch_data)
        feats_result = np.zeros((n_img, feat_dim))
        tot_batch = n_img//batch_size if n_img % batch_size == 0 else n_img//batch_size + 1
        # print('Number of batch: {}'.format(tot_batch))
        # Preprocess
        with torch.no_grad():
            aligned = np.transpose(np.array(batch_data), (0, 3, 1, 2))
            imgs = torch.Tensor(aligned).to(self.device)
            imgs.div_(255).sub_(0.5).div_(0.5)
            # Foreach batch
            for i in range(tot_batch):
                lower  = i*batch_size
                higher = min((i+1)*batch_size, n_img)
                feats_result[lower: higher] = self.model(imgs[lower: higher]).detach().cpu().numpy()
        norm  = np.linalg.norm(feats_result, axis = 1, keepdims = True)
        feats_result = np.divide(feats_result, norm)
        # self.check_assertion(feats_result)
        return feats_result

class FaceEmbeddingBatchTRT(object):
    def __init__(self, input_shape = (112, 112), batch_size = 8, trt_model_path = 'weights/resnet124-batchsize_8-fp32.trt', engine = 'TRT', triton_input_names = None,
                                            triton_output_names = None,
                                            triton_model_name = None):
        print('[INFO] Create FaceEmbedding model with {} engine'.format(engine))
        self.engine = engine

        self.input_shape = input_shape
        self.batch_size = batch_size
        if self.engine == 'TRT':
            import tensorrt as trt
            from exec_backends.trt_loader import TrtModel
            self.model = TrtModel(trt_model_path)
        elif self.engine == 'ONNX':
            self.model = onnxruntime.InferenceSession(trt_model_path, providers=['CPUExecutionProvider'])
        elif self.engine == 'TRITON':
            import tritonclient.grpc.aio
            self.model = tritonclient.grpc.InferenceServerClient(trt_model_path)
            self.triton_input_names = triton_input_names
            self.triton_output_names = triton_output_names
            self.triton_model_name = triton_model_name
        else:
            raise NotImplementedError("Current support only TRT and ONNX engine")

    def get_features(self, inp, feat_dim = 512, flip = 0):
        batch_data = inp.copy()
        if flip:
            batch_data = flip_batch(batch_data)
        # Allocate
        n_img       = len(batch_data)
        n_batch = n_img//self.batch_size if n_img % self.batch_size == 0 else  n_img//self.batch_size + 1
        # paded_batch = np.zeros((self.batch_size * n_batch, 3, self.input_shape[1], self.input_shape[0]), dtype = np.float32)
        feats_batch = np.zeros((self.batch_size * n_batch, feat_dim), dtype = np.float32)
        # # Preprocess
        aligned             = np.transpose(np.array(batch_data).astype("float32"), (0, 3, 1, 2))
        aligned             = ((aligned / 255.0) - 0.5)/0.5
        # paded_batch[:n_img] = aligned
        # print('will infer for {} batches'.format(n_batch))
        for i in range(n_batch):
            lower = i*self.batch_size
            higher = min((i+1)*self.batch_size, n_img)
            if self.engine == 'TRT':
                feats_batch[lower:higher] = np.squeeze(self.model.run(aligned[lower:higher])[0])   
            if self.engine == 'TRITON':
                img_tensor = aligned[lower:higher]
                import tritonclient.grpc.aio
                from tritonclient.utils import np_to_triton_dtype
                inputs = [
                    tritonclient.grpc.InferInput(self.triton_input_names[0], img_tensor.shape, np_to_triton_dtype(np.float32))
                ]
                inputs[0].set_data_from_numpy(np.float32(img_tensor))
                outputs = [
                    tritonclient.grpc.InferRequestedOutput(triton_output_name) for triton_output_name in self.triton_output_names
                ]
                res = self.model.infer(model_name=self.triton_model_name, inputs=inputs, outputs=outputs)
                net_outs = [res.as_numpy(triton_output_name) for triton_output_name in self.triton_output_names]
                feats_batch[lower:higher] = np.squeeze(net_outs[0])
                # net_outs = np.array(net_outs)
            else:
                ort_inputs = {self.model.get_inputs()[0].name: aligned[lower:higher]}
                feats_batch[lower:higher] = np.squeeze(self.model.run(None, ort_inputs)[0]) 
               # First output
        # Normalize
        norm         = np.linalg.norm(feats_batch[: n_img], axis = 1, keepdims = True)
        feats_result = np.divide(feats_batch[: n_img], norm)
        return feats_result


if __name__ == '__main__':
    model_path = 'weights/backbone.pth'
    # trt_model_path = 'weights/resnet124-batchsize_8-imp.trt'
    trt_model_path = 'weights/resnet124-dynamic.trt'
    face_embedd_tor = FaceEmbedding(model_path, gpu_id = 0)
    face_embedd_trt = FaceEmbeddingBatchTRT(trt_model_path = trt_model_path, batch_size = 32)

    bgr = cv2.imread("test_images/crop.jpg")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (112, 112))
    im2 = cv2.imread("test_images/lumia.jpg")
    im2 = cv2.resize(im2, (112, 112))
    im3 = cv2.imread("test_images/Stallone.jpg")
    im3 = cv2.resize(im3, (112, 112))
    im4 = cv2.imread("test_images/TH.png")
    im4 = cv2.resize(im4, (112, 112))
    im5 = cv2.imread("test_images/TH1.jpg")
    im5 = cv2.resize(im5, (112, 112))
    tik = time.time()
    for i in range(1):
        inp = [rgb, bgr, im2, im3, im4, im5]*6
        torch_out = np.squeeze(face_embedd_tor.get_features(inp))
        trt_out = np.squeeze(face_embedd_trt.get_features(inp))
        # print(torch_out[:, : 20])
        # print(trt_out[:, : 20])
        for j in range(len(inp)):
            print(np.sqrt(np.sum((torch_out[j]-trt_out[j])**2)))
    print(time.time() - tik)