import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
from PIL import Image
import cv2
import tensorrt as trt  # Bỏ torch.cuda.nvtx vì không cần nữa (8.5.1.7)
import time
from multiprocessing import Pool
from multiprocessing.dummy import Pool as ThreadPool
import requests

TRT_LOGGER = trt.Logger()

def preprocess_pil(img_path):
    img = Image.open(img_path)
    pil_img_resized = img.resize((224, 224), Image.BILINEAR)
    img_np = np.array(pil_img_resized).astype(np.float32) / 255.0
    if img_np.ndim == 2:
        img_np = np.stack([img_np] * 3, axis=-1)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_np = (img_np - mean) / std
    img_np = np.transpose(img_np, (2, 0, 1))
    return img_np

def get_bgr_image_from_url(image_url):
    r = requests.get(image_url, timeout=10)
    bgr = cv2.imdecode(np.frombuffer(r.content, np.uint8), -1)
    if bgr is None: 
        print('Img url: ', image_url)

    if len(bgr.shape) == 2:
        bgr = cv2.cvtColor(bgr, cv2.COLOR_GRAY2BGR)
    elif len(bgr.shape) == 3:
        if bgr.shape[-1] == 3:
            pass
        elif bgr.shape[-1] == 4:
            bgr = cv2.cvtColor(bgr, cv2.COLOR_BGRA2BGR)
        else:
            raise NotImplementedError("")
    else:
        raise NotImplementedError("")  
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    
    # Chuyển đổi từ NumPy array sang PIL image
    pil = Image.fromarray(rgb)
    return pil

def preprocess(img_url):
    img = get_bgr_image_from_url(img_url)
    # Đổi kích thước ảnh về (224, 224)
    pil_img_resized = img.resize((224, 224), Image.BILINEAR)
    img_np = np.array(pil_img_resized).astype(np.float32) / 255.0
    if img_np.ndim == 2:
        img_np = np.stack([img_np] * 3, axis=-1)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_np = (img_np - mean) / std
    img_np = np.transpose(img_np, (2, 0, 1))
    return img_np

# def batch_image():

class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()

def allocate_buffers(engine):
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()
    out_shapes = []
    input_shapes = []
    out_names = []
    max_batch_size = engine.get_profile_shape(0, 0)[2][0]
    for binding in engine:
        binding_shape = engine.get_binding_shape(binding)
        if binding_shape[0] == -1:
            binding_shape = (1,) + binding_shape[1:]
        size = trt.volume(binding_shape) * max_batch_size
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        bindings.append(int(device_mem))
        if engine.binding_is_input(binding):
            inputs.append(HostDeviceMem(host_mem, device_mem))
            input_shapes.append(engine.get_binding_shape(binding))
        else:
            outputs.append(HostDeviceMem(host_mem, device_mem))
            out_shapes.append(engine.get_binding_shape(binding))
            out_names.append(binding)
    return inputs, outputs, bindings, stream, input_shapes, out_shapes, out_names, max_batch_size

def do_inference(context, bindings, inputs, outputs, stream):
    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
    context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
    stream.synchronize()
    return [out.host for out in outputs]

class TrtModel(object):
    def __init__(self, model):
        self.engine_file = model
        self.engine = None
        self.inputs = None
        self.outputs = None
        self.bindings = None
        self.stream = None
        self.context = None
        self.input_shapes = None
        self.out_shapes = None
        self.max_batch_size = 1
        self.optimization_profile_index = 0

    def build(self):
        with open(self.engine_file, 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.inputs, self.outputs, self.bindings, self.stream, self.input_shapes, self.out_shapes, self.out_names, self.max_batch_size = allocate_buffers(self.engine)
        self.context = self.engine.create_execution_context()
        self.context.set_optimization_profile_async(self.optimization_profile_index, self.stream.handle)
    @staticmethod    
    def safe_process_image(image):
        try:
            return preprocess(image)  # Gọi hàm xử lý ảnh của bạn
        except Exception as e:
            print(f"Lỗi khi xử lý ảnh {image}: {e}")
            return None  # Trả về None nếu gặp lỗi
    def run(self, images, count=0, deflatten: bool = True, as_dict=False):
        if self.engine is None:
            self.build()
        
        # if len(input) > 1:
        #     lst_processed = []
        #     for img_link in input:
        #         img_processed = preprocess(img_link)
        #         img_processed = np.expand_dims(img_processed, axis=0)
        #         lst_processed.append(img_processed)

        #     input = np.concatenate(lst_processed, axis=0)
        # else:
        #     img_processed = preprocess(input[0])
        #     input = np.expand_dims(img_processed, axis=0)
        tik0 = time.time()
        if len(images) > 1:
            # Khởi tạo mảng rỗng với kích thước phù hợp cho danh sách ảnh đầu vào
            sample_img = preprocess(images[0])
            tik1 = time.time()
            input_shape = (len(images),) + sample_img.shape
            input = np.empty(input_shape, dtype=sample_img.dtype)
            tik2 = time.time()
            

            with ThreadPool(1) as pool:
                processed_images = pool.map(self.safe_process_image, images)

            input[:] = processed_images
            tik3 = time.time()
        else:
            # Nếu chỉ có 1 ảnh, xử lý nó rồi thêm một chiều để giữ nguyên dạng
            img_processed = preprocess(images[0])
            input = np.expand_dims(img_processed, axis=0)
        print(input.shape)
        batch_size = input.shape[0]
        allocate_place = np.prod(input.shape)
        self.inputs[0].host[:allocate_place] = input.flatten(order='C').astype(np.float32)

        self.context.set_binding_shape(0, input.shape)

        tik4 = time.time()
        trt_outputs = do_inference(
            self.context, bindings=self.bindings,
            inputs=self.inputs, outputs=self.outputs, stream=self.stream)
        if len(images) > 1:
            print("=" * 40)
            print("{:<20} | {:>10} seconds".format("Stage", "Time"))
            print("-" * 40)
            print("{:<20} | {:>10.4f} seconds".format("Preprocess/images:", tik1 - tik0))
            print("{:<20} | {:>10.4f} seconds".format("Create vectors:", tik2 - tik1))
            print("{:<20} | {:>10.4f} seconds".format("Batch images:", tik3 - tik2))
            print("{:<20} | {:>10.4f} seconds".format("Allocate:", tik4 - tik3))
            print("{:<20} | {:>10.4f} seconds".format("Inference:", time.time() - tik4))
            print("=" * 40)

        if deflatten:
            trt_outputs = [output.reshape(shape) for output, shape in zip(trt_outputs, self.out_shapes)]
        if as_dict:
            return {name: trt_outputs[i] for i, name in enumerate(self.out_names)}

        return [trt_output[:batch_size] for trt_output in trt_outputs]

def main():
    model = TrtModel('model/dinov2_11value.trt')

    img_path = '1112355.jpg'
    result = model.run(img_path)

    print(result[0].shape)


if __name__ == '__main__':
    main()
