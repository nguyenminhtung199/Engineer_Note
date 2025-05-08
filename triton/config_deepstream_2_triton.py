import configparser
import os
import shutil
import glob

def convert_docker_path_to_host(docker_path, mount_source, mount_target):
    if docker_path.startswith(mount_target):
        return docker_path.replace(mount_target, mount_source, 1)
    return docker_path  # Nếu không thuộc mount target thì giữ nguyên

def parse_deepstream_config(ds_config_path, mount_source, mount_target, model_name=None):
    """
    Đọc file config của DeepStream và trích xuất các thông số cần thiết.
    """
    config = configparser.ConfigParser(strict=False)
    config.read(ds_config_path)
    # Lấy giá trị cần thiết từ file config DeepStream
    ds_params = config["property"]
    print(ds_params)
    if model_name is None:
      model_name = ds_params.get("model-engine-file", "").split("/")[-1].split(".")[0]
    input_dims = ds_params.get("input-dims", "3;224;224;0").split(";")[:3]  # Chỉ lấy 3 giá trị đầu
    default_model_filename = ds_params.get("model-engine-file", "").split("/")[-1]
    model_dir = convert_docker_path_to_host(ds_params.get("model-engine-file", ""), mount_source, mount_target)
    if not os.path.exists(model_dir):
      model_dir = convert_docker_path_to_host(model_dir, mount_source, '..')
    label_file_path = ds_params.get("labelfile-path", "")
    if label_file_path:
      label_file_path = convert_docker_path_to_host(label_file_path, mount_source, mount_target)
      if not os.path.exists(label_file_path):
        label_file_path = convert_docker_path_to_host(label_file_path, mount_source, '..')
      with open(label_file_path, "r") as f:
        content = f.read().strip()  # Đọc toàn bộ nội dung và loại bỏ khoảng trắng dư thừa
        if ";" in content:
            labels = content.split(";")  # Nếu có dấu ;
        else:
            labels = content.splitlines()  # Nếu xuống dòng
      num_classes = len([label.strip() for label in labels if label.strip()])
    else:
      num_classes = 1000
    input_dims = list(map(int, input_dims))
    batch_size = int(ds_params.get("batch-size", 1))
    network_mode = ds_params.get("network-mode", "2")  # FP16 nếu là 2
    
    # Chuyển đổi network-mode sang Triton format
    precision_mode = "fp32"
    if network_mode == "1":
        precision_mode = "int8"
    elif network_mode == "2":
        precision_mode = "fp16"

    return {
        "model_name": model_name,
        "default_model_filename": default_model_filename,
        "input_dims": input_dims,
        "batch_size": batch_size,
        "precision_mode": precision_mode,
        "num_classes": num_classes,
        "model_dir": model_dir,
        "label_file_path": label_file_path
    }

def generate_triton_config(save_path, ds_params):
    path_model = os.path.join(save_path, ds_params['model_name'])
    os.makedirs(os.path.dirname(path_model), exist_ok=True)
    triton_config_path = os.path.join(path_model, "config.pbtxt")
    model_save_path = os.path.join(path_model, "1")
    os.makedirs(model_save_path, exist_ok=True)
    shutil.copy(ds_params['model_dir'], model_save_path)
    if ds_params['label_file_path']:
        shutil.copy(ds_params['label_file_path'], model_save_path)
    config_pbtxt = f"""
name: "{ds_params['model_name']}"
platform: "tensorrt_plan"
max_batch_size: {ds_params['batch_size']}
default_model_filename: "{ds_params['default_model_filename']}"

input [
  {{
    name: "input"
    data_type: TYPE_{ds_params['precision_mode'].upper()}
    format: FORMAT_NCHW
    dims: {ds_params['input_dims']}
  }}
]

output [
  {{
    name: "output"
    data_type: TYPE_{ds_params['precision_mode'].upper()}
    dims: [{ds_params['num_classes']}]
  }}
]

instance_group [
  {{
    kind: KIND_GPU
    count: 1
    gpus: [0]
  }}
]


"""

    with open(triton_config_path, "w") as f:
        f.write(config_pbtxt.strip())

    print(f"File Triton config đã được tạo: {triton_config_path}")


deepstream_config_path = "config_car_brand_convnext_v9.txt" 
save_path = "."
mount_source = "/home1/data/tungcao/hawkice/deepstream-hawkice-v2"
mount_target = "/deepstream"

# #Add argparse
# import argparse
# parser = argparse.ArgumentParser(description='Convert Deepstream config to Triton config')
# parser.add_argument('--deepstream_config_path', type=str, help='Path to Deepstream config file')
# parser.add_argument('--save_path', type=str, help='Path to save Triton config file')
# parser.add_argument('--mount_source', type=str, help='Mount source')
# parser.add_argument('--mount_target', type=str, help='Mount target')
# args = parser.parse_args()

# deepstream_config_path = args.deepstream_config_path
# save_path = args.save_path
# mount_source = args.mount_source
# mount_target = args.mount_target
config_paths = [
    "configs_new_acc/config_v010_lpd_retinalp.txt",
    "configs_new_acc/config_lpr_yolov8.txt",
    "configs_new_acc/config_v010_lpr_yolov8_res.txt",
    "configs_new_acc/config_car_brand_convnext_v9.txt",
    "configs_new_acc/config_car_brand_resnet18_keep_padding.txt",
    "configs_new_acc/config_face_embedding_unconstrained_r50.txt",
    "configs_new_acc/config_face_attribute_mask.txt",
    "configs_new_acc/config_person_attribute_local_accuracy.txt",
    "configs_new_acc/config_person_attribute_local_performance.txt",
    "configs_new_acc/config_color_mobilenetv3_large_add_motor.txt",
    "configs_new_acc/config_motorbike_brand_convnext_v10_balance.txt",
    "configs_new_acc/config_motorbike_brand_resnet18_performance.txt"
]
model_names = [
    "lpd",
    "lpr",
    "lpr_restoration",
    "car_brands_accuracy",
    "car_brands_performance",
    "face_embedding_unconstrained",
    "face_attribute_mask",
    "person_attribute_accuracy",
    "person_attribute_performance",
    "vehicle_colors",
    "moto_brands_accuracy",
    "moto_brands_performance"
]
for deepstream_config in config_paths:
  try: 
    deepstream_config_path = os.path.join(mount_source, deepstream_config)
    ds_params = parse_deepstream_config(deepstream_config_path, "/home1/data/tungcao/hawkice/deepstream-hawkice-v2", "/deepstream", model_name=model_names[config_paths.index(deepstream_config)])
    generate_triton_config(save_path, ds_params)
  except Exception as e:
    raise e
    print(f"Error: {e}")
    print(f"Deepstream config: {deepstream_config_path}")
    continue
