CUDA_VISIBLE_DEVICES=5 tritonserver --model-repository=/workspace/models/ --model-control-mode=explicit --load-model=car_brands_accuracy

Cách đổi tên tritonserver để hiển thị người dùng khi dùng nvidia-smi2 
Sau khi khởi chạy docker triton, chạy dòng lệnh
ln -s /opt/tritonserver/bin/tritonserver /opt/tritonserver/bin/tritonserver_new
Sau đó thay vì dùng
tritonserver --models ...
Thì có thể dùng
tritonserver_new --models ...