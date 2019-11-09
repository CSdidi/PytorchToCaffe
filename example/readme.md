## How to run

1. Converting a pytorch model to caffe model  
```
cd PytorchToCaffe
python example/tfc_pytorch_to_caffe.py -i pt_path -o caffe_model_dir -n num_neuron
```
2. evaluate the converted caffe model
```
python eval_caffe.py -p prototxt_file_path -w weight_file_path -t test_file_path
```