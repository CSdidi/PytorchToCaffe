import sys
sys.path.insert(0,'.')
import torch
from torch.autograd import Variable
from net import ClsNet
import argparse
import os
import pytorch_to_caffe

parser = argparse.ArgumentParser(description='args for model transformation')
parser.add_argument('--input_file', '-i', help='input pytorch model', required=True)
parser.add_argument('--output_dir', '-o', help='output dir', required=True)
parser.add_argument('--num_neuron', '-n', help='number of neurons', required=True)
args = parser.parse_args()

if __name__=='__main__':
    '''
    process input params
    '''
    print('\ninput_params\n')
    print('input_file:{}'.format(args.input_file))
    print('output_dir:{}'.format(args.output_dir))
    print('num_neuron:{}'.format(args.num_neuron))
    input_file = args.input_file
    output_dir = args.output_dir
    num_neuron = int(args.num_neuron)
    if num_neuron <= 0:
        print('invalid num_neuron')
        exit(1)
    if not os.path.exists(input_file):
        print('input file does not exist: {}'.format(input_file))
        exit(1)
    print('===================\n')

    '''
    load model
    '''
    name='tfc'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = ClsNet(num_neuron)
    model.load_state_dict(torch.load(input_file,map_location=torch.device(device)))
    model.eval()

    '''
    transfer pt model to caffemodel
    '''
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    input=Variable(torch.ones([1,30]))
    pytorch_to_caffe.trans_net(model,input,name)
    prototxt_file = '{}/{}_{}.prototxt'.format(output_dir,name,num_neuron)
    pytorch_to_caffe.save_prototxt(prototxt_file)
    pytorch_to_caffe.save_caffemodel('{}/{}_{}.caffemodel'.format(output_dir,name,num_neuron))

    '''
    process the generated prototxt
    '''
    f1 = open(prototxt_file,"r")
    content = f1.readlines()
    f1.close()

    for i in range(len(content)):
        if i in range(16):
            content[i] = ''
        else:
            content[i] = content[i].replace("view_blob1","Data1")

    content.insert(0, 'layer {\nname: \"Data1\"\ntype: "Input"\ntop: \"Data1\"\ninput_param {\nshape {\ndim: 1\ndim: 30\ndim: 1\ndim: 1\n}\n}\n}\n')
    content.insert(len(content),'layer {\nname: \"Softmax1\"\ntype: \"Softmax\"\nbottom: \"fc_blob3\"\ntop: \"Softmax1\"\n}\n')

    with open(prototxt_file,"w") as f2:
        f2.writelines(content)