import numpy as np
import sys
import caffe
import argparse
import os

parser = argparse.ArgumentParser(description='args for model transformation')
parser.add_argument('--prototxt', '-p', help='prototxt file', required=True)
parser.add_argument('--weight', '-w', help='weight file', required=True)
parser.add_argument('--test_file', '-t', help='test file', required=True)
args = parser.parse_args()

if __name__=='__main__':
    # Set the right path to your model definition file, pretrained model weights,
    # and the image you would like to classify.
    prototxt = args.prototxt
    weight = args.weight
    test_f = args.test_file

    # load the model
    caffe.set_mode_gpu()
    caffe.set_device(0)
    net = caffe.Net(prototxt, weight, caffe.TEST)
    print("successfully loaded model")

    # load test data
    test_data = np.load(test_f)
    print("----type----")
    print(type(test_data))
    print("----shape----")
    print(test_data.shape)

    # eval
    correct = 0
    rows = test_data.shape[0]
    cols = test_data.shape[1]
    for i in range(0, rows):
        line = test_data[i]
        gt = line[cols-2].astype(np.int)
        md5 = line[cols-1]
        input = np.delete(line, [cols-2,cols-1]).astype(np.int)
        input = input.reshape(1,cols-2,1,1)
        net.blobs['Data1'].data[...] = input
        out = net.forward()
        target = out['Softmax1'].argmax()
        if target != gt:
            print('idx:{},data:{},gt:{},pred:{},prob:{},md5:{}'.format(i,np.delete(line, [cols-2,cols-1]).astype(np.int),gt,target,out['Softmax1'][0][target],md5))
        else:
            correct += 1

    # print result
    print('\nTest set: Accuracy: {}/{} ({:.4f}%)\n'.format(
        correct, rows,
        100. * correct / rows))