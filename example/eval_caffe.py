import numpy as np
import sys
import caffe

# Set the right path to your model definition file, pretrained model weights,
# and the image you would like to classify.
prototxt = 'models/tfc_deploy.prototxt'
model = 'models/tfc.caffemodel'
test_f = 'data/test_6k.npy'

# load the model
caffe.set_mode_gpu()
caffe.set_device(0)
net = caffe.Net(prototxt, model,caffe.TEST)
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
    pred = line[cols-1]
    input = np.delete(line, -1)
    input = input.reshape(1,cols-1,1,1)
    net.blobs['Data1'].data[...] = input
    out = net.forward()
    target = out['Softmax1'].argmax()
    if target != pred:
        print('idx:', i, 'data:', line, ' target:{}'.format(target), out['Softmax1'][0][target], ' pred:', pred)
    else:
        correct += 1

# print result
print('\nTest set: Accuracy: {}/{} ({:.4f}%)\n'.format(
    correct, rows,
    100. * correct / rows))