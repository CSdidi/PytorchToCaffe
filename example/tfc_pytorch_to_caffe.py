import sys
sys.path.insert(0,'.')
import torch
from torch.autograd import Variable
from net import ClsNet
import pytorch_to_caffe

if __name__=='__main__':
    name='tfc'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = ClsNet()
    model.load_state_dict(torch.load("/data/projects/tfc-logic/data/cls_epoch_1000.pt",map_location=torch.device(device)))
    model.eval()
    input=Variable(torch.ones([1,27]))
    pytorch_to_caffe.trans_net(model,input,name)
    pytorch_to_caffe.save_prototxt('{}.prototxt'.format(name))
    pytorch_to_caffe.save_caffemodel('{}.caffemodel'.format(name))