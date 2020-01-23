import sys, os, json
import numpy as np
import argparse

import torch
import torch.nn.functional as F
import pretrainedmodels
import pretrainedmodels.utils as utils
os.environ["CUDA_VISIBLE_DEVICES"]="0"

# can we use CUDA?
cuda = torch.cuda.is_available()  # False
print ("=> using cuda: {cuda}".format(cuda=cuda))

def argument_func():
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Testing')
    parser.add_argument('--img_folder', type = str, help = 'Folder path to image')
    return parser

def testing(in_model, folder):
    in_model.eval()
    load_img = utils.LoadImage()
    tf_img = utils.TransformImage(in_model)

    # load a json file for Imagenet to get actual labels
    with open('/home/li-yun/pytorch/imagenet/imagenet_class_index.json', encoding='utf-8') as data_file:
        imagenet_labels = json.loads(data_file.read())
    
    # read all images for a given folder
    for f in sorted(os.listdir(folder)):
        if os.path.isfile(os.path.join(folder, f)):
            print('Images: ' + f)
            input_img = load_img(os.path.join(folder, f))
            input_tensor = tf_img(input_img)
            input_tensor = input_tensor.unsqueeze(0)
            if cuda:
                input_tensor = input_tensor.cuda(async=True)
            x = torch.autograd.Variable(input_tensor, requires_grad=False)
            output_logits = in_model(x) 
            # get top-5 classes
            pred_vals, pred_index = output_logits.topk(5, 1, True, True)
            # dim = 1 for summing across columns.
            prob_values = F.softmax(pred_vals, dim = 1).data.cpu().numpy()[0]
            labs_ind = pred_index.cpu().numpy()[0]
            prob_index = 0
            for label_ind in labs_ind:
                print(imagenet_labels[str(label_ind)][1],'{:.10f}'.format(prob_values[prob_index]))
                prob_index += 1
            print('==============')


# main function
def main():
    argu_obj = argument_func()
    args = argu_obj.parse_args()
    img_folder_path = args.img_folder

    # load a pre-trained model
    model = pretrainedmodels.__dict__['inceptionv4'](num_classes=1000, pretrained='imagenet')
    if cuda:
        model.cuda()
    else:
        model.cpu()
    # testing
    testing(model, img_folder_path)
    
if __name__ == '__main__':
    main()
