"""Target Model library

Library for extracting probabilities, activation maps and gradients via Target model.
Target models include VGG19, ResNet50 and InceptionV3 trained on ImageNet.
Editing here allows you to set up your desired model and dataset.

Dataset: Custom dataset class.

TargetModel: Extract information from a set of images via target model.
  register_forward: Regist forwarding hook.
  register_backward: Regist backwarding hook.
  model_create: Create a target model and save layers.
  run: Runs class function.
"""
from torchvision import transforms,models
import torch
from torch.autograd import Variable
from torch.utils.data import Dataset
import numpy as np
from PIL import Image

class Dataset(Dataset):
    """Custom Dataset
    
    This class creates torch dataset from a numpy type image set.
    """
    __slots__ = ('data',
                 'transforms')
    def __init__(self, 
                 data, 
                 transforms):
        """Define custom dataset class
        
        This allows you to use any dataset you want.
De      Dpending on user dataset, you may need to modify this.
    
        Args:
          data: Set of images.
          transforms: Transforms for preprocessing (Normalize, ToTensor, ...).
        """
        self.data = data
        self.transform = transforms
        
    def __getitem__(self, index):
        x = self.data[index]
        if self.transform:
            x = Image.fromarray(self.data[index])
            x = self.transform(x)
        return x
    
    def __len__(self):
        return len(self.data)

class TargetModel:
    """Extracting feature from a set of images
    
    Information from a set of images can be extracted by defining and 
    creating a target model and registering a specific layer.
    """
    __slots__ = ('images', 
                 'arg_model', 
                 'layer_num', 
                 'class_name',
                 'preprocess',
                 'categories')
    def __init__(self, 
                 images, 
                 arg_model, 
                 layer_num, 
                 class_name):
        """Define the target model class
        
        Make a preprocess and label that dataset.
        If you are using a custom data set, change "self.categories".
    
        Args:
          images: Set of images.
          arg_model: Target model name.
          layer_num: Specific layer number.
          class_name: Class name (kit_fox).
        """
        self.images = images
        self.arg_model = arg_model
        self.layer_num = layer_num
        self.class_name = class_name.replace('_',' ')

        self.preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        with open("imagenet_classes.txt", "r") as f:
            self.categories = np.array([s.strip() for s in f.readlines()])
        
    def register_forward(self, hook, childs, layer_num):
        """Registing specific layers for extracing activation maps
        
        Regist a hook function to extract activation maps 
        resulting from forwarding in a specific layer.
        
        Args:
          hook: Work function for layer.
          childs: List of layers.
          layer_num: Specific layer number.
        """
        childs[layer_num].register_forward_hook(hook)
        
    def register_backward(self, hook, childs, layer_num):
        """Registing specific layers for extracing gradients
        
        Regist a hook function to extract activation maps 
        resulting from backwarding in a specific layer.
    
        Args:
          hook: Work function for layer.
          childs: List of layers.
          layer_num: Specific layer number.
        """
        childs[layer_num].register_backward_hook(hook)
        
    def model_create(self, arg_model):
        """Creating pre-trained model 
        
        Create a target model and save 
        the conv and bottleneck layers of the model.
        
        Args:
          arg_model: Target model name.
        
        Returns:
          model: Target model object.
          childs: List of layers.
        """
        if arg_model == "vgg19":
            model = models.vgg19(pretrained=True).cuda()
        elif arg_model == "resnet50":
            model = models.resnet50(pretrained=True).cuda()
        elif arg_model == "inception_v3":
            model = models.inception_v3(pretrained=True).cuda()
        else:
            print('no model name')
            return -1
        model.eval();
        childs = []
        for module in model.children():
            if arg_model == "vgg19":
                try:
                    for child in module:
                        if isinstance(child, torch.nn.modules.conv.Conv2d):
                            childs.append(child)
                except:
                    continue
            elif arg_model == "resnet50":
                try:
                    for child in module:
                        childs.append(child)
                except:
                    continue
            else:
                if isinstance(module, models.inception.BasicConv2d) or \
                   isinstance(module, models.inception.InceptionA) or \
                   isinstance(module, models.inception.InceptionB) or \
                   isinstance(module, models.inception.InceptionC) or \
                   isinstance(module, models.inception.InceptionD) or \
                   isinstance(module, models.inception.InceptionE):
                    childs.append(module)
        return model, childs
                
    def run(self):
        """Runs class function
        
        Creating models and datasets.
        Extracting information by registering hooks in specific layers.
          
        Returns:
          List of probabilities, activation maps and gradients of images.
        """
        # create the target model trained some dataset.
        model, childs = self.model_create(self.arg_model)
        
        # create custom dataset.
        input_dataset = torch.utils.data.DataLoader(
            Dataset(self.images, self.preprocess),
            batch_size=32, shuffle=False, pin_memory=True)
        
        gradients = []
        # Define backward hook.
        def backward_hook(module, grad_input, grad_output):
            results = torch.squeeze(grad_output[0]).cpu().data.numpy()
            if len(results.shape)  < 4:
                results = np.expand_dims(results, axis=0)
            gradients.append(results)
        self.register_backward(backward_hook, childs, self.layer_num)

        activations = []
        # Define forward hook.
        def forward_hook(module, input, output):
            results = output.cpu().data.numpy()
            if len(results.shape)  < 4:
                results = np.expand_dims(results, axis=0)
            activations.append(results)
        self.register_forward(forward_hook, childs, self.layer_num)
        
        max_iter = 20
        prob = []
        # Runs model.
        for idx, input in enumerate(input_dataset):
            input = Variable(input).cuda()
            predict = (model(input))
            probabilities = torch.nn.functional.softmax(predict, dim=1)
            top_prob, top_catid = torch.topk(probabilities, max_iter)
            predict.mean().backward(retain_graph=True)
            for i in range(top_prob.size(0)):
                for j in range(max_iter):
                    # Get the probabilities.
                    if self.class_name in self.categories[top_catid[i][j]]:
                        prob.append(top_prob[i][j].item())
                        break
                    elif j == (max_iter -1):
                        prob.append(1e-5)
        result = [prob,
                  activations,
                  gradients,
                  ]
        del model
        del childs
        torch.cuda.empty_cache()
        return result
