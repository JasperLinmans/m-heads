"""
Multi-head model based on Pytorch.
"""

from diagmodels.models.pytorch import pytorchmodelbase as dmspytorchmodelbase
from diagmodels.models.pytorch.densenet import Bottleneck, SingleLayer, Transition, make_dense_block
from diagmodels.errors import modelerrors as dmsmodelerrors

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import Counter
from collections import OrderedDict
from scipy.stats import entropy
from scipy.spatial.distance import jensenshannon
from numpy.linalg import norm

def JSD(P, Q):
    _P = P / norm(P, ord=1)
    _Q = Q / norm(Q, ord=1)
    _M = 0.5 * (_P + _Q)
    return 0.5 * (entropy(_P, _M) + entropy(_Q, _M))

#----------------------------------------------------------------------------------------------------
        
class DenseNetMHeads(nn.Module):
    
    def __init__(self, input_shape, layers_per_block, growth_rate, reduction, num_classes, bottleneck,
                 head_depth, num_heads, bottleneck_rate, valid_padding, dropout_rate=0.0, pool_initial=False, pool_final=True):
                 
        super(DenseNetMHeads, self).__init__()
        
        if bottleneck:
            assert head_depth % 2 == 0
        
        channels = 2 * growth_rate
        self.num_heads = num_heads
        padding = 0 if valid_padding else 1
        
        # Check block where heads start
        #
        dense_blocks = head_depth // 2 if bottleneck else head_depth     
        head_block_index = None 
        head_dense_blocks_remaining = None
        for block_index, layers in enumerate(layers_per_block[::-1]):
            dense_blocks -= layers
            if dense_blocks <= 0:
                head_block_index = len(layers_per_block) - 1 - block_index
                head_dense_blocks_remaining = abs(dense_blocks)
                break
        
        # First convolution.
        #
        if pool_initial:
            initial_padding = 0 if valid_padding else 3
            self.layers = nn.Sequential(OrderedDict([
                ('conv0', nn.Conv2d(input_shape[0], channels, kernel_size=7, stride=2, padding=initial_padding, bias=False)),
                ('bn0', nn.BatchNorm2d(channels)),
                ('act0', nn.ReLU()),
                ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=padding, ceil_mode=False))
            ]))
        else:
            self.layers = nn.Sequential(OrderedDict([
                ('conv0', nn.Conv2d(input_shape[0], channels, kernel_size=3, padding=padding, bias=False))
            ]))
        
        for block_index, n_blocks in enumerate(layers_per_block):
            
            if block_index == head_block_index:
                n_blocks = head_dense_blocks_remaining
            
                if not n_blocks:
                    break
            
                self.layers.add_module('dense{}'.format(block_index), make_dense_block(channels, growth_rate, n_blocks, bottleneck, dropout_rate, bottleneck_rate, padding))
                channels += n_blocks * growth_rate
                
                break
            
            self.layers.add_module('dense{}'.format(block_index), make_dense_block(channels, growth_rate, n_blocks, bottleneck, dropout_rate, bottleneck_rate, padding))      
            channels += n_blocks * growth_rate
            
            out_channels = int(math.floor(channels * reduction))
            self.layers.add_module('trans{}'.format(block_index), Transition(channels, out_channels, padding))
            channels = out_channels

        def head(channels):
            
            layers = nn.Sequential()
            
            for block_index, n_blocks in enumerate(layers_per_block[head_block_index:]):
                
                # Set amount of blocks based on the ones already defined in the previous part of the network.
                #
                if block_index == 0:
                    
                    n_blocks -= head_dense_blocks_remaining
                
                layers.add_module('head_dense{}'.format(block_index), make_dense_block(channels, growth_rate, n_blocks, bottleneck, dropout_rate, bottleneck_rate, padding))
                channels += n_blocks * growth_rate
                
                # Skip transition layer on the final layers.
                #
                if block_index == len(layers_per_block[head_block_index:]) - 1:
                    break

                # Define transition layers.
                #
                out_channels = int(math.floor(channels * reduction))
                layers.add_module('head_trans{}'.format(block_index), Transition(channels, out_channels, padding))
                channels = out_channels
            
            layers.add_module('bn', nn.BatchNorm2d(channels))
            layers.add_module('act', nn.ReLU())
            intermediate_out = self.layers(torch.rand(*input_shape).unsqueeze(0))
            intermediate_out = layers(intermediate_out)
            final_conv_size = intermediate_out.shape[-1]
            if pool_final:
                layers.add_module('pool', nn.AvgPool2d(kernel_size=final_conv_size, stride=1, padding=0))
                final_conv_size = 1
            layers.add_module('fconv', nn.Conv2d(channels, num_classes, kernel_size=final_conv_size, stride=1, padding=0, bias=True))

            return layers

        self.mheads = nn.ModuleList(
            [head(channels) for _ in range(self.num_heads)]
        )
    
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.reset_parameters()
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        out = self.layers(x)
        head_predictions = [self.mheads[i](out) for i in range(self.num_heads)]
        head_predictions = torch.stack(head_predictions, dim=-1)
        
        # Remove spatial dimensions when none are left.
        # [batch, channels, h, w, heads] -> [batch, channels, heads] if (h, w) == (1, 1)
        #
        return head_predictions.squeeze(-2).squeeze(-2)


class MHeads(dmspytorchmodelbase.PytorchModelBase):
    def __init__(self, name=None, description=None):
        """
        Initialize model parameters.

        Args:
            name (str): Name of the model.
            description (str): Description of the model.
        """

        # Initialize base class.
        #
        super().__init__(name, description)

        # Initialize members.
        #
        self.__backbone         = None     # Backbone used.
        self.__input_shape      = None     # Input shape: (channels, rows, cols).
        self.__num_classes      = None     # Number of output classes.
        self.__head_depth       = None     # Depth of the CNN backbone.
        self.__num_heads        = None     # Number of heads used by the multi-head model.
        self.__mhead_random     = None     # Factor used to randomly assign a winning head.
        self.__mhead_eps        = None     # Factor used to assign gradient weights.
        self.__learning_rate    = None     # Learning rate of the optimizer.
        self.__dropout_rate     = None     # dropout rate used during training.
        self.__layers_per_block = None     # Densenet backbone parameter.
        self.__growth_rate      = None     # Densenet backbone parameter.
        self.__bottleneck       = None     # Densenet backbone parameter.
        self.__reduction        = None     # Densenet backbone parameter.
        self.__bottleneck_rate  = None     # Densenet backbone parameter.
        self.__pool_initial     = None     # Densenet backbone parameter.
        self.__pool_final       = None     # Densenet backbone parameter.
        self.__valid_padding    = None     # Determines padding variant.
        self.__optimizer_type   = None     # String determining the optimizer used.
        self.__optimizer_mom    = None     # Determines optimizer momentum parameter.
        self.__optimizer_wd     = None     # Determines optimizer weight decay parameter.
        self.__multi_gpu        = None     # Determines the use of parallel training or not.
        self.head_report        = None     # Keep track of winning heads.

    def configure(self, input_shape, num_classes, backbone, optimizer_type, optimizer_mom, optimizer_wd, head_depth, 
                  num_heads, mhead_random, mhead_eps, learning_rate, dropout_rate, layers_per_block, growth_rate, bottleneck, 
                  reduction, pool_initial, pool_final, valid_padding, bottleneck_rate, multi_gpu):
        """
        Save the network configuration parameters.

        Args:
            input_shape (tuple): Number of channels and size of the individual input images. (rows, cols, channels)
            classes (int): Number of output classes.

        Raises:
            InvalidInputShapeError: The input shape is not valid.
            InvalidModelClassCountError: The number of classes is not valid.
        """

        # Check input parameters.
        #
        if len(input_shape) != 3 or min(input_shape) <= 0:
            raise dmsmodelerrors.InvalidInputShapeError(input_shape)

        if num_classes <= 1:
            raise dmsmodelerrors.InvalidModelClassCountError(num_classes)

        # Save parameters.
        #
        self.__input_shape = input_shape
        self.__num_classes = num_classes
        self.__backbone = backbone
        self.__optimizer_type = optimizer_type
        self.__optimizer_mom = optimizer_mom
        self.__optimizer_wd = optimizer_wd
        self.__head_depth = head_depth
        self.__num_heads = num_heads
        self.__mhead_random = mhead_random
        self.__mhead_eps = mhead_eps
        self.__learning_rate = learning_rate
        self.__dropout_rate = dropout_rate
        self.__layers_per_block = layers_per_block
        self.__growth_rate = growth_rate
        self.__bottleneck = bottleneck
        self.__reduction = reduction
        self.__pool_initial = pool_initial
        self.__pool_final = pool_final
        self.__valid_padding = valid_padding
        self.__bottleneck_rate = bottleneck_rate
        self.__multi_gpu = multi_gpu
        
        self.reset_head_report()

    def _customdata(self):
        """
        Get a map of custom data structure to add to the dictionary of exported values.
        Returns:
            dict: Dictionary of values to add to the exported data.
        """

        return {'input_shape': self.__input_shape,
                'backbone': self.__backbone,
                'optimizer_type': self.__optimizer_type,
                'optimizer_mom': self.__optimizer_mom,
                'optimizer_wd': self.__optimizer_wd,
                'head_depth': self.__head_depth,
                'num_heads': self.__num_heads,
                'mhead_random': self.__mhead_random,
                'mhead_eps': self.__mhead_eps,
                'learning_rate': self.__learning_rate,
                'valid_padding': self.__valid_padding,
                'pool_initial': self.__pool_initial,
                'pool_final': self.__pool_final,
                'growth_rate': self.__growth_rate,
                'num_classes': self.__num_classes,
                'layers_per_block': self.__layers_per_block,
                'bottleneck': self.__bottleneck,
                'reduction': self.__reduction,
                'dropout_rate': self.__dropout_rate,
                'bottleneck_rate': self.__bottleneck_rate, 
                'multi_gpu': self.__multi_gpu}

    def _setcustomdata(self, data_maps):
        """
        Set the class specific data.
        Args:
            data_maps (dict): Custom data map.
        """

        self.__backbone = data_maps.get('backbone', 'densenet')
        self.__optimizer_type = data_maps.get('optimizer_type', 'adam')
        self.__optimizer_mom = data_maps.get('optimizer_mom', 0.9)
        self.__optimizer_wd = data_maps.get('optimizer_wd', 0.0001)
        self.__head_depth = data_maps.get('head_depth', 16)
        self.__num_heads = data_maps.get('num_heads', 5)
        self.__mhead_random = data_maps.get('mhead_random', 0.01)
        self.__mhead_eps = data_maps.get('mhead_eps', 0.05)
        self.__learning_rate = data_maps.get('learning_rate', 0.0001)
        self.__input_shape = data_maps.get('input_shape', [3, 279, 279])
        self.__valid_padding = data_maps.get('valid_padding', True)
        self.__pool_initial = data_maps.get('pool_initial', True)
        self.__pool_final = data_maps.get('pool_final', False)
        self.__growth_rate = data_maps.get('growth_rate', 32)
        self.__num_classes = data_maps.get('num_classes', 2)
        self.__layers_per_block = data_maps.get('layers_per_block', [4, 4, 4])
        self.__bottleneck = data_maps.get('bottleneck', True)
        self.__reduction = data_maps.get('reduction', 0.5)
        self.__dropout_rate = data_maps.get('dropout_rate', 0.0)
        self.__bottleneck_rate = data_maps.get('bottleneck_rate', 2)
        self.__multi_gpu = data_maps.get('multi_gpu', False)

    def _networkdefinition(self):
        """
        Network definition.

        Returns
            Tensor: The network architecture.
        """
        
        if self.__backbone == 'densenet':
            
            return DenseNetMHeads(input_shape = self.__input_shape,
                                  pool_initial = self.__pool_initial,
                                  pool_final = self.__pool_final,
                                  layers_per_block = self.__layers_per_block,
                                  growth_rate = self.__growth_rate,
                                  reduction = self.__reduction,
                                  num_classes = self.__num_classes,
                                  bottleneck = self.__bottleneck,
                                  dropout_rate = self.__dropout_rate,
                                  head_depth = self.__head_depth,
                                  num_heads = self.__num_heads, 
                                  bottleneck_rate = self.__bottleneck_rate,
                                  valid_padding = self.__valid_padding)
        else: 
            raise ValueError('Incorrect model backbone: {}'.format(self.__backbone))
            
    def build(self):
        """Build the network instance with the pre-configured parameters."""

        self._model_instance = self._networkdefinition()
        self._model_instance.cuda()
        
        if self.__multi_gpu:
            self._model_instance = nn.DataParallel(self._model_instance)

        if self.__optimizer_type == 'adam':
            self._optimizer = torch.optim.Adam(self._model_instance.parameters(), lr=self.__learning_rate, weight_decay=self.__optimizer_wd)
        elif self.__optimizer_type == 'sgd':
            self._optimizer = torch.optim.SGD(self._model_instance.parameters(), lr=self.__learning_rate, momentum=self.__optimizer_mom, 
                                              nesterov=True, weight_decay=self.__optimizer_wd)
        else:
            raise ValueError('Incorrect optimizer type: {}'.format(self.__optimizer_type))
            
    def calculate_loss(self, logits, targets, sample_weight=None, class_weight=None, inference=False):
        """
        Expects logits shape: [batch_size, num_classes, num_heads]
        """

        if targets.dim() == 2:
            targets = torch.argmax(targets, dim=1)
            
        # Compute loss for each
        losses = [F.cross_entropy(logits[..., i], targets, weight=class_weight) for i in range(self.__num_heads)]

        # Compute head with least loss
        _, best_head_idx = torch.stack(losses, dim=-1).min(-1)
        best_head_idx = int(best_head_idx.cpu().numpy().squeeze())

        if self._PytorchModelBase__is_training:
            self.head_report['training'][best_head_idx] += 1
        elif inference:
            self.head_report['inference'][best_head_idx] += 1
        else:
            self.head_report['validation'][best_head_idx] += 1

        # Occasionally randomly choose a head to avoid idle heads
        if np.random.binomial(1, self.__mhead_random):
            best_head_idx = np.random.choice(range(self.__num_heads))
        
        # Multiply losses for each head (include random element)
        multiplier  = [self.__mhead_eps / (self.__num_heads - 1) for _ in range(self.__num_heads)]
        multiplier[best_head_idx] = 1 - self.__mhead_eps

        loss = sum([losses[i] * multiplier[i] for i in range(self.__num_heads)])
        return loss
        
    def calculate_metric(self, name, output_values, y, sample_weight):

        # Calculate prediction values.
        #
        softmax_output = torch.nn.functional.softmax(output_values, 1)
        mean_output = torch.mean(softmax_output, dim=-1)
        predictions = torch.argmax(mean_output, dim=1).cpu()

        if name == 'entropy':
            
            all_entropy = []
            for single_sample_output in mean_output:
                
                single_entropy_values = np.copy(single_sample_output.detach().cpu())
                if len(single_entropy_values.shape) == 1:
                    all_entropy.append(entropy(single_entropy_values))
                    continue
                
                for row in range(single_entropy_values.shape[2]):
                    for col in range(single_entropy_values.shape[1]):
                        
                        single_entropy_values[0, row, col] = entropy(single_entropy_values[:, row, col])
                
                all_entropy.append(single_entropy_values[0])
                
            return np.array(all_entropy)
    
        # Calculate ground truth values.
        #
        if y.dim() == 2:
            y = torch.argmax(y, dim=1)
        y = y.cpu()

        metric = None
        if name == 'accuracy':
            if sample_weight is not None:
                metric = (predictions[sample_weight.squeeze(1) > 0] == y[sample_weight.squeeze(1) > 0]).sum()
                metric /= (sample_weight.squeeze(1) > 0).sum()
            else:
                metric = (predictions == y).sum().numpy() / predictions.numel()
        return metric

    def reset_head_report(self, inference=False):
        if inference:
            self.head_report = {'inference': Counter({i: 0 for i in range(self.__num_heads)})}
        else:
            self.head_report = {'training': Counter({i: 0 for i in range(self.__num_heads)}),
                                'validation': Counter({i: 0 for i in range(self.__num_heads)})}
    
    def predict(self, x, y=None, sample_weight=None, class_weight=None, *args, **kwargs):
        """
        Use the network for evaluation.
        Args:
            x (numpy.ndarray or list of numpy.ndarray): contains image data.
        Returns:
            dict: Output of the evaluation function.
        """

        # Calculate reconstruction information if needed
        #
        if self._PytorchModelBase__output_offset is None:
            self._PytorchModelBase__output_offset, _, _ = self.getreconstructioninformation((x.shape[2], x.shape[3]))

        # Turn model into eval mode
        #
        if self._model_instance.training:
            torch.set_grad_enabled(False)
            self._model_instance.eval()
            self._PytorchModelBase__is_training = False
        
        if sample_weight is not None:
            sample_weight = self._PytorchModelBase__matchdatatonetwork(sample_weight)
        
        # Create metric dictionary
        #
        output = {}
        
        # Normalize data to [-1, 1]
        #
        x = (x - 0.5) * 2

        # Predict function is always given BWHC format.
        #
        x = torch.from_numpy(x).cuda()

        # Forward pass
        #
        output_values = self._model_instance(x)
    
        predictions = torch.nn.functional.softmax(output_values, 1)
        predictions = torch.mean(predictions, dim=-1)
        output['predictions'] = predictions.detach().cpu().numpy()

        # Calculate entropy using two identical channels, to make it compatible with our library code.
        #
        output['entropy'] = self.calculate_metric('entropy', output_values, y=None, sample_weight=None)
        output['entropy'] = np.stack([output['entropy'], output['entropy']], 1)

        return output


    def getreconstructioninformation(self, input_shape=None):
        """
        Calculate the scale factor and padding to reconstruct the input shape.

        This function calculates the information needed to reconstruct the input image shape given the output of a layer. For each layer leading up to the
        output it will return the number of pixels lost/gained on all image edges.

        For transposed convolutions the function checks the stride and cropping method to compute the correct upsampling factor.

        Args:
            input_shape (sequence of ints): input_shape to calculate the reconstruction information for. Order should be (width, height)
                set of layers.

        Returns:
            np.array: lost pixels
            np.array: downsample factor
            np.array: interpolation factor

        Raises:
            MissingNetworkError: The network is not defined.
        """

        # PyTorch tactic:
        # 1. Add hooks to all modules
        # 2. Do forward pass with fake data
        # 3. Follow linear path back from prediction via grad_fn
        # 4. Calculate recon-information with this linear path
        #

        # Check input shape, should be two dimensional (width, height).
        #
        if len(input_shape) != 2:
            input_shape = input_shape[:2]
            
        if self._model_instance is None:
            raise dmsmodelerrors.MissingNetworkError()

        # 1. Add hooks to all relevant modules
        #
        def forw_lambda(module, inpt, outpt):
            self._forwardstatshook(module, inpt, outpt)

        hooks = []

        supported_layers = (torch.nn.Conv2d,
                            torch.nn.MaxPool2d,
                            torch.nn.AvgPool2d,
                            torch.nn.AdaptiveAvgPool2d,
                            torch.nn.AdaptiveMaxPool2d,
                            torch.nn.ConvTranspose2d,
                            torch.nn.UpsamplingBilinear2d,
                            torch.nn.UpsamplingNearest2d)

        # TODO: check whether this works for all pytorch models.
        #
        modules = []

        if self.__multi_gpu:
            for mod in self._model_instance.module.layers.modules():
                if isinstance(mod, supported_layers):
                    modules.append(mod)

            for mod in self._model_instance.module.mheads[0].modules():
                if isinstance(mod, supported_layers):
                    modules.append(mod)             
        else:
            for mod in self._model_instance.layers.modules():
                if isinstance(mod, supported_layers):
                    modules.append(mod)

            for mod in self._model_instance.mheads[0].modules():
                if isinstance(mod, supported_layers):
                    modules.append(mod)             


        # Do actual reconstruction information calculation with the linear path
        #
        return self._getreconstructioninformationforlayers(input_shape, modules)

    def _restoremodelparameters(self, parameters):
        """
        Restores the state of the model and the optimizer

        Args:
            parameters: Dictionary of parameters
        """

        if not self._model_instance:
            self.build()
        
        if self.__multi_gpu:
            try: 
                self._model_instance = self._model_instance.module
            except:
                pass
        # Load the state dict
        #
        self._model_instance.load_state_dict(parameters['model'])
        self._optimizer.load_state_dict(parameters['optimizer'])

        if self.__multi_gpu:
            self._model_instance = nn.DataParallel(self._model_instance)
