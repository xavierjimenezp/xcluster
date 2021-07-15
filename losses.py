#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from tensorflow.keras.losses import binary_crossentropy
import tensorflow.keras.backend as K
import tensorflow as tf
import numpy as np
from sklearn.metrics import precision_score, recall_score
from tensorflow.python.types.core import Value

epsilon = 1e-5
smooth = 1

def iou(y_true, y_pred):
    def f(y_true, y_pred):
        intersection = (y_true * y_pred).sum()
        union = y_true.sum() + y_pred.sum() - intersection
        x = (intersection + 1e-15) / (union + 1e-15)
        x = x.astype(np.float32)
        return x
    return tf.numpy_function(f, [y_true, y_pred], tf.float32)

def f1(y_true, y_pred):
    def f(y_true, y_pred):
        y_true = K.flatten(y_true)
        y_pred = K.flatten(y_pred)
        return 2 * (K.sum(y_true * y_pred)+ K.epsilon()) / (K.sum(y_true) + K.sum(y_pred) + K.epsilon())
    return tf.numpy_function(f, [y_true, y_pred], tf.float32)

def dsc(y_true, y_pred):
    smooth = 0.00001
    tp = K.sum(y_true * y_pred)
    fn = K.sum(y_true * (1-y_pred))
    fp = K.sum((1-y_true) * y_pred)
    # Calculate Dice score
    score = (tp + smooth)/(tp + 0.5*fp + 0.5*fn+ smooth)
    return score

def precision(y_true, y_pred):
    smooth = 0.00001
    tp = K.sum(y_true * y_pred)
    fn = K.sum(y_true * (1-y_pred))
    fp = K.sum((1-y_true) * y_pred)
    # Calculate Dice score
    score = (tp + smooth)/(tp + fp + smooth)
    return score

def recall(y_true, y_pred):
    smooth = 0.00001
    tp = K.sum(y_true * y_pred)
    fn = K.sum(y_true * (1-y_pred))
    fp = K.sum((1-y_true) * y_pred)
    # Calculate Dice score
    score = (tp + smooth)/(tp + fn + smooth)
    return score

def modified_tversky_loss(y_true, y_pred):
    """
    Paper: Tversky loss function for image segmentation using 3D fully convolutional deep networks
    Link: https://arxiv.org/abs/1706.05721
    delta: controls weight given to false positive and false negatives. 
    this equates to the Tversky index when delta = 0.7
    smooth: smoothing constant to prevent division by zero errors
    """
    delta = 0.5
    smooth = 0.00001
    size = 1

    axis = identify_axis(y_true.get_shape())

    # Calculate force true negatives (tn)  
    tn_list = []
    for batch in range(y_true.get_shape()[0]):
        for cl in range(y_pred.get_shape()[3]):
            if cl == 0:
                max_val = tf.reduce_max(y_true[batch,:,:,cl], axis=[0,1], keepdims=True)
                if max_val != 1:
                    tn_list.append(0.)
                    continue
                equal_one = tf.equal(y_true[batch,:,:,cl], max_val)
                one_coords = tf.where(equal_one)
                c_last = one_coords[0]
                c_max = one_coords[0]
                c_max_list = []
                c_min_list = [c_last] 

                for c in one_coords:
                    diff = c - c_last
                    dx = diff[0]
                    dy = diff[1]
                    if dx < 0:
                        dx = -dx
                    if dy < 0:
                        dy = -dy

                    if dx <= 3 and dy <= 3:
                        c_max = c
                        continue
                    else:
                        c_max_list.append(c_max)
                        c_min_list.append(c)
                        c_last = c
                c_max_list.append(c_max)

                assert len(c_max_list) == len(c_min_list)

                tn = []
                for i in range(len(c_min_list)):
                    x_min, y_min = int(c_min_list[i][0]), int(c_min_list[i][1])
                    x_max, y_max = int(c_max_list[i][0])+1, int(c_max_list[i][1])+1
                    # print(x_min)
                    # print(x_max)
                    # print(y_min)
                    # print(y_max)
                    # print(y_true[batch, x_min-size:x_max+size, y_min-size:y_max+size, cl])
                    # print(y_pred[batch, x_min-size:x_max+size, y_min-size:y_max+size, cl])
                    # print(K.sum((1 - y_true[batch, x_min-size:x_max+size, y_min-size:y_max+size, cl]) * y_pred[batch, x_min-size:x_max+size, y_min-size:y_max+size, cl]))
                    tn.append(K.sum((1 - y_true[batch, x_min-size:x_max+size, y_min-size:y_max+size, cl]) * y_pred[batch, x_min-size:x_max+size, y_min-size:y_max+size, cl]))
                tn_list.append(K.sum(tn))
            else:
                tn_list.append(0.)
    
    tn = tf.reshape(tn_list, (y_true.get_shape()[0],y_true.get_shape()[3]))
    # print(tn)

    # Calculate true positives (tp), false negatives (fn) and false positives (fp)
    tp = K.sum(y_true * y_pred, axis=axis)
    fn = K.sum(y_true * (1-y_pred), axis=axis)
    try:
        fp = K.sum((1-y_true) * y_pred, axis=axis) - tn
    except:
        fp = K.sum((1-y_true) * y_pred, axis=axis)
        print(tn)
        print(fp)
        return
    # print(fp)
    # fp = fp - tn
    # print(fp)
    # return

    tversky_class = (tp + smooth)/(tp + delta*fn + (1-delta)*fp + smooth)
    # Sum up classes to one score
    tversky_loss = K.sum(1-tversky_class, axis=[-1])
    # adjusts loss to account for number of classes
    num_classes = K.cast(K.shape(y_true)[-1],'float32')
    tversky_loss = tversky_loss / num_classes

    return tversky_loss

# def dice_loss(y_true, y_pred):
#     loss = 1 - dsc(y_true, y_pred)
#     return loss

# def bce_dice_loss(y_true, y_pred):
#     loss = binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)
#     return loss

# def confusion(y_true, y_pred):
#     smooth=1
#     y_pred_pos = K.clip(y_pred, 0, 1)
#     y_pred_neg = 1 - y_pred_pos
#     y_pos = K.clip(y_true, 0, 1)
#     y_neg = 1 - y_pos
#     tp = K.sum(y_pos * y_pred_pos)
#     fp = K.sum(y_neg * y_pred_pos)
#     fn = K.sum(y_pos * y_pred_neg) 
#     prec = (tp + smooth)/(tp+fp+smooth)
#     recall = (tp+smooth)/(tp+fn+smooth)
#     return prec, recall

# def tp(y_true, y_pred):
#     smooth = 1
#     y_pred_pos = K.round(K.clip(y_pred, 0, 1))
#     y_pos = K.round(K.clip(y_true, 0, 1))
#     tp = (K.sum(y_pos * y_pred_pos) + smooth)/ (K.sum(y_pos) + smooth) 
#     return tp 

# def tn(y_true, y_pred):
#     smooth = 1
#     y_pred_pos = K.round(K.clip(y_pred, 0, 1))
#     y_pred_neg = 1 - y_pred_pos
#     y_pos = K.round(K.clip(y_true, 0, 1))
#     y_neg = 1 - y_pos 
#     tn = (K.sum(y_neg * y_pred_neg) + smooth) / (K.sum(y_neg) + smooth )
#     return tn 

# def tversky(y_true, y_pred):
#     smooth = 1
#     y_true_pos = K.flatten(y_true)
#     y_pred_pos = K.flatten(y_pred)
#     true_pos = K.sum(y_true_pos * y_pred_pos)
#     false_neg = K.sum(y_true_pos * (1-y_pred_pos))
#     false_pos = K.sum((1-y_true_pos)*y_pred_pos)
#     alpha = 0.9
#     return (true_pos + smooth)/(true_pos + alpha*false_neg + (1-alpha)*false_pos + smooth)

# def tversky_loss(y_true, y_pred):
#     return 1 - tversky(y_true,y_pred)

# def focal_tversky(y_true,y_pred):
#     pt_1 = tversky(y_true, y_pred)
#     gamma = 0.75
#     return K.pow((1-pt_1), gamma)



# Helper function to enable loss function to be flexibly used for 
# both 2D or 3D image segmentation - source: https://github.com/frankkramer-lab/MIScnn
def identify_axis(shape):
    # Three dimensional
    if len(shape) == 5 : return [1,2,3]
    # Two dimensional
    elif len(shape) == 4 : return [1,2]
    # Exception - Unknown
    else : raise ValueError('Metric: Shape of tensor is neither 2D or 3D.')


# Dice loss
def dice_loss(y_true, y_pred):
    """
    delta: controls weight given to false positive and false negatives. 
    this equates to the Dice score when delta = 0.5
    smooth: smoothing constant to prevent division by zero errors
    """
    delta = 0.5
    smooth = 0.000001
    axis = identify_axis(y_true.get_shape())
    # Calculate true positives (tp), false negatives (fn) and false positives (fp)
    tp = K.sum(y_true * y_pred, axis=axis)
    fn = K.sum(y_true * (1-y_pred), axis=axis)
    fp = K.sum((1-y_true) * y_pred, axis=axis)
    # Calculate Dice score
    dice_class = (tp + smooth)/(tp + delta*fn + (1-delta)*fp + smooth)
    # Sum up classes to one score
    dice_loss = K.sum(1-dice_class, axis=[-1])
    # adjusts loss to account for number of classes
    num_classes = K.cast(K.shape(y_true)[-1],'float32')
    dice_loss = dice_loss / num_classes

    return dice_loss

# Tversky loss    
def tversky_loss(delta):
    def loss_function(y_true, y_pred):
        """
        Paper: Tversky loss function for image segmentation using 3D fully convolutional deep networks
        Link: https://arxiv.org/abs/1706.05721
        delta: controls weight given to false positive and false negatives. 
        this equates to the Tversky index when delta = 0.7
        smooth: smoothing constant to prevent division by zero errors
        """
        smooth = 0.00001
        axis = identify_axis(y_true.get_shape())
        # Calculate true positives (tp), false negatives (fn) and false positives (fp)   
        tp = K.sum(y_true * y_pred, axis=axis)
        fn = K.sum(y_true * (1-y_pred), axis=axis)
        fp = K.sum((1-y_true) * y_pred, axis=axis)
        tversky_class = (tp + smooth)/(tp + delta*fn + (1-delta)*fp + smooth)
        # Sum up classes to one score
        tversky_loss = K.sum(1-tversky_class, axis=[-1])
        # adjusts loss to account for number of classes
        num_classes = K.cast(K.shape(y_true)[-1],'float32')
        tversky_loss = tversky_loss / num_classes

        return tversky_loss
    return loss_function

# Dice coefficient for use in Combo loss
def dice_coefficient(y_true, y_pred):
    """
    delta: controls weight given to false positive and false negatives. 
    this equates to the Dice score when delta = 0.5
    smooth: smoothing constant to prevent division by zero errors
    """
    delta = 0.5
    smooth = 0.000001
    axis = identify_axis(y_true.get_shape())
    # Calculate true positives (tp), false negatives (fn) and false positives (fp)   
    tp = K.sum(y_true * y_pred, axis=axis)
    fn = K.sum(y_true * (1-y_pred), axis=axis)
    fp = K.sum((1-y_true) * y_pred, axis=axis)
    dice_class = (tp + smooth)/(tp + delta*fn + (1-delta)*fp + smooth)
    # Sum up classes to one score
    dice = K.sum(dice_class, axis=[-1])
    # adjusts loss to account for number of classes
    num_classes = K.cast(K.shape(y_true)[-1],'float32')
    dice = dice / num_classes

    return dice

# Combo loss
def combo_loss(alpha=0.5,beta=0.5):
    """
    Paper: Combo Loss: Handling Input and Output Imbalance in Multi-Organ Segmentation
    Link: https://arxiv.org/abs/1805.02798
    :param beta: controls relative weight of false positives and false negatives. 
                beta > 0.5 penalises false negatives more than false positives.
    :params: alpha controls weighting of dice and cross-entropy loss.
    """
    def loss_function(y_true,y_pred):
        dice = dice_coefficient(y_true, y_pred)
        axis = identify_axis(y_true.get_shape())
        # Clip values to prevent division by zero error
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
        cross_entropy = -y_true * K.log(y_pred)

        if beta is not None:
            beta_weight = np.array([beta, 1-beta])
            cross_entropy = beta_weight * cross_entropy
        # sum over classes
        cross_entropy = K.mean(K.sum(cross_entropy, axis=[-1]))
        if alpha is not None:
            combo_loss = (alpha * cross_entropy) - ((1 - alpha) * dice)
        else:
            combo_loss = cross_entropy - dice
        return combo_loss
        
    return loss_function


# Focal Tversky loss
def focal_tversky_loss(delta=0.3, gamma=0.75):
    def loss_function(y_true, y_pred):
        """
        Paper: A Novel Focal Tversky loss function with improved Attention U-Net for lesion segmentation
        Link: https://arxiv.org/abs/1810.07842
        :param gamma: focal parameter controls degree of down-weighting of easy examples
        
        delta: controls weight given to false positive and false negatives. 
        this equates to the Focal Tversky loss when delta = 0.7
        smooth: smooithing constant to prevent division by 0 errors
        """
        smooth=0.000001
        # Clip values to prevent division by zero error
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1. - epsilon) 
        axis = identify_axis(y_true.get_shape())
        # Calculate true positives (tp), false negatives (fn) and false positives (fp)     
        tp = K.sum(y_true * y_pred, axis=axis)
        fn = K.sum(y_true * (1-y_pred), axis=axis)
        fp = K.sum((1-y_true) * y_pred, axis=axis)
        tversky_class = (tp + smooth)/(tp + delta*fn + (1-delta)*fp + smooth)
        # Sum up classes to one score
        focal_tversky_loss = K.sum(K.pow((1-tversky_class), gamma), axis=[-1])
        # adjusts loss to account for number of classes
        num_classes = K.cast(K.shape(y_true)[-1],'float32')
        focal_tversky_loss = focal_tversky_loss / num_classes
        return focal_tversky_loss

    return loss_function

# (modified) Focal Dice loss
def focal_dice_loss(delta=0.7, gamma_fd=0.75):
    def loss_function(y_true, y_pred):
        """
        :param delta: controls weight given to false positive and false negatives. 
                        this equates to the Focal Tversky loss when delta = 0.7
        :param gamma_fd: focal parameter controls degree of down-weighting of easy examples
        
        smooth: smooithing constant to prevent division by 0 errors
        """
        smooth=0.000001
        # Clip values to prevent division by zero error
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
        axis = identify_axis(y_true.get_shape())
        # Calculate true positives (tp), false negatives (fn) and false positives (fp)     
        tp = K.sum(y_true * y_pred, axis=axis)
        fn = K.sum(y_true * (1-y_pred), axis=axis)
        fp = K.sum((1-y_true) * y_pred, axis=axis)
        dice_class = (tp + smooth)/(tp + delta*fn + (1-delta)*fp + smooth)
        # Sum up classes to one score
        focal_dice_loss = K.sum(K.pow((1-dice_class), gamma_fd), axis=[-1])
        # adjusts loss to account for number of classes
        num_classes = K.cast(K.shape(y_true)[-1],'float32')
        focal_dice_loss = focal_dice_loss / num_classes
        return focal_dice_loss

    return loss_function


# (modified) Focal loss
def focal_loss(alpha=None, beta=None, gamma_f=2.):
    def loss_function(y_true, y_pred):
        """
        :param alpha: controls weight given to each class
        :param beta: controls relative weight of false positives and false negatives. Beta > 0.5 penalises 
                    false negatives more than false positives.
        :param gamma_f: focal parameter controls degree of down-weighting of easy examples. 
        """ 
        axis = identify_axis(y_true.get_shape())
        # Clip values to prevent division by zero error
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
        cross_entropy = -y_true * K.log(y_pred)

        if beta is not None:
            beta_weight = np.array([beta, 1-beta])
            cross_entropy = beta_weight * cross_entropy

        if alpha is not None:
            alpha_weight = np.array(alpha, dtype=np.float32)
            focal_loss = alpha_weight * K.pow(1 - y_pred, gamma_f) * cross_entropy
        else:
            focal_loss = K.pow(1 - y_pred, gamma_f) * cross_entropy

        focal_loss = K.mean(K.sum(focal_loss, axis=[-1]))
        return focal_loss

    return loss_function

# Mixed Focal loss
def mixed_focal_loss(weight=None, alpha=None, beta=None, delta=0.7, gamma_f=2.,gamma_fd=0.75):
    """
    Default is the linear unweighted sum of the Focal loss and Focal Tversky loss
    :param weight: represents lambda parameter and controls weight given to Focal Tversky loss and Focal loss
    :param alpha: controls weight given to each class
    :param beta: controls relative weight of false positives and false negatives. Beta > 0.5 penalises 
                    false negatives more than false positives.
    :param gamma_f: modified Focal loss' focal parameter controls degree of down-weighting of easy examples
    :param gamma_fd: modified Focal Dice loss' focal parameter controls degree of down-weighting of easy examples
    """
    def loss_function(y_true,y_pred):
        # Obtain Focal Dice loss
        focal_dice = focal_dice_loss(delta=delta, gamma_fd=gamma_fd)(y_true,y_pred)
        # Obtain Focal loss
        focal = focal_loss(alpha=alpha, beta=beta, gamma_f=gamma_f)(y_true,y_pred)
        # return weighted sum of Focal loss and Focal Dice loss
        if weight is not None:
            return (weight * focal_dice) + ((1-weight) * focal)  
        else:
            return focal_dice + focal

    return loss_function