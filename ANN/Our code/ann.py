# This is our ANN code 

from collections import Counter,defaultdict
from functools import partial 
import math, random
import numpy as np
import _pickle
import sys
import operator
import matplotlib.pyplot as plt
from sklearn import preprocessing
#from sklearn.metrics import average_precision_score 
#from sklearn.metrics import accuracy_score

npa=np.array

global_variable=0
global_count =0
global_loss_list=[]
bias1=0.2
bias2=0.3

def get_value():
    global global_variable
    return global_variable

def set_value(new_value):
    global  global_variable
    global_variable = new_value

def get_count():
    global global_count
    return global_count

def set_count(new_value):
    global global_count
    global_count = new_value

def get_loss():
    global global_loss_list
    return global_loss_list

def set_loss(new_value):
    global global_loss_list
    global_loss_list.append(new_value)

def set_loss(new_value):
    global global_loss_list
    global_loss_list.append(new_value)

def get_bias():
    global bias1, bias2
    return bias1, bias2

def set_bias(new_b1,new_b2):
    global bias1, bias2
    bias1=new_b1
    bias2=new_b2


#MAKE IMAGES FROM CIFAR-10 DATASET
def read_cifar10(data_type='training', one_hot=True):
    num_channels = 3
    class_num = 10
    img_width = 32
    img_height = 32

    if data_type == 'training':

        file_path = './CIFAR10_data/data_batch_1'
        #print("Loading data: " + file_path)
        data = _pickle.load(open(file_path, mode='rb'), encoding='bytes')
        raw_img = data[b'data']
        label = np.array(data[b'labels'])
        raw_float = np.array(raw_img, dtype=float) / 255.0
        images11 = raw_float.reshape([-1, num_channels* img_height* img_width])
        images1=np.array(images11).tolist()
      
        file_path = './CIFAR10_data/data_batch_2'
        #print("Loading data: " + file_path)
        data = _pickle.load(open(file_path, mode='rb'), encoding='bytes')
        raw_img = data[b'data']
        label = np.concatenate((label, np.array(data[b'labels'])), axis=0)
        raw_float = np.array(raw_img, dtype=float) / 255.0
        images22 = raw_float.reshape([-1, num_channels * img_height * img_width])
        images2=np.array(images22).tolist()

        file_path = './CIFAR10_data/data_batch_3'
        #print("Loading data: " + file_path)
        data = _pickle.load(open(file_path, mode='rb'), encoding='bytes')
        raw_img = data[b'data']
        label = np.concatenate((label, np.array(data[b'labels'])), axis=0)
        raw_float = np.array(raw_img, dtype=float) / 255.0
        images33 = raw_float.reshape([-1, num_channels * img_height * img_width])
        images3=np.array(images33).tolist()

        file_path = './CIFAR10_data/data_batch_4'
        #print("Loading data: " + file_path)
        data = _pickle.load(open(file_path, mode='rb'), encoding='bytes')
        raw_img = data[b'data']
        label = np.concatenate((label, np.array(data[b'labels'])), axis=0)
        raw_float = np.array(raw_img, dtype=float) / 255.0
        images44 = raw_float.reshape([-1, num_channels * img_height * img_width])
        images4=np.array(images44).tolist()

        file_path = './CIFAR10_data/data_batch_5'
        #print("Loading data: " + file_path)
        data = _pickle.load(open(file_path, mode='rb'), encoding='bytes')
        raw_img = data[b'data']
        label = np.concatenate((label, np.array(data[b'labels'])), axis=0)
        raw_float = np.array(raw_img, dtype=float) / 255.0
        images55 = raw_float.reshape([-1, num_channels * img_height * img_width])
        images5=np.array(images55).tolist()


    elif data_type == 'testing':
        file_path = './CIFAR10_data/test_batch'
        #print("Loading data: " + file_path)
        data = _pickle.load(open(file_path, mode='rb'), encoding='bytes')
        raw_img = data[b'data']
        label = np.array(data[b'labels'])
        raw_float = np.array(raw_img, dtype=float) / 255.0
        #images = raw_float.reshape([-1, num_channels, img_height, img_width])
        #img = images.transpose([0, 2, 3, 1])
        images = raw_float.reshape([-1, num_channels * img_height * img_width])

    else:

        raise ValueError("data_type must be 'training' or 'testing'")

    if one_hot == True:
        data_num, = np.shape(label)
        output_vector = np.empty((data_num, class_num))
        for i in range(data_num):
            class_value = label[i]
            for j in range(class_num):
                if j == class_value:
                    output_vector[i, j] = 1
                else:
                    output_vector[i, j] = 0
        label = output_vector

    return images1,images2,images3,images4,images5, label

#MAKE ZIP FUNCTION
def dot(v, w):
    #"""v_1 * w_1 + ... + v_n * w_n"""
    return sum(v_i * w_i for v_i, w_i in zip(v, w))

#MAKE ESTIMATE FUNCTION 
def estimate_accuracy(cur_label,outputs, image_num):
    label_max_index , vl= max(enumerate(cur_label), key=operator.itemgetter(1))
    outputs_max_index, vo= max(enumerate(outputs), key=operator.itemgetter(1))

    g=get_count()
   
    if label_max_index==outputs_max_index:
        g=g+1
        set_count(g)
    print("%d  training: accuracy (account correct classify)..... %f" %(image_num-1, (g / image_num)*100))
    #print("training: mean precision (account all 10 output score) %f" %average_precision_score(cur_label,outputs))
    return 1


#ACTIVATION FUNCTION 1
def sigmoid(yHat):
    s =[1.0/(1.0+math.exp(-_yHat)) for _yHat in yHat]
    return (s)

#ACTIVATION FUNCTION 2
def ReLU(yHat):
    r=[]
    for _yHat in yHat:
        if _yHat>0:
            r.append(_yHat)
        else :
            r.append(0)
    return r 

#ACTIVATION FUNCTION 3
def softmax(yHat):
    tempmax=0
    tempmax=max(yHat)
    #yHat-=tempmax
    s =[_yHat for _yHat in yHat]
    return np.exp(s)/np.sum(np.exp(s))

#CALCULATE NET(X*W+B)
def neuron_output(weights,inputs, bias):
    b=0
    bias1,bias2=get_bias()        
    if bias == 0:
    	b=bias1
    else :
        b= bias2   
  
    a=(dot(weights, inputs)+bias)    
    return a

#CACULATE HIDDEN/OUTPUT LATER'S UNIT FOR INPUT
def feed_forward(neural_network, input_vector):
    outputarr=[]

    for i,layer in enumerate(neural_network): 
        input_with_bias=input_vector
        output =[neuron_output(neuron_weight,input_with_bias, (i+1)/10) for neuron_weight in layer] 
        #print(output)
        
        #change activation function
        if i==0:
            activated_output=np.log(ReLU(output))
        else:
            activated_output=np.log(ReLU(output))

        #print( activated_output)
        outputarr.append(activated_output)
        input_vector = activated_output

    return outputarr 

#MAKE FUNCTION TO ADJUST WEIGHTS
def backpropagate(network,input_vectors, target,hidden_size):
 
    target_index =[]
    output_index=[]
    rate = 0.001

    #print()
    tr=get_value() 
    #print(tr) 

    hidden_weights, output_weights = network
    hidden_units, output_units = feed_forward(network, input_vectors) 
    b1,b2=get_bias()

    #estimate_accuracy
    estimate_accuracy(target, output_units, tr+1)

    #####ADJUST WEIGHT THROUGH BACKPRO USING ACTIVATION FUNCTION####
    new_output_weights=output_weights
    new_hidden_weights=hidden_weights

    #ERROR TO OUTPUT LAYER (sigmoid)
    #output_deltas = [output*(1-output)*(output-target[i]) for i,output in enumerate(output_units)]
    #ERROR TO OUTPUT LAYER (ReLU)
    output_deltas = [output*(output-target[i]) for i,output in enumerate(output_units)]
    b1-=rate * np.average(output_deltas,axis=0)
   
    #ADJUST WEIGHTS FOR OUTPUT LAYER (NETWORK[-1])
    for i,output_weight in enumerate(output_weights):
        for j, hidden_unit in enumerate(hidden_units):
            output_weight[j]-=output_deltas[i]*hidden_unit*rate
            new_output_weights[i][j]=output_weight[j]
   
    #ERROR TO HIDDEN LAYER (sigmoid)
    #hidden_deltas=[hidden_unit*(1-hidden_unit)*dot(output_deltas,[n[i] for n in output_weights]) for i, hidden_unit in enumerate(hidden_units)]
    #ERROR TO HIDDEN LAYER (ReLU)
    hidden_deltas=[hidden_unit*dot(output_deltas,[n[i] for n in output_weights]) for i, hidden_unit in enumerate(hidden_units)]
    b2-=rate * np.average(hidden_deltas,axis=0)

    #ADJUST WEIGHTS FOR HIDDEN LAYER (NETWORK[0])
    for i, hidden_weight in enumerate(hidden_weights):
        for j, input_vector in enumerate(input_vectors):
            hidden_weight[j]-=hidden_deltas[i]*input_vector*rate
            new_hidden_weights[i][j]=hidden_weight[j]

    #UPDATE NETWORK
    netework=[new_hidden_weights,new_output_weights]
    set_value(tr+1)
    set_bias(b1,b2)

    return network

#START-INITIALIZING,TRAINING(BACKPROPA,FEEDFOWARD),TESTING
def main():
    def predict(input,tag):
        return feed_forward(network,input)[-1]

    #MAKE TRAINING DATASET
    img1,img2,img3,img4,img5,labels=read_cifar10('training',True)
    label=np.array(labels).tolist()
    #MAKE TEST DATASET
    num_channels = 3
    class_num = 10
    img_width = 32
    img_height = 32  
 
    file_path = './CIFAR10_data/test_batch'
    data = _pickle.load(open(file_path, mode='rb'), encoding='bytes')
    raw_img = data[b'data']
    train_label = np.array(data[b'labels'])
    raw_float = np.array(raw_img, dtype=float) / 255.0
    images = raw_float.reshape([-1, num_channels * img_height * img_width])

    data_num, = np.shape(train_label)
    output_vector = np.empty((data_num, class_num))
    for i in range(data_num):
        class_value = train_label[i]
        for j in range(class_num):
            if j == class_value:
                output_vector[i, j] = 1
            else:
               output_vector[i, j] = 0
    train_label = output_vector

    #preprocessing
    #print("preprocessing...")
    #_img1,_images= preprocessing(img1,images)
    #_img1 = preprocessing.scale(img1)
    #_images = preprocessing.scale(images)

    index_list =[]
    data_list=[]
    for i in train_label[0:1000]:
        index,value=max(enumerate(i), key=operator.itemgetter(1))
        index_list.append(index)
 
    #INITAILIZE NETWORK
    random.seed(1)
    input_size=32*32*3
    num_hidden =100
    output_size=10
    minibatch=200
    epoch=20

    hidden_layer = [[random.uniform(0.0,0.1) for __ in range(input_size)] for __ in range(num_hidden)]
    output_layer = [[random.uniform(0.0,0.1) for __ in range(num_hidden)] for __ in range(output_size)]
    network = [hidden_layer, output_layer]
    #print("check-1")
    #print(output_layer[5][5])


    #print("###############################backpropa#################################")
    #BACKPROPAGATION TO ADJUST WEIGHT

   # setting minibatch = 20
    for iter in range(epoch):
        for i in range(0, 10000, minibatch):
            for input_vector, target_vector in zip(img1[i:i+minibatch],labels[i:i+minibatch]):
                newtork=backpropagate(network, input_vector, target_vector,num_hidden)
  
    #################################TEST############################################
    #SAMPLE TEST
    set_count(0) 
    for ii, input in enumerate(images[0:1000]):
        outputs=predict(input[0:32*32*3],1)
        index2,valu2=max(enumerate(outputs), key=operator.itemgetter(1))
        data_list.append(index2)
        #print ("######################testting image number... %d" %ii)
        #print(softmax(outputs))
        acc=estimate_accuracy(train_label[ii],outputs, ii+1)
        set_acc(acc)
        losss=np.square(outputs-train_label[ii]).sum()
        #set_loss(loss)
        print("loss=outputs-target ... %f" %(losss))
   
    t= [i for i in range(1000)]
    plt.figure()
    plt.plot(t, get_acc())
    plt.show()
    #total accuracy
    #a=accuracy_score(index_list,data_list)
    #print("total accyracy!!... %d" %a)
 
    return network


####################main##########################
main()
print("neural network finish!")



