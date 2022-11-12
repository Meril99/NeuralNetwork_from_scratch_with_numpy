import struct
import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
import math
from sklearn.datasets import make_blobs
from tqdm import tqdm #barre de progression taquadoum
import numpy as np
import warnings




#suppress warnings
warnings.filterwarnings('ignore')
#sav files
file1='t10k-images.idx3-ubyte'
file2='t10k-labels.idx1-ubyte'
file3='train-images.idx3-ubyte'
file4='train-labels.idx1-ubyte'
"""
@authors: MIANGOUILA Méril 
"""



# provided function for reading idx files
def read_idx(filename):
    '''Reads an idx file and returns an ndarray'''
    with open(filename, 'rb') as f:
        zero, data_type, dims = struct.unpack('>HBB', f.read(4))
        shape = tuple(struct.unpack('>I', f.read(4))[0] for d in range(dims))
        return np.frombuffer(f.read(), dtype=np.uint8).reshape(shape)

# Task 1: reading the MNIST files into Python ndarrays
        
# Task 2: visualize a few bitmap images
        
# Task 3: input pre-preprocessing    

# Task 4: output pre-processing
        
# Task 5-6: creating and initializing matrices of weights
        
# Task 7: defining functions sigmoid, softmax, and sigmoid'
        
# Task 8-9: forward pass
        
# Task 10: backpropagation
        
# Task 11: weight updates
        
# Task 12: computing error on test data
        
# Task 13: error with initial weights
        
# Task 14-15: training

# Task 16-18: batch training




def read_idx(filename):
    '''Reads an idx file and returns an ndarray'''
    with open(filename, 'rb') as f:
        zero, data_type, dims = struct.unpack('>HBB', f.read(4))
        shape = tuple(struct.unpack('>I', f.read(4))[0] for d in range(dims))
        return np.frombuffer(f.read(), dtype=np.uint8).reshape(shape)

# def preprocessing(arr3d_array):
#     arr2d_array=arr3d_array.reshape(arr3d_array.shape[0],arr3d_array.shape[1]*arr3d_array.shape[2])
#     arr2d_array = normalize(arr2d_array)
#     return arr2d_array
  
def preprocessing(arr3d_array):
    arr2d_array=arr3d_array.reshape(arr3d_array.shape[0],arr3d_array.shape[1]*arr3d_array.shape[2])
    arr2d_array = (arr2d_array - arr2d_array.min())/ (arr2d_array.max() - arr2d_array.min())
    return arr2d_array
    

def preprocessing_label(vector):
    matrice = np.zeros((vector.size, 10))
    matrice[np.arange(vector.size), vector] = 1
    return matrice
    
    

def graphic_view():
    X, y = make_blobs(n_samples=60000, n_features=28*28, centers=10, random_state=0)
    y = y.reshape((y.shape[0], 1))

    print(X.shape,y.shape)
    print('dimensions de X:', X.shape)
    print('dimensions de y:', y.shape)

    plt.scatter(X[:,0], X[:, 1], c=y, cmap='summer')
    plt.show()

def weights_matrice(m,n):
    matrix=np.random.standard_normal((m,n))
    b = np.random.randn(1)
    sqrt=math.sqrt(n)
    scalaire=1/sqrt
    matrix=scalaire*matrix
    return matrix
    
    

def sigmoide(z_i):
    return 1/(1+np.exp(-z_i))

# normaliser poids
def softmax(array):
    return (np.exp(array - array.max()))/np.sum(np.exp(array - array.max()))


def dsigmoid(x):
    return (np.exp(-x))/((np.exp(-x)+1)**2)






def forward_pass_batch(batch,w1_all__,w2_all__,w3_all__):
    z1_all_images= batch.dot(w1_all__) 
    a1_all_images= sigmoide(z1_all_images) 
    z2_all_images=a1_all_images.dot(w2_all__) 
    a2_all_images=sigmoide(z2_all_images) 
    z3_all_images=a2_all_images.dot(w3_all__) 
    a3_all_images=softmax(z3_all_images) 
    return batch, z1_all_images, z2_all_images, z3_all_images, a1_all_images, a2_all_images, a3_all_images

    


    
def backpropagation_batch(batch_,z1_all_images_,z2_all_images_,z3_all_images_,a1_all_images_,a2_all_images_,a3_all_images_,label_,w1__,w2__,w3__):
    e_3_=a3_all_images_-label_ 
    e_2_=np.multiply(e_3_.dot(w3__.T),dsigmoid(z2_all_images_)) 
    e_1_=np.multiply(e_2_.dot(w2__.T),dsigmoid(z1_all_images_)) 
    delta_w3_all=a2_all_images_.T.dot(e_3_) 
    delta_w2_all=a1_all_images_.T.dot(e_2_) 
    delta_w1_all=batch_.T.dot(e_1_) 
    return delta_w3_all, delta_w2_all, delta_w1_all
    
    

    
    
    
    

def update(dw1, dw2,dw3,w_1__,w_2__,w_3__, lambd):
    w_1_new = w_1__ - (lambd*dw1)
    w_2_new = w_2__ - (lambd*dw2)
    w_3_new = w_3__ - (lambd*dw3)
    return (w_1_new,w_2_new,w_3_new)



def compute_error(A,y):
    epsilon = 1e-15
    return 1 / len(y) * np.sum(-y * np.log(A + epsilon) - (1 - y) * np.log(1 - A + epsilon))

def compute_batch_error(test_data_x_test,test_data_label_test,updated_w1,updated_w2,updated_w3):
    error_total=np.empty([0,test_data_x_test.shape[0]],dtype=float)
    for j in range(0,test_data_x_test.shape[0]):
        new_test_data,new_z1__,new_z2__,new_z3__,new_a1__,new_a2__,new_a3__=forward_pass_batch(test_data_x_test[j],updated_w1,updated_w2,updated_w3)

        indice_max_y=np.argmax(test_data_label_test[j],axis=1)
       
        indice_max_a=np.argmax(new_a3__,axis=1)
       
        error_per_batch=1-np.mean(indice_max_a==indice_max_y)
        error_total=np.append(error_total,error_per_batch)
        
    return np.mean(error_total)






def intialisation(m,n,p,c):
    poids1=weights_matrice(m,n) 
    poids2=weights_matrice(n,p)
    poids3=weights_matrice(p,c)
    return (poids1,poids2,poids3)



def train_sets_batch(test_data_x,test_data_y,train_data_x, train_label_array, learning_rate = 0.155555555555, epochs = 100,batch=60):
    
    train_data_x=preprocessing(train_data_x)
    train_label_array=preprocessing_label(train_label_array)
    test_data_x=preprocessing(test_data_x)
    test_data_y=preprocessing_label(test_data_y)
    
    sets_list=np.array_split(train_data_x, batch)
    sets_array=np.array(sets_list)
    y_sets=np.array_split(train_label_array, batch)
    y_sets_array=np.array(y_sets)
    batch_array=sets_array
    
    x_tests=np.array_split(test_data_x, batch/6)
    x_tests=np.array(x_tests)
    
    y_tests=np.array_split(test_data_y, batch/6)
    y_tests=np.array(y_tests)
    

    w1,w2,w3=intialisation(784,128,64,10)
    for i in range(0,epochs):
        for j in range(0,batch):
            new_batch,z1__batch,z2__batch,z3__batch,a1__batch,a2__batch,a3__batch=forward_pass_batch(batch_array[j],w1,w2,w3)
            delta_w3__,delta_w2__,delta_w1__=backpropagation_batch(new_batch,z1__batch,z2__batch,z3__batch,a1__batch,a2__batch,a3__batch,y_sets_array[j],w1,w2,w3)
            w1_updated__,w2_updated__,w3_updated__=update(delta_w1__, delta_w2__,delta_w3__,w1,w2,w3, learning_rate)
            w1,w2,w3=w1_updated__,w2_updated__,w3_updated__
        mean_total_errors=compute_batch_error(x_tests,y_tests,w1_updated__,w2_updated__,w3_updated__)
        print('epoch: ' + str(i) + ' error rate: ' + str(mean_total_errors)+'\n')
    error=mean_total_errors
    return error





if __name__ == "__main__":
    x_train,y_train,x_test,y_test=read_idx(file3),read_idx(file4),read_idx(file1),read_idx(file2)
    train_sets_batch(x_test,y_test,x_train, y_train)