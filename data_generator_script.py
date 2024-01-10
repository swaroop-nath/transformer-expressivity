import numpy as np
from sklearn.neighbors import KDTree
from tqdm import trange
import os

def data_scheme_one(emb_dim, train_instances, val_instances, test_instances, init_limits=(-1, 1), classes=['10', '20']):
    '''
        Defines dataset according to the following Bayesian Net:
        
            ----         ----
           | X1 | ----> | X2 |
            ----         ----
             |         /
             |       /
            \ /   |/_
            ----         
           | Y1 | 
            ----         
        The functions are (the values -1 and 1 are just place-holders):
        X1 ~ U(-1, 1)
        X2 = cbrt(X1) ε [-1, 1]
        
        Y1 = X1 + X2 ε [-2, 2]
    '''
    def generate_and_store(num_instances, centroid_kd_trees, dataset_path, type='train'):
        x1 = np.random.uniform(low=init_limits[0], high=init_limits[1], size=(num_instances, emb_dim)) # (N, emb_dim)
        x2 = np.cbrt(x1) # (N, emb_dim), noise ~ N(0, 0.01) can be added | σ = 0.1, variance = 0.01
        y1 = x1 + x2 # (N, emb_dim), noise ~ N(0, 0.01) can be added | σ = 0.1, variance = 0.01
        
        input_train = ""
        output_train_reg = ""
        output_train_cls = {k: "" for k in centroid_kd_trees.keys()}
        for idx in trange(num_instances):
            input_instance = str([list(x1[idx]), list(x2[idx])])
            output_instance = str([list(y1[idx])])
            input_train += input_instance + '\n'
            output_train_reg += output_instance + '\n'
            for key in output_train_cls.keys():
                instance_centroid_class_y1 = centroid_kd_trees[key].query([y1[idx]], k=1)[1].squeeze().item()
                output_train_cls[key] += output_instance + ',' + str([instance_centroid_class_y1]) + '\n'
        
        if not os.path.exists(dataset_path + f'regression/'): os.makedirs(dataset_path + f'regression/')
        for key in output_train_cls.keys():
            if not os.path.exists(dataset_path + f'classification/{key}'): os.makedirs(dataset_path + f'classification/{key}')  
        with open(dataset_path + f'regression/{type}_input.csv', 'w') as file:
            file.write(input_train.strip())
        with open(dataset_path + f'regression/{type}_output.csv', 'w') as file:
            file.write(output_train_reg.strip())
            
        for key in output_train_cls.keys():
            with open(dataset_path + f'classification/{key}/{type}_output.csv', 'w') as file:
                file.write(output_train_cls[key].strip())
            with open(dataset_path + f'classification/{key}/{type}_input.csv', 'w') as file:
                file.write(input_train.strip())
        
    dataset_path = f'./synth-data/m2n1-fcbn-cbrt-d{emb_dim}/'
    train_data_instances = train_instances
    test_data_instances = test_instances
    
    quantized_y1_centroids = {k: np.linspace(start=[init_limits[0]]*emb_dim, stop=[init_limits[1]]*emb_dim, num=eval(k)) for k in classes}
    
    centroid_kd_trees = {k: KDTree(v, leaf_size=1) for k, v in quantized_y1_centroids.items()}
    
    # Train
    generate_and_store(train_data_instances, centroid_kd_trees, dataset_path, type='train')
    
    # Valid
    generate_and_store(val_instances, centroid_kd_trees, dataset_path, type='valid')
    
    #Test
    generate_and_store(test_data_instances, centroid_kd_trees, dataset_path, type='test')
  
def data_scheme_mixed(emb_dim, train_instances, val_instances, test_instances, init_limits=(-1, 1), classes=['10', '20']):
    '''
        Defines dataset according to the following Bayesian Net:
        
            ----         ----         ----         ----
           | X1 | ----> | X2 | ----> | X3 | ----> | X4 |
            ----         ----         ----         ----
             |         /  |         /  |         /
             |       /    |       /    |       /
            \ /   |/_    \ /   |/_    \ /   |/_  
            ----         ----         ----         
           | Y1 | ----> | Y2 | ----> | Y3 | 
            ----         ----         ----        
        Additionally, all X's affect all Y's; Y1 affects Y3.
         
        The functions are (the values -1 and 1 are just place-holders):
        X1 ~ U(-1, 1)
        X2 = cbrt(X1) ε [-1, 1] --> Checks for poly function self-attention
        X3 = 2 * log(2 + X1) + X2 / 10 ε [0, log(3)] --> Checks for logarithmic function self-attention
        X4 = exp(X2) + X3 ε [0.3678, 2.7183] --> Checks for exponential function self-attention
        
        Y1 = (X1 + X2 + X3 + X4) / 5 ε [-0.3264, 1.0391] --> Checks for simple addition cross-attention
        Y2 = (X1 * Y1 + exp(X2) + X3 + log(X4)) / 5 ε [-0.3343, 1.0469] --> Checks for multiplication, logarithmic and exponential in cross-attention
        Y3 = (X1 + Y2 + Y1 + sq(X2) + X3 * X4) / 5 ε [-0.3321, 1.0766] --> Checks for multiplication and poly in cross-attention
        
        Normalizing the outputs makes the task a bit easier
    '''
    def generate_and_store(num_instances, centroid_kd_trees, dataset_path, type='train'):
        x1 = np.random.uniform(low=init_limits[0], high=init_limits[1], size=(num_instances, emb_dim))
        x2 = np.cbrt(x1) # Depends on x1
        x3 = 2 * np.log(x1 + 2) + x2 / 10 # Depends on x2 and x1
        x4 = np.exp(x2) + x3 # Depends on x2 and x3
        
        y1 = (x1 + x2 + x3 + x4) / 5
        y2 = (x1 * y1 + np.exp(x2) + x3 + np.log(x4)) / 5
        y3 = (x1 + y2 + y1 + np.square(x2) + x3 * x4) / 5
        
        input_train = []
        output_train_reg = []
        output_train_cls = {k: [] for k in centroid_kd_trees.keys()}
        for idx in trange(num_instances):
            input_instance = str([list(x1[idx]), list(x2[idx]), list(x3[idx]), list(x4[idx])])
            output_instance = str([list(y1[idx]), list(y2[idx]), list(y3[idx])])
            input_train.append(input_instance)
            output_train_reg.append(output_instance)
            for key in output_train_cls.keys():
                instance_centroid_class_y1 = centroid_kd_trees[key].query([y1[idx]], k=1)[1].squeeze().item()
                instance_centroid_class_y2 = centroid_kd_trees[key].query([y2[idx]], k=1)[1].squeeze().item()
                instance_centroid_class_y3 = centroid_kd_trees[key].query([y3[idx]], k=1)[1].squeeze().item()
                output_train_cls[key].append(output_instance + ',' + str([instance_centroid_class_y1, instance_centroid_class_y2, instance_centroid_class_y3]))
        
        input_train = '\n'.join(input_train).strip()
        output_train_reg = '\n'.join(output_train_reg).strip()
        output_train_cls = {k: '\n'.join(v).strip() for k, v in output_train_cls.items()}
        if not os.path.exists(dataset_path + f'regression/'): os.makedirs(dataset_path + f'regression/')
        for key in output_train_cls.keys():
            if not os.path.exists(dataset_path + f'classification/{key}'): os.makedirs(dataset_path + f'classification/{key}')  
        with open(dataset_path + f'regression/{type}_input.csv', 'w') as file:
            file.write(input_train.strip())
        with open(dataset_path + f'regression/{type}_output.csv', 'w') as file:
            file.write(output_train_reg.strip())
            
        for key in output_train_cls.keys():
            with open(dataset_path + f'classification/{key}/{type}_output.csv', 'w') as file:
                file.write(output_train_cls[key].strip())
            with open(dataset_path + f'classification/{key}/{type}_input.csv', 'w') as file:
                file.write(input_train.strip())
                
    dataset_path = f'./synth-data/m4n3-mixbn-mix-d{emb_dim}/'
    train_data_instances = train_instances
    test_data_instances = test_instances
    
    quantized_y1_centroids = {k: np.linspace(start=[init_limits[0]]*emb_dim, stop=[init_limits[1]]*emb_dim, num=eval(k)) for k in classes}
    
    centroid_kd_trees = {k: KDTree(v, leaf_size=1) for k, v in quantized_y1_centroids.items()}
    
    # Train
    generate_and_store(train_data_instances, centroid_kd_trees, dataset_path, type='train')
    
    # Valid
    generate_and_store(val_instances, centroid_kd_trees, dataset_path, type='valid')
    
    #Test
    generate_and_store(test_data_instances, centroid_kd_trees, dataset_path, type='test')
  
if __name__ == '__main__':
    # data_scheme_one(32, int(1e5), int(1e4), int(1e4), classes=['5', '10', '15', '20'])
    data_scheme_mixed(256, int(2*1e5), int(1e4), int(2*1e4), classes=['5', '10', '15', '20'])