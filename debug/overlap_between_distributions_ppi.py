import numpy as np
import argparse

def overlap_nodes_num_ppi(train_h_path,train_y_path,test_h_path,test_y_path):
    unsupervised_h = np.load(test_h_path)
    unsupervised_y = np.load(test_y_path)
    supervised_h = np.load(train_h_path)
    supervised_y = np.load(train_y_path)

    cnt = 0
    for u,u_y in zip(unsupervised_h,unsupervised_y):
        dist_temp,y_temp = [],[]
        for v,v_y in zip(supervised_h,supervised_y):
            dist = np.linalg.norm(v-u,ord=2)
            dist**=2
            dist_temp.append(dist)
            y_temp.append(v_y)
        dist_temp_np = np.array(dist_temp)
        y_temp_np = np.array(y_temp)
        
        v_y_index = np.argmin(dist_temp_np)
        u_y_pred = y_temp_np[v_y_index]
        if u_y != u_y_pred:
            cnt+=1
    print(f"overlaps_nodes_num:{cnt}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default='None')
    #train
    parser.add_argument('--train_y_path', type=str, default='None')
    parser.add_argument('--L0_train_h_path', type=str, default='None')
    parser.add_argument('--L1_train_h_path', type=str, default='None')
    parser.add_argument('--L2_train_h_path', type=str, default='None')
    parser.add_argument('--L3_train_h_path', type=str, default='None')
    parser.add_argument('--L4_train_h_path', type=str, default='None')
    parser.add_argument('--L5_train_h_path', type=str, default='None')
    parser.add_argument('--L6_train_h_path', type=str, default='None')
    parser.add_argument('--L7_train_h_path', type=str, default='None')
    parser.add_argument('--L8_train_h_path', type=str, default='None')
    parser.add_argument('--L9_train_h_path', type=str, default='None')
    #test
    parser.add_argument('--test_y_path', type=str, default='None')
    parser.add_argument('--L0_test_h_path', type=str, default='None')
    parser.add_argument('--L1_test_h_path', type=str, default='None')
    parser.add_argument('--L2_test_h_path', type=str, default='None')
    parser.add_argument('--L3_test_h_path', type=str, default='None')
    parser.add_argument('--L4_test_h_path', type=str, default='None')
    parser.add_argument('--L5_test_h_path', type=str, default='None')
    parser.add_argument('--L6_test_h_path', type=str, default='None')
    parser.add_argument('--L7_test_h_path', type=str, default='None')
    parser.add_argument('--L8_test_h_path', type=str, default='None')
    parser.add_argument('--L9_test_h_path', type=str, default='None')

    args = parser.parse_args()
    

    L = [i for i in range(16)]
    train_h_paths =[
        args.L0_train_h_path,
        args.L1_train_h_path,
        args.L2_train_h_path,
        args.L3_train_h_path,
        args.L4_train_h_path,
        args.L5_train_h_path,
        args.L6_train_h_path,
        args.L7_train_h_path,
        args.L8_train_h_path,
        args.L9_train_h_path,
    ]
    test_h_paths =[
        args.L0_test_h_path,
        args.L1_test_h_path,
        args.L2_test_h_path,
        args.L3_test_h_path,
        args.L4_test_h_path,
        args.L5_test_h_path,
        args.L6_test_h_path,
        args.L7_test_h_path,
        args.L8_test_h_path,
        args.L9_test_h_path,
    ]
    
    print(f"dataset:{args.name}")
    for l,train_h_path,test_h_path in zip(L,train_h_paths,test_h_paths):
        print(f":{l}層目")
        overlap_nodes_num_ppi(train_h_path,args.train_y_path,test_h_path,args.test_y_path)