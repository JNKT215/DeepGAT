import numpy as np
import argparse

def split_c(h_path,y_path):
    y = np.load(y_path)
    h = np.load(h_path)
    
    c1,c2 = [],[]
    for index,c_i in enumerate(y):
        if c_i == 0:
            c1.append(h[index])
        if c_i == 1:
            c2.append(h[index])
    return np.array(c1),np.array(c2)

def calc_center_of_gravity(c1,c2):
    c1_n,c2_n = c1.shape[0],c2.shape[0]
    c1_sum = c1.sum(axis=0)
    c2_sum = c2.sum(axis=0)
    c1_cog = c1_sum/c1_n
    c2_cog = c2_sum/c2_n
    return c1_cog,c2_cog

def dist_between_c_and_sample(h_path,c1_cog,c2_cog):
    h = np.load(h_path)
    c1_sample_diff = h - c1_cog
    c2_sample_diff = h - c2_cog

    #引いたものを2乗する
    c1_dist = c1_sample_diff **2
    c2_dist = c2_sample_diff **2

    #乗算した結果を，足し合わせる
    c1_dist_sum = c1_dist.sum(axis=1)
    c2_dist_sum = c2_dist.sum(axis=1)

    return c1_dist_sum,c2_dist_sum

def get_miss_classify_num(y_path,c1_dist_sum,c2_dist_sum):
    y = np.load(y_path)
    positve_negative_list = []

    for c1,c2 in zip(c1_dist_sum,c2_dist_sum):
        if c1 < c2:
            positve_negative_list.append(0)
        elif c1 > c2:
            positve_negative_list.append(1)
            
    np_positve_negative_list = np.array(positve_negative_list)
    # print(*positve_negative_list)

    ans =  np.equal(y,np_positve_negative_list)
    # print(ans)
    ans_cnt = np.count_nonzero(ans == False)
    # print(ans_cnt)
    return ans_cnt


def exe(h_path,y_path):
    c1,c2 = split_c(h_path,y_path)
    c1_cog,c2_cog = calc_center_of_gravity(c1,c2)
    c1_dist_sum,c2_dist_sum = dist_between_c_and_sample(h_path,c1_cog,c2_cog)
    miss_classify_cnt = get_miss_classify_num(y_path,c1_dist_sum,c2_dist_sum)
    print(f"miss_classify_cnt:{miss_classify_cnt}")
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default='None')
    parser.add_argument('--y_path', type=str, default='None')
    parser.add_argument('--L0_h_path', type=str, default='None')
    parser.add_argument('--L1_h_path', type=str, default='None')
    parser.add_argument('--L2_h_path', type=str, default='None')
    parser.add_argument('--L3_h_path', type=str, default='None')
    parser.add_argument('--L4_h_path', type=str, default='None')
    parser.add_argument('--L5_h_path', type=str, default='None')
    parser.add_argument('--L6_h_path', type=str, default='None')
    parser.add_argument('--L7_h_path', type=str, default='None')
    parser.add_argument('--L8_h_path', type=str, default='None')
    parser.add_argument('--L9_h_path', type=str, default='None')
    parser.add_argument('--L10_h_path', type=str, default='None')
    parser.add_argument('--L11_h_path', type=str, default='None')
    parser.add_argument('--L12_h_path', type=str, default='None')
    parser.add_argument('--L13_h_path', type=str, default='None')
    parser.add_argument('--L14_h_path', type=str, default='None')
    parser.add_argument('--L15_h_path', type=str, default='None')
    args = parser.parse_args()

    L = [i for i in range(16)]
    h_paths =[
        args.L0_h_path,
        args.L1_h_path,
        args.L2_h_path,
        args.L3_h_path,
        args.L4_h_path,
        args.L5_h_path,
        args.L6_h_path,
        args.L7_h_path,
        args.L8_h_path,
        args.L9_h_path,
        args.L10_h_path,
        args.L11_h_path,
        args.L12_h_path,
        args.L13_h_path,
        args.L14_h_path,
        args.L15_h_path,
    ]
    
    print(f"dataset:{args.name}")
    for l,h_path in zip(L,h_paths):
        print(f":{l}層目")
        exe(h_path,args.y_path)
    
    