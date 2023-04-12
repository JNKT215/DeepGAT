import numpy as np
import os
import matplotlib.pyplot as plt
import argparse


def calc_kldiv(a_vec, b_vec):
    return np.sum([a * np.log(a/b) for a, b in zip(a_vec, b_vec)])

def calc_v_attention_kldiv_ver2(L2_attention,L9_attention):
    output = []
    for v_attention_l2, v_attention_l9 in zip(L2_attention, L9_attention):
        epsilon_vec = np.array([1e-5 for _ in range(v_attention_l2.shape[0])])
        v_attention_l2 += epsilon_vec
        v_attention_l9 += epsilon_vec
        kl_div = calc_kldiv(v_attention_l2, v_attention_l9)
        output.append(kl_div)
    return output

def visualize_attention_kldiv(kl_divs,save_dir,output_name,outlier):
    divs  = [v for n, v in kl_divs.items()]
    names = [n for n, v in kl_divs.items()]

    fig, ax = plt.subplots()
    ax.boxplot(divs, sym=outlier)
    ax.set_xticklabels(names)
    # ax.set_ylabel("KLD(Att_l=2,Att_l=9)")
    plt.tick_params(labelsize=18)
    plt.ylim([-0.1,3.1])
    ax.set_yticks([0,1,2,3])
    plt.savefig(f'{save_dir}{output_name}.png')

def load_attention(args):
    DeepGAT_L2_Attention = np.load(args.DeepGAT_L2_att, allow_pickle=True)
    DeepGAT_L9_Attention = np.load(args.DeepGAT_L9_att, allow_pickle=True)
    GAT_L2_Attention = np.load(args.GAT_L2_att, allow_pickle=True)
    GAT_L9_Attention = np.load(args.GAT_L9_att, allow_pickle=True)
    return DeepGAT_L2_Attention,DeepGAT_L9_Attention,GAT_L2_Attention,GAT_L9_Attention
    
    

if __name__ == "__main__":
    save_dir = "DeepGAT/output/kldiv/"
    os.makedirs(save_dir, exist_ok=True)
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default='None')
    parser.add_argument('--DeepGAT_L2_att', type=str, default='None')
    parser.add_argument('--DeepGAT_L9_att', type=str, default='None')
    parser.add_argument('--GAT_L2_att', type=str, default='None')
    parser.add_argument('--GAT_L9_att', type=str, default='None')
    parser.add_argument('--output_name', type=str, default='None')
    parser.add_argument('--outlier', type=str, default='')
    args = parser.parse_args()

    
    DeepGAT_L2_Attention,DeepGAT_L9_Attention,GAT_L2_Attention,GAT_L9_Attention = load_attention(args)
    
    
    kl_divs = {}
    kl_divs["DeepGAT"] = calc_v_attention_kldiv_ver2(DeepGAT_L2_Attention,DeepGAT_L9_Attention)
    kl_divs["GAT"] = calc_v_attention_kldiv_ver2(GAT_L2_Attention,GAT_L9_Attention)
    print(f"dataset:{args.name}")
    visualize_attention_kldiv(kl_divs,save_dir,args.output_name,args.outlier)
    
    