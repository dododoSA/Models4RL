import numpy as np
import torch
import torch.nn as nn


def _gen_sequence(s, layer_class, af, oaf):
    layers = []
    for j in range(len(s) - 1):
        a = af if j < len(s) - 2 else oaf
        layers += [layer_class(s[j], s[j+1]), a()]

    return nn.Sequential(*layers)


def make_linear_network(sizes, activation_func, output_activation_func=nn.Identity):
    """
    全結合層を作成する関数
    Args:
        sizes (list like): [input, ... , intermediate, ... , output]
        activation_func (nn.Module): 活性化関数
        output_activation_func (nn.Module): 出力層の活性化関数
    """
    return _gen_sequence(sizes, nn.Linear, activation_func, output_activation_func)


def make_dueling_network(adv_sizes, val_sizes, adv_act_func, val_act_func):
    """
    Dueling Networkを作成する関数
    Args:
        adv_sizes (list like): [hidden_output, ... , adv_output]
        val_sizes (list like): [hidden_output, ... , val_output]
        adv_activation_func (nn.Module): 活性化関数(出力層は活性化なし)
        val_activation_func (nn.Module): 活性化関数(出力層は活性化なし)
    """
    adv_sequential = _gen_sequence(adv_sizes, nn.Linear, adv_act_func, nn.Identity)
    val_sequential = _gen_sequence(val_sizes, nn.Linear, val_act_func, nn.Identity)

    return adv_sequential, val_sequential


def dueling_forward(x, adv_seq, val_seq):
    """
    アドバンテージに関するネットワークと、状態価値に関するネットワークからQ値を出力する関数
    Args:
        x (hidden outputs): 基本的に中間層からの出力を入力
        adv_seq: アドバンテージに関するネットワーク
        val_seq: 状態価値に関するネットワーク

    Reterns:
        Q値
    """
    adv = adv_seq(x)
    val = val_seq(x)

    adv = adv.repeat(1, 1) # この辺のテンソルの扱いがへたくそなのでそのうち修正
    val = val.repeat(1, adv.size(1))

    tmp = adv.mean(1).unsqueeze(1)
    output = val + adv - tmp.expand(tmp.size(0), adv.size(1))
    return output.squeeze(0)




# ==================== Actor Critic ======================
# 出来たら整理したい

class QNetwork(nn.Module): # よくよく考えたらqじゃなくてvな気がしてきた
    def __init__(self, sizes):
        super().__init__()
        # example
        # sizes = [state_num] + [hidden_size] * 2 + [1]
        self.fc = make_linear_network(sizes, nn.ELU, nn.ELU)

    def forward(self, x):
        return self.fc(x)


class PolicyNetwork(nn.Module):
    def __init__(self, sizes):
        super().__init__()
        # example
        # sizes = [state_num] + [hidden_size] + [action_num]
        self.fc = make_linear_network(sizes, nn.ELU, nn.Softmax)

    def forward(self, x):
        return self.fc(x)


class ACNetwork(nn.Module):
    """
    価値を推定するネットワークと方策を学習するネットワークを完全別個として定義
    """
    def __init__(self, q_sizes, p_sizes):
        super().__init__()
        self.p = PolicyNetwork(p_sizes)
        self.q = QNetwork(q_sizes)


def create_ac_network(q_sizes, p_sizes):
    return ACNetwork(q_sizes, p_sizes)