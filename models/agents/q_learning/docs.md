# Q Learning
## 概要
TD誤差に基づいて行動価値関数Qを更新するTD学習の一つ。

後述の更新式の通り、実際に採用していない行動を利用しているので、方策オフ型であることがわかる

有限マルコフ決定過程において全ての状態が十分にサンプリングできるようなエピソードを無限回施行した場合、最適な評価値に収束することが理論的に証明されている。

参考： https://ja.wikipedia.org/wiki/Q%E5%AD%A6%E7%BF%92

## 更新式


```math
Q(s,a)←Q(s,a)+\alpha(r+\gamma max_{a'}Q(s', a') - Q(s,a))
```

## 実装

Qテーブルを直接定義するので観測した連続値は離散値に変換する必要がある

```python
# observation:              結果得られた新たな状態
# self.observation_space:   Agentがあらかじめ与えられている状態空間の情報
# self.discretize_nums:     状態空間をどのように分割するかをあらわす変数
next_state = discretize_Box_state(observation, self.observation_space, self.discretize_nums)
```

Qテーブルの更新
```
# self.state: next_stateの一個前の状態
# self.action: self.stateにてとった行動
def update_q_table(self, reward, next_state) -> None:
        td_error = reward + self.gamma*np.max(self.q_table[next_state]) - self.q_table[self.state, self.action]
        self.q_table[self.state, self.action] += self.alpha*td_error
```

## 実験結果
Qテーブルは-0.1~0.1でランダムに初期化

### 状態、報酬設計
https://github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py
```
Observation:
        Type: Box(4)
        Num     Observation               Min                     Max
        0       Cart Position             -4.8                    4.8
        1       Cart Velocity             -Inf                    Inf
        2       Pole Angle                -0.418 rad (-24 deg)    0.418 rad (24 deg)
        3       Pole Angular Velocity     -Inf                    Inf
```
状態の分割数は順に8,2,8,2(infは8分割してもほぼ意味ないので)

```
Reward:
        Reward is 1 for every step taken, including the termination step
        ＋ 195step以下で失敗した場合-200
```

### 探索アルゴリズム
ε-greedy法
```
epsilon = 1 / episode
```

### グラフ
直近10ステップでの平均
稀にほぼ200みたいな時があるけどだいたいこんな感じ
![result1](https://user-images.githubusercontent.com/32331100/90546872-a322df80-e1c5-11ea-8d01-f44f9a52812e.png)
