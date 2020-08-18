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
def update_q_table(self, reward, next_state) -> None:
        td_error = reward + self.gamma*np.max(self.q_table[next_state]) - self.q_table[self.state, self.action]
        self.q_table[self.state, self.action] += self.alpha*td_error
```