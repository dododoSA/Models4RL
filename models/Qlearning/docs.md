# Q Learning
## �T�v
TD�덷�Ɋ�Â��čs�����l�֐�Q���X�V����TD�w�K�̈�B

��q�̍X�V���̒ʂ�A���ۂɍ̗p���Ă��Ȃ��s���𗘗p���Ă���̂ŁA����I�t�^�ł��邱�Ƃ��킩��

�L���}���R�t����ߒ��ɂ����đS�Ă̏�Ԃ��\���ɃT���v�����O�ł���悤�ȃG�s�\�[�h�𖳌���{�s�����ꍇ�A�œK�ȕ]���l�Ɏ������邱�Ƃ����_�I�ɏؖ�����Ă���B

�Q�l�F https://ja.wikipedia.org/wiki/Q%E5%AD%A6%E7%BF%92

## �X�V��


```math
Q(s,a)��Q(s,a)+\alpha(r+\gamma max_{a'}Q(s', a') - Q(s,a))
```

## ����

Q�e�[�u���𒼐ڒ�`����̂Ŋϑ������A���l�͗��U�l�ɕϊ�����K�v������

```python
# observation:              ���ʓ���ꂽ�V���ȏ��
# self.observation_space:   Agent�����炩���ߗ^�����Ă����ԋ�Ԃ̏��
# self.discretize_nums:     ��ԋ�Ԃ��ǂ̂悤�ɕ������邩������킷�ϐ�
next_state = discretize_Box_state(observation, self.observation_space, self.discretize_nums)
```

Q�e�[�u���̍X�V
```
def update_q_table(self, reward, next_state) -> None:
        td_error = reward + self.gamma*np.max(self.q_table[next_state]) - self.q_table[self.state, self.action]
        self.q_table[self.state, self.action] += self.alpha*td_error
```