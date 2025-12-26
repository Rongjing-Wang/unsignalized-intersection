#替代缓冲区
from collections import deque
import random
import rank_based


class ReplayBuffer(object):

    def __init__(self, buffer_size, batch_size=32, learn_start=2000, steps=100000, rand_s=False):
        # buffer_size：缓冲区的最大容量（最多能存储多少个经验）。
        # batch_size：在采样时每次抽取的经验数量，默认设为32。
        # learn_start：开始学习之前需要收集的步数（或经验数量），默认值为2000。
        # steps：可能是整个训练过程中希望采样的总步数，默认100000（具体含义依赖rank_based模块的实现）。
        # rand_s：布尔值，决定是否使用内置的随机采样（当为True时使用Python内置的随机采样方法；否则使用rank_based.Experience进行采样）。
        self.buffer_size = buffer_size # 缓冲区的最大大小
        self.num_experiences = 0  # 当前存储的经验数量
        self.buffer = deque()  # 使用双端队列来存储经验
        self.rand_s = rand_s  # 决定是否使用随机采样
        conf = {'size': self.buffer_size,
                'learn_start': learn_start,
                'partition_num': 32,
                'steps': steps,
                'batch_size': batch_size} # 配置rank_based.Experience所需的参数
        self.replay_memory = rank_based.Experience(conf)  # 创建一个rank_based.Experience实例，用于经验采样

    def getBatch(self, batch_size): #用于从缓冲区中采样出一个批次数据，大小为 batch_size。
        # random draw N，随机抽取 N 个样本
        if self.rand_s:
            return random.sample(self.buffer, batch_size), None, None
        batch, w, e_id = self.replay_memory.sample(self.num_experiences)  # 从rank_based.Experience中采样
        # batch：采样得到的经验批次。
        # w：采样时得到的权重（如果使用优先级经验回放的话）
        # e_id：采样经验的索引ID列表或其他标识信息。
        self.e_id = e_id  # 存储采样的经验ID
        self.w_id = w  # 存储采样的权重
        return batch, self.w_id, self.e_id


    def add(self, state, action, reward, next_state, done):
        new_experience = (state, action, reward, next_state, done)
        self.num_experiences += 1 # 经验数量加一
        if self.rand_s:
            if self.num_experiences < self.buffer_size:
                self.buffer.append(new_experience)  # 如果缓冲区未满，添加经验
            else:
                self.buffer.popleft()  # 如果缓冲区已满，删除最早的经验
                self.buffer.append(new_experience) # 添加新的经验
        else:
            self.replay_memory.store(new_experience)  # 使用rank_based.Experience存储经验


    def update_priority(self, indices, delta):
        self.replay_memory.update_priority(indices, delta) # 更新rank_based.Experience中经验的优先级
