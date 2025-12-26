# 模型训练的主代码
import numpy as np
import tensorflow as tf
import os
import scipy.io as scio
import argparse
import cv2
from shutil import copyfile
import matplotlib.pyplot as plt
from traffic_interaction_scene import TrafficInteraction #用来创建交通仿真环境
import time
from model_agent_maddpg import GAT_MADDPG # MADDPG才是自己创建的模型
from replay_buffer import ReplayBuffer # 自己写的
import pandas as pd

import io
from PIL import Image
from scipy.ndimage import gaussian_filter1d


class DummySummaryWriter:
    def add_summary(self, *args, **kwargs):
        pass

# create_init_update函数用于初始化和更新目标网络的参数，tau控制更新的速率。

def create_init_update(actor_scope, target_actor_scope, tau):
    """
    创建初始化和更新操作，用于同步目标网络的参数。
    :param actor_scope: 在线网络的scope
    :param target_actor_scope: 目标网络的scope
    :param tau: 软更新的速率
    :return: 初始化操作和更新操作
    """
    # 获取在线和目标网络的变量
    actor_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=actor_scope)
    target_actor_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=target_actor_scope)

    # 初始化目标网络操作，将在线网络的权重复制到目标网络
    init_ops = []
    for var, target_var in zip(actor_vars, target_actor_vars):
        init_ops.append(target_var.assign(var))

    # 软更新目标网络操作
    update_ops = []
    for var, target_var in zip(actor_vars, target_actor_vars):
        update_ops.append(target_var.assign(
            target_var * (1 - tau) + var * tau))  # 目标网络 = (1 - tau) * 目标网络 + tau * 在线网络

    return init_ops, update_ops

    return target_init, target_update

#get_agents_action函数根据状态和模型，使用噪声生成动作。

#获取智能体的动作函数
def get_agents_action(sta, sess, agent, noise_range=0.2):
    """
    :param sta: the state of the agent
    :param sess: the session of tf
    :param agent: the model of the agent
    :param noise_range: the noise range added to the agent model output
    :return: the action of the agent in its current state
    """
    agent1_action = agent.action(state=[sta], sess=sess) + np.random.randn(1) * noise_range
    return agent1_action

#train_agent_seq函数从经验池中采样数据，然后训练critic和actor模型，并更新目标网络。
#训练智能体的函数，使用深度确定性策略梯度（DDPG）算法训练智能体
def train_agent_seq(agent_ddpg, agent_ddpg_target, agent_memory, agent_actor_target_update,
                    agent_critic_target_update, sess, summary_writer, epoch, args):
    dummy_writer = DummySummaryWriter()

    # 获取批量数据
    batch, w_id, eid = agent_memory.getBatch(args.batch_size)
    if not batch:
        return

    agent_num = args.o_agent_num + 1  # 智能体总数，包括当前智能体和其他智能体
    total_obs_batch = np.zeros((args.batch_size, agent_num, agent_num * 4))  # 当前状态
    rew_batch = np.zeros((args.batch_size,))  # 奖励
    total_act_batch = np.zeros((args.batch_size, agent_num))  # 动作
    total_next_obs_batch = np.zeros((args.batch_size, agent_num, agent_num * 4))  # 下一状态
    next_state_mask = np.zeros((args.batch_size,))

    # 填充经验数据到对应的数组
    for k, (s0, a, r, s1, done) in enumerate(batch):  # s0 当前状态，a 动作，r 奖励，s1 下一状态，done 是否结束
        total_obs_batch[k] = s0
        rew_batch[k] = r
        total_act_batch[k] = a
        if not done:
            total_next_obs_batch[k] = s1
            next_state_mask[k] = 1

    # 计算渐进式折扣因子（γ）
    gamma = np.tanh((epoch + 6) / 12.0) * 0.8  # 你可以根据具体的算法调整此公式

    # 计算目标Q值并使用GAE进行回报计算
    e_id = eid
    obs_batch = total_obs_batch[:, 0, :]  # 获取本agent当前状态集
    act_batch = np.array(total_act_batch[:, 0])  # 获取本agent的动作集
    act_batch = act_batch.reshape(-1, 1)  # 这里确保 act_batch 的形状为 (batch_size, 1)

    # 提取其他智能体的动作
    other_act = []
    for n in range(1, agent_num):
        other_act.append(total_act_batch[:, n])
    other_act_batch = np.vstack(other_act).transpose()

    # 计算TD误差（target与critic估计的Q值之间的差异）
    target = rew_batch.reshape(-1, 1)  # 将奖励调整为目标Q值，这里也需要确保 target 是 (batch_size, 1)
    td_error = abs(agent_ddpg_target.Q(
        state=obs_batch, action=act_batch, other_action=other_act_batch, sess=sess) - target)

    if e_id is not None:
        agent_memory.update_priority(e_id, td_error)  # 更新优先级

    # 训练过程中的GAE计算
    values = agent_ddpg.Q(state=obs_batch, action=act_batch, other_action=other_act_batch, sess=sess)  # 获取当前值函数
    values = values.reshape(-1)  # 确保values是一维数组

    # 计算GAE
    gae = compute_gae(target.reshape(-1), values, rew_batch, gamma, lambda_=0.95)  # target也需要转化为一维数组
    target = gae.reshape(-1, 1)  # 更新目标值为GAE，并确保目标形状为 (batch_size, 1)

    # 训练Critic网络
    agent_ddpg.train_critic(state=obs_batch, action=act_batch, other_action=other_act_batch, target=target,
                            sess=sess, summary_writer=dummy_writer, lr=args.critic_lr)

    # 训练Actor网络
    agent_ddpg.train_actor(state=obs_batch, other_action=other_act_batch, sess=sess, summary_writer=dummy_writer,
                           lr=args.actor_lr)

    # 更新目标网络
    sess.run([agent_actor_target_update, agent_critic_target_update])  # 从online模型更新到target模型



#parse_args函数用于解析命令行参数，这些参数控制实验的各种设置。
# 解析命令行参数函数
def parse_args():
    parser = argparse.ArgumentParser("GAT_MADDPG experiments for multiagent traffic interaction environments")
    #添加参数：num_episodes
    parser.add_argument("--num_episodes", type=int, default=1, help="number of episodes")  # episode次数
    parser.add_argument("--o_agent_num", type=int, default=4, help="other agent numbers")
    parser.add_argument("--seq_max_step", type=int, default=12, help="the step of multi-step learning")

    parser.add_argument("--actor_lr", type=float, default=1e-4, help="learning rate for Adam optimizer")  # 动作的学习率
    parser.add_argument("--critic_lr", type=float, default=1e-4, help="learning rate for Adam optimizer")  # 批评的学习率
    parser.add_argument("--gamma", type=float, default=0.80, help="discount factor")  # 折扣率
    parser.add_argument("--trans_r", type=float, default=0.998, help="transfer rate for online model to target model")
    parser.add_argument("--batch_size", type=int, default=128,
                        help="number of episodes to optimize at the same time")  # 经验采样数目
    parser.add_argument("--learn_start", type=int, default=20000,
                        help="learn start step")  # 经验采样数目
    parser.add_argument("--lane_num", type=int, default=8,
                        help="the num of lane of intersection")  # 车道总数，12表示双向六车道交叉口
    parser.add_argument("--num_units", type=int, default=64, help="number of units in the mlp")
    parser.add_argument("--collision_thr", type=float, default=1, help="the threshold for collision") #2
    parser.add_argument("--actual_lane", action="store_true", default=False, help="")
    parser.add_argument("--c_mode", type=str, default="closer",
                        help="the way of choosing closer cars, front ,front-end or closer")

    parser.add_argument("--model", type=str, default="GAT_MADDPG",
                        help="the model for training, GAT_MADDPG or DDPG")

    parser.add_argument("--exp_name", type=str, default="test ", help="name of the experiment")  # 实验名
    parser.add_argument("--type", type=str, default="test", help="type of experiment train or test")
    parser.add_argument("--mat_path", type=str, default="./data/train/arvTimeNewVeh_for_train.mat",
                        help="the path of mat file")
    parser.add_argument("--save_dir", type=str, default="model_data",
                        help="directory in which training state and model should be saved")  # 模型存储
    parser.add_argument("--save_rate", type=int, default=1,
                        help="save model once every time this many episodes are completed")  # 存储模型的回合间隔
    parser.add_argument("--load_dir", type=str, default="",
                        help="directory in which training state and model are loaded")  # 模型加载目录
    parser.add_argument("--video_name", type=str, default="",
                        help="if it not empty, program will generate a result video (.mp4 format defaultly)with the result imgs")
    parser.add_argument("--visible", action="store_true", default=False, help="visible or not")
    # Evaluation
    parser.add_argument("--restore", action="store_true", default=False)  # 恢复之前的模型，在 load-dir 或 save-dir
    parser.add_argument("--benchmark", action="store_true", default=False)  # 用保存的模型跑测试
    parser.add_argument("--batch_test", action="store_true", default=False)  # 是否批量测试
    parser.add_argument("--benchmark_iters", type=int, default=1000, help="number of iterations run for benchmarking")
    parser.add_argument("--benchmark-dir", type=str, default="./benchmark_files/",
                        help="directory where benchmark data is saved")
    parser.add_argument("--plots-dir", type=str, default="./learning_curves/",
                        help="directory where plot data is saved")  # 训练曲线的目录

    # 添加图注意力网络的注意力头数
    parser.add_argument('--num_attention_heads', type=int, default=4)  # 默认4个注意力头

    return parser.parse_args()


#基准测试函数
#用于评估训练好的模型在不同交通场景中的性能，主要通过模拟车辆的行驶情况并计算碰撞率。
def benchmark(model, arrive_time, sess): #model：用于测试的训练好的模型，arrive_time：车辆到达时间的数据，sess：TensorFlow 会话，用于执行模型的预测。
    total_c = 0 #用于记录处理的总车辆数量。
    collisions_count = 0 #用于记录碰撞发生的次数。
    for mat_file in ["arvTimeNewVeh_300.mat", "arvTimeNewVeh_600.mat", "arvTimeNewVeh_900.mat"]: #循环加载三个不同的 .mat 文件
        data = scio.loadmat(mat_file)  # 使用scipy.io 加载.mat 文件中的数据
        arrive_time = data["arvTimeNewVeh"]#从加载的.mat文件中提取车辆到达时间数据，用于初始化交通环境。
        env = TrafficInteraction(arrive_time, 150, args, vm=6, virtual_l=not args.actual_lane)#vm=6表示车辆的速度，virtual_l=not args.actual_lane 表示是否使用虚拟车道模式。
        # env = TrafficInteraction(arrive_time, 150, args, vm=6, vM=20, v0=12)
        for i in range(args.benchmark_iters):
            for lane in range(4):
                for ind, veh in enumerate(env.veh_info[lane]):
                    o_n = veh["state"] #获取车辆的状态。
                    agent1_action = [[0]] #初始化动作为0
                    if veh["control"]:#判断车辆是否可控
                        agent1_action = get_agents_action(o_n[0], sess, model, noise_range=0)  # 模型根据当前状态进行预测
                    #执行动作并更新环境
                    env.step(lane, ind, agent1_action[0][0])  # 环境根据输入的动作返回下一时刻的状态和奖励
            #更新场景状态并记录碰撞
            state_next, reward, actions, collisions, estm_collisions, collisions_per_veh = env.scene_update() #更新场景，返回下一状态、奖励、动作、碰撞信息等。
            for k in range(len(actions)):
                if collisions_per_veh[k][0] > 0:
                    collisions_count += 1
            if i % 1000 == 0:
                print("i: %s collisions_rate: %s" % (i, float(collisions_count) / (env.id_seq + total_c)))
            env.delete_vehicle()
        total_c += env.id_seq
        print("vehicle number: %s; collisions occurred number: %s; collisions rate: %s" % (
            total_c, collisions_count, float(collisions_count) / total_c))
    return float(collisions_count) / total_c


#特征重要性分析，函数定义和图像初始化
def actor_feature_importance_analyze(state, model, sess, idx=0):
    plt.figure(0)
    imps = np.zeros(state.shape[0])
    base = get_agents_action(state, sess, model)[0] #调用get_agents_action函数，基于当前状态state获取智能体的基准动作（不扰动特征时的动作）。[0]表示提取第一维的动作值。
    for j in range(imps.shape[0]):
        fes = [] #用于存储扰动后的状态
        for i in range(100):
            tmp = state.copy()
            tmp[j] += np.random.rand(1) * 10
            fes.append(tmp)
        #计算动作的变化并求平均绝对变化值
        imps[j] = np.mean(abs((model.action(state=fes, sess=sess).reshape(100) - base[0])))
    if sum(imps) > 1:
        print(state, imps)
    plt.bar([i for i in range(len(imps))], imps)
    plt.savefig("result_img/feature_importance_curve_%s.png" % idx)
    plt.close()


# 特征重要性分析工具
#测试函数
def test():
    # 1. 创建测试用的MADDPG模型
    agent1_ddpg_test = GAT_MADDPG('agent1', actor_lr=args.actor_lr, critic_lr=args.critic_lr,
                             nb_other_action=args.o_agent_num,  # 更新此处为 nb_other_action
                             num_units=args.num_units, num_attention_heads=args.num_attention_heads, model=args.model)


    # 2. 创建TensorFlow Saver
    saver = tf.train.Saver()

    # 3. 配置会话
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())

    # 5. 从save_dir/exp_name中恢复模型：如果指定文件不存在，就使用latest_checkpoint加载最近的模型
    model_path = os.path.join(args.save_dir, args.exp_name, "best.cptk")
    if not os.path.exists(model_path + ".meta"):
        model_path = tf.train.latest_checkpoint(os.path.join(args.save_dir, args.exp_name))
    saver.restore(sess, model_path)
    print("load cptk file from " + model_path)

    # 5. 读取测试用 mat 文件
    mat_path = os.path.join("./data/test", args.mat_path)
    data = scio.loadmat(mat_path)  # 加载.mat数据
    arrive_time = data["arvTimeNewVeh"]
    print('Load test data from mat_path:', mat_path)

    # 6. 初始化环境, 去掉所有可视化/窗口显示相关的逻辑
    env = TrafficInteraction(arrive_time, 150, args,
                             show_col=False,  # 不渲染图像
                             virtual_l=not args.actual_lane,
                             lane_num=args.lane_num)

    # 7. 定义统计列表(每步)；若测试中只跑一次episode，这样写就够了
    reward_list_allsteps = []
    collisions_list_allsteps = []
    jerk_list_allsteps = []

    collisions_count = 0  # 总碰撞次数
    lock_total = 0  # 不知道是否要统计，保留逻辑
    time_total = []  # 存储动作推断时长
    jerk_total = 0

    # 8. 执行 1000 步测试
    for i in range(2000):
        for lane in range(args.lane_num):
            for ind, veh in enumerate(env.veh_info[lane]):
                o_n = veh["state"]
                agent1_action = [[0]]
                if veh["control"]:
                    temp_t = time.time()
                    # 不加可视化 => 仅推断
                    agent1_action = get_agents_action(o_n[0], sess, agent1_ddpg_test, noise_range=0)
                    time_total.append(time.time() - temp_t)
                env.step(lane, ind, agent1_action[0][0])

        # 环境更新, 获取各种信息
        ids, state_next, reward, actions, collisions, estm_collisions, collisions_per_veh, jerks, lock = env.scene_update()

        jerk_total += sum(jerks)
        lock_total += lock

        # ===== 收集统计量 =====
        # 1) reward_list_allsteps
        reward_list_allsteps += reward
        # 2) jerk_list_allsteps
        jerk_list_allsteps += jerks
        # 3) collisions_list_allsteps
        #   collisions_per_veh[k][0] > 0 表示是否碰撞
        for k in range(len(actions)):
            if collisions_per_veh[k][0] > 0:
                collisions_count += 1
                # 对每辆车做 0/1 标记
                collisions_list_allsteps.append(1)
            else:
                collisions_list_allsteps.append(0)

        # 每隔 100 步打印一次信息
        if i % 100 == 0:
            print("Step: %d collisions_rate: %.4f reward std: %.4f reward mean: %.4f lock_num: %d" %
                  (i,
                   float(collisions_count) / env.id_seq,
                   np.std(reward),
                   np.mean(reward),
                   lock_total))

        # 不做任何可视化图像窗口 => 删除 visible.show()、cv2.imshow()、video_writer 等
        env.delete_vehicle()

    # 9. 测试结束, 打印整体统计信息
    print("======== Test Summary ========")
    print("vehicle number:", env.id_seq)
    print("collisions occurred number:", collisions_count)
    print("collisions rate:", float(collisions_count) / env.id_seq if env.id_seq > 0 else 0)
    print("time_mean:", np.mean(time_total) if len(time_total) > 0 else 0)
    print("pT-m: %.4f s" % (float(env.passed_veh_step_total) / (env.passed_veh + 0.0001) * env.deltaT))
    print("jerks: %.4f" % (jerk_total / env.passed_veh if env.passed_veh > 0 else 0))

    sess.close()

    # =========================
    # 10. 绘制三张图(仅显示,不保存)
    #    reward_list_allsteps / collisions_list_allsteps / jerk_list_allsteps
    # =========================

    # (a) Reward
    plt.figure()
    plt.title("Reward (All Steps)")
    plt.plot(reward_list_allsteps, color='blue', label="Reward")
    plt.legend()
    plt.show()
    plt.close()

    # (b) Collisions
    plt.figure()
    plt.title("Collisions (All Steps) [0 or 1 per step/veh]")
    plt.plot(collisions_list_allsteps, color='red', label="Collisions")
    plt.legend()
    plt.show()
    plt.close()

    # (c) Jerk
    plt.figure()
    plt.title("Jerk (All Steps)")
    plt.plot(jerk_list_allsteps, color='green', label="Jerk")
    plt.legend()
    plt.show()
    plt.close()


#批量测试函数
def batch_test():
    agent1_ddpg_test = GAT_MADDPG('agent1', actor_lr=args.actor_lr, critic_lr=args.critic_lr,
                             nb_other_action=args.o_agent_num,  # 更新此处为 nb_other_action
                             num_units=args.num_units, num_attention_heads=args.num_attention_heads, model=args.model)

    saver = tf.train.Saver()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    #config = tf.ConfigProto(device_count={'GPU': 0})
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())

    model_path = os.path.join(args.save_dir, args.exp_name, "best.cptk")
    if not os.path.exists(model_path + ".meta"):
        model_path = tf.train.latest_checkpoint(os.path.join(args.save_dir, args.exp_name))
    saver.restore(sess, model_path)
    print("load cptk file from " + model_path)
    dens = [1200, 1000, 900, 800, 600, 400, 200]
    tw = open(args.exp_name + "_batch_test_result_12_v1_best.txt", "w")
    for d in dens:
        dens_f = "arvTimeNewVeh_new_%s_%s.mat" % (d, args.lane_num)
        mat_path = os.path.join("./data/test", dens_f)
        print(mat_path)
        tw.write(mat_path + "\n")
        data = scio.loadmat(mat_path)  # 加载.mat数据
        arrive_time = data["arvTimeNewVeh"]
        env = TrafficInteraction(arrive_time, 150, args, show_col=False, virtual_l=not args.actual_lane,
                                 lane_num=args.lane_num)
        jerk_total = 0
        collisions_count = 0
        lock_total = 0
        for i in range(20000):
            for lane in range(args.lane_num):
                for ind, veh in enumerate(env.veh_info[lane]):
                    o_n = veh["state"]
                    agent1_action = [[0]]
                    if veh["control"]:
                        agent1_action = get_agents_action(o_n[0], sess, agent1_ddpg_test,
                                                          noise_range=0)  # 模型根据当前状态进行预测
                    env.step(lane, ind, agent1_action[0][0])  # 环境根据输入的动作返回下一时刻的状态和奖励
            ids, state_next, reward, actions, collisions, estm_collisions, collisions_per_veh, jerks, lock = env.scene_update()
            jerk_total += sum(jerks)
            lock_total += lock
            for k in range(len(actions)):
                if collisions_per_veh[k][0] > 0:
                    collisions_count += 1
            if i % 1000 == 0:
                print("i: %s collisions_rate: %s reward std: %s reward mean: %s lock_num: %s" % (
                    i, float(collisions_count) / env.id_seq, np.std(reward), np.mean(reward), lock_total))
            env.delete_vehicle()
        result_txt = "vehicle number %s  collisions occurred number %s collisions rate %s pT-m %0.4f s jerks %s " \
                     "lock_num %s" % (
                         env.id_seq, collisions_count, float(collisions_count) / env.id_seq,
                         float(env.passed_veh_step_total) / (env.passed_veh + 0.0001) * env.deltaT,
                         jerk_total / env.passed_veh,
                         lock_total)
        print(result_txt)
        tw.write(result_txt + "\n")
    tw.close()
    sess.close()


def train():
    # 初始化agent1的GAT_MADDPG网络与目标网络
    agent1_ddpg = GAT_MADDPG('agent1', actor_lr=args.actor_lr, critic_lr=args.critic_lr,
                             nb_other_action=args.o_agent_num,  # 更新此处为 nb_other_action
                             num_units=args.num_units, num_attention_heads=args.num_attention_heads, model=args.model)

    agent1_ddpg_target = GAT_MADDPG('agent1_target', actor_lr=args.actor_lr, critic_lr=args.critic_lr,
                                    nb_other_action=args.o_agent_num,  # 更新此处为 nb_other_action
                                    num_units=args.num_units, num_attention_heads=args.num_attention_heads, model=args.model)

    # 2. 创建 Saver，用于保存模型
    saver = tf.train.Saver()

    # 3. 创建从在线网络同步到目标网络的操作
    agent1_actor_target_init, agent1_actor_target_update = create_init_update(
        'agent1actor', 'agent1_targetactor', tau=args.trans_r
    )
    agent1_critic_target_init, agent1_critic_target_update = create_init_update(
        'agent1_critic', 'agent1_target_critic', tau=args.trans_r
    )

    # 4. 配置 Session
    config = tf.ConfigProto(device_count={'GPU': 0})  # 若想用GPU可注释掉
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.5
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())
    sess.run([agent1_actor_target_init, agent1_critic_target_init])

    # 如果指定了恢复模型
    if args.restore:
        cptk_path = tf.train.latest_checkpoint(os.path.join(args.save_dir, args.exp_name))
        saver.restore(sess, cptk_path)
        print("load cptk file from", cptk_path)

    # 5. 创建经验回放池
    agent1_memory_seq = ReplayBuffer(500000, args.batch_size, args.learn_start, 50000, rand_s=True)

    # 6. 定义一些统计变量（Python列表）
    #    用于保存训练过程中的关键指标，后面画图用
    reward_list_allsteps = []       # 所有步的 reward
    collisions_list_allsteps = []   # 所有步的碰撞(0或1)
    estm_collisions_list_allsteps = []
    jerk_list_allsteps = []         # 记录所有的 jerk
    collisions_count = 0            # 累积碰撞次数

    # 如果想记录“每个 epoch 的平均奖励、碰撞率”等，可以多定义一些列表
    epoch_avg_reward = []
    epoch_collisions_rate = []
    epoch_jerk_mean = []

    # 其他辅助变量
    statistic_count = 0
    mean_window_length = 50
    state_now = []
    rate_latest = 1.0
    test_rate_latest = 1.0
    time_total = []
    seq_max_step = args.seq_max_step
    count_n = 0  # 用于统计训练步数

    # 创建一个空的列表，用于记录每次的 collisions_count
    c_collisions = []
    # 新增：记录累计车辆数量（每一步）
    c_total_vehicles = []

    # 7. 主训练循环
    for epoch in range(args.num_episodes):
        collisions_count_last = collisions_count
        # 动态调整 gamma (可选)
        args.gamma = np.tanh(float(epoch + 6) / 12.0) * 0.5 #0.9

        # 加载训练数据 (示例：从 .mat 文件)
        data = scio.loadmat("./data/train/arvTimeNewVeh_for_train.mat")
        arrive_time = data["arvTimeNewVeh"]
        print('456')

        # 创建环境
        env = TrafficInteraction(arrive_time, 150, args, vm=6,
                                 virtual_l=not args.actual_lane, lane_num=args.lane_num)

        # 统计本 epoch 的reward、碰撞等，用于汇总后画图
        rewards_this_epoch = []
        collisions_this_epoch = 0
        jerk_this_epoch = []

        # 每个epoch内部跑 6000 步
        for i in range(1000):
            state_now.clear()
            # 遍历每个车道每辆车
            for lane in range(args.lane_num):
                for ind, veh in enumerate(env.veh_info[lane]):
                    o_n = veh["state"]
                    agent1_action = [[0]]
                    if veh["control"]:
                        count_n += 1
                        # 调用DDPG获取动作（加0.2噪声做探索）
                        agent1_action = get_agents_action(o_n[0], sess, agent1_ddpg, noise_range=0.2)
                        state_now.append(o_n)
                    # 执行动作
                    env.step(lane, ind, agent1_action[0][0])

            # 环境更新，获取新的观测和信息
            ids, state_next, reward, actions, collisions, estm_collisions, collisions_per_veh, jerks, lock = env.scene_update()

            # ====== 将多步轨迹写入 ReplayBuffer (n-step 逻辑) ======
            for seq, car_index in enumerate(ids):
                env.veh_info[car_index[0]][car_index[1]]["buffer"].append(
                    [state_now[seq], actions[seq], reward[seq], state_next[seq],
                     env.veh_info[car_index[0]][car_index[1]]["Done"]]
                )
                # 如果车完成 or 超过步长，就把它写进Memory
                if (env.veh_info[car_index[0]][car_index[1]]["Done"]
                        or env.veh_info[car_index[0]][car_index[1]]["count"] > seq_max_step):

                    seq_data = env.veh_info[car_index[0]][car_index[1]]["buffer"]
                    if env.veh_info[car_index[0]][car_index[1]]["Done"]:
                        r_target = seq_data[-1][2]
                    else:
                        other_act_next = []
                        for n in range(1, args.o_agent_num + 1):
                            other_act_next.append(
                                agent1_ddpg_target.action([seq_data[-1][3][n]], sess)[0][0]
                            )
                        # r + gamma * Q'(s', a')
                        r_target = seq_data[-1][2] + \
                                   args.gamma * agent1_ddpg_target.Q(
                            state=[seq_data[-1][3][0]],
                            action=agent1_ddpg_target.action([seq_data[-1][3][0]], sess),
                            other_action=[other_act_next],
                            sess=sess
                        )[0][0]

                    for cur_data in reversed(seq_data[:-1]):
                        r_target = cur_data[2] + args.gamma * r_target

                    agent1_memory_seq.add(
                        np.array(seq_data[0][0]),
                        np.array(seq_data[0][1]),
                        r_target,
                        np.array(seq_data[0][3]),
                        False
                    )
                    # 移除该车的 buffer 头
                    env.veh_info[car_index[0]][car_index[1]]["buffer"].pop(0)
                    env.veh_info[car_index[0]][car_index[1]]["count"] -= 1

            # ====== 累加一些统计量 ======
            # 1) reward
            reward_list_allsteps += reward
            rewards_this_epoch += list(reward)

            # 2) jerk
            jerk_list_allsteps += jerks
            jerk_this_epoch += list(jerks)

            # 3) collisions
            if len(collisions_per_veh) > 0:
                collisions_list_allsteps += list(np.array(collisions_per_veh)[:, 0])
                estm_collisions_list_allsteps += list(np.array(collisions_per_veh)[:, 1])

            # 计算碰撞总数
            for k in range(len(actions)):
                if collisions_per_veh[k][0] > 0:
                    collisions_count += 1
                    collisions_this_epoch += 1

            # 将每次的 collisions_count 记录到 collisions 列表中
            c_collisions.append(collisions_count)
            # ---- 新增：记录每步累计车辆数量 ----
            c_total_vehicles.append(env.id_seq)

            # 当采样(或训练步数)大于一定阈值才开始更新
            if count_n > 10000:
                statistic_count += 1
                time_t = time.time()
                # 更新在线网络 & 目标网络
                train_agent_seq(agent1_ddpg, agent1_ddpg_target, agent1_memory_seq,
                                agent1_actor_target_update, agent1_critic_target_update,
                                sess, None, i, args)  # 原来传 summary_writer，这里传None
                time_total.append(time.time() - time_t)

                # 打印部分信息
                if i % 100 == 0:
                    c_rate_now = (collisions_count - collisions_count_last) / float(env.id_seq)
                    print("epoch={}, i={}, count_n={}, collisions_count={}, c_rate_now={:.4f}, "
                          "reward_mean={:.3f}, jerk_mean={:.3f}"
                          .format(epoch, i, count_n, collisions_count, c_rate_now,
                                  np.mean(reward_list_allsteps[-mean_window_length:]),
                                  np.mean(jerk_list_allsteps[-mean_window_length:])
                                  ))

            # 删除已离开场景的车辆
            env.delete_vehicle()

        # ==== epoch结束后的统计 ====
        # 这里示范记录 epoch 的平均奖励、碰撞率、jerk 等
        epoch_avg_reward.append(np.mean(rewards_this_epoch) if len(rewards_this_epoch) > 0 else 0)
        c_rate_epoch = (collisions_count - collisions_count_last) / float(env.id_seq)
        epoch_collisions_rate.append(c_rate_epoch)
        epoch_jerk_mean.append(np.mean(jerk_this_epoch) if len(jerk_this_epoch) > 0 else 0)

        # ==== 保存模型 ====
        if epoch % args.save_rate == 0:
            cptk_path = os.path.join(args.save_dir, args.exp_name, f"{epoch}.cptk")
            print('update model to', cptk_path)
            saver.save(sess, cptk_path)

            # 根据碰撞率来判断是否更新 best.cptk
            if rate_latest > c_rate_epoch:
                rate_latest = c_rate_epoch
                # 拷贝为 best.cptk
                copyfile(cptk_path + ".data-00000-of-00001",
                         os.path.join(args.save_dir, args.exp_name, 'best.cptk.data-00000-of-00001'))
                copyfile(cptk_path + ".index",
                         os.path.join(args.save_dir, args.exp_name, 'best.cptk.index'))
                copyfile(cptk_path + ".meta",
                         os.path.join(args.save_dir, args.exp_name, 'best.cptk.meta'))

        # 每2个epoch跑一次benchmark测试(可选)
        if epoch % 2 == 0 and args.benchmark:
            c_rate = benchmark(agent1_ddpg, arrive_time, sess)
            if c_rate < test_rate_latest:
                test_rate_latest = c_rate
                # 更新 test_best.cptk
                copyfile(os.path.join(args.save_dir, args.exp_name, f"{epoch}.cptk.data-00000-of-00001"),
                         os.path.join(args.save_dir, args.exp_name, 'test_best.cptk.data-00000-of-00001'))
                copyfile(os.path.join(args.save_dir, args.exp_name, f"{epoch}.cptk.index"),
                         os.path.join(args.save_dir, args.exp_name, 'test_best.cptk.index'))
                copyfile(os.path.join(args.save_dir, args.exp_name, f"{epoch}.cptk.meta"),
                         os.path.join(args.save_dir, args.exp_name, 'test_best.cptk.meta'))

        # 每5个epoch衰减一次学习率
        if epoch % 5 == 4:
            args.actor_lr *= 0.9
            args.critic_lr *= 0.9

    # 8. 训练结束后，关闭会话
    sess.close()

    # # print('c_collisions:',c_collisions)
    # # 训练结束后保存 collisions 数据到 Excel 文件
    # file_path = r"C:\Users\Lenovo\Desktop\DRL statistic\collisions - update.xlsx"
    #
    # # 将 collisions 列表转换为 DataFrame 并保存到 Excel
    # df = pd.DataFrame(c_collisions, columns=["Collisions Count"])
    # df.to_excel(file_path, index=False)

    # A) 每步的数据汇总图（原来是三行 subplot）
#    现在拆成三张图，并仅 show，不保存

    # # 保存累计车辆数量
    # file_path_veh = r"C:\Users\Lenovo\Desktop\DRL statistic\vehicles_total.xlsx"
    # df_veh = pd.DataFrame(c_total_vehicles, columns=["Total Vehicles"])
    # df_veh.to_excel(file_path_veh, index=False)

    # 1) Reward (Smoothing with larger window and downsampling)
    plt.figure()
    plt.title("Reward (All Steps) - Smoothed")
    downsampled_reward = downsample_data(reward_list_allsteps, factor=50)
    smoothed_reward = smooth_loss_curve(downsampled_reward, window_size=200)  # Larger smoothing window

    # # Path to save the data
    # file_path = r"C:\Users\Lenovo\Desktop\DRL statistic\reward.xlsx"
    #
    # # Create a DataFrame and save it to Excel
    # df = pd.DataFrame(smoothed_reward, columns=["Smoothed Reward3"])
    # df.to_excel(file_path, index=False)

    plt.plot(smoothed_reward, color='blue', label="Smoothed Reward")
    plt.legend()
    plt.show()
    plt.close()

    # 2) Collisions (Smoothing with larger window and downsampling)
    plt.figure()
    plt.title("Collisions (All Steps) - Smoothed")
    downsampled_collisions = downsample_data(collisions_list_allsteps, factor=50)
    smoothed_collisions = smooth_loss_curve(downsampled_collisions, window_size=200)  # Larger smoothing window
    plt.plot(smoothed_collisions, color='red', label="Smoothed Collisions")
    plt.legend()
    plt.show()
    plt.close()

    # 3) Jerk (No changes)
    plt.figure()
    plt.title("Jerk (All Steps)")
    plt.plot(jerk_list_allsteps, color='green', label="Jerk")
    plt.legend()
    plt.show()
    plt.close()

    # B) 每个 epoch 的统计（保持不变）
    # 1) Average Reward per Epoch
    plt.figure()
    plt.title("Average Reward per Epoch")
    plt.plot(epoch_avg_reward, marker='o', label="AvgReward")
    plt.legend()
    plt.show()
    plt.close()

    # 2) Collision Rate per Epoch
    plt.figure()
    plt.title("Collision Rate per Epoch")
    plt.plot(epoch_collisions_rate, marker='s', color='red', label="CollisionRate")
    plt.legend()
    plt.show()
    plt.close()

    # 3) Mean Jerk per Epoch
    plt.figure()
    plt.title("Mean Jerk per Epoch")
    plt.plot(epoch_jerk_mean, marker='^', color='green', label="MeanJerk")
    plt.legend()
    plt.show()
    plt.close()

    # 获取损失值历史记录
    actor_loss_history, critic_loss_history = agent1_ddpg.get_loss_history()

    # Apply smoothing to the loss curves (you can adjust the window size)
    smoothed_actor_loss = smooth_loss_curve(actor_loss_history, window_size=100)
    smoothed_critic_loss = smooth_loss_curve(critic_loss_history, window_size=100)

    # 绘制Actor Loss曲线
    plt.figure()
    plt.plot(smoothed_actor_loss, label='Actor Loss', alpha=0.8)
    plt.title("Actor Loss Over Training Steps")
    plt.xlabel("Training Steps")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    # 设置横坐标每10000一个刻度
    plt.xticks(np.arange(0, len(smoothed_actor_loss), step=10000))
    plt.show()

    # 绘制Critic Loss曲线
    plt.figure()
    plt.plot(smoothed_critic_loss, label='Critic Loss', alpha=0.8)
    plt.title("Critic Loss Over Training Steps")
    plt.xlabel("Training Steps")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    # 设置横坐标每10000一个刻度
    plt.xticks(np.arange(0, len(smoothed_critic_loss), step=10000))
    plt.show()

def downsample_data(data, factor=100):
    """ Downsample the data by averaging over the specified factor """
    return [np.mean(data[i:i+factor]) for i in range(0, len(data), factor)]

def smooth_loss_curve(loss_history, window_size=200):
    """ Apply a simple moving average to smooth the loss curve """
    return np.convolve(loss_history, np.ones(window_size)/window_size, mode='valid')


# 泛化优势估计 (GAE)
def compute_gae(target_values, values, rewards, gamma, lambda_):
    gae = np.zeros_like(target_values)  # 初始化GAE
    # 确保values的切片操作不会导致长度不匹配
    values = np.concatenate([values, values[-1:]])  # 将values的最后一个值加到values末尾，保持长度一致
    delta = rewards + gamma * values[1:] - values[:-1]  # 计算每步的TD误差

    gae[-1] = delta[-1]  # 最后一个时间步的GAE就是delta值
    # 从倒数第二步开始，逐步计算每个时间步的GAE
    for t in range(len(target_values) - 2, -1, -1):
        gae[t] = delta[t] + gamma * lambda_ * gae[t + 1]  # GAE公式的递归计算

    return gae





if __name__ == '__main__':
    print('111')
    args = parse_args()
    print('11111')
    if not os.path.exists("result_imgs"):
        os.makedirs("result_imgs")
    if not os.path.exists("exp_result_imgs"):
        os.makedirs("exp_result_imgs")
    if not os.path.exists(os.path.join(args.save_dir, args.exp_name)):
        os.makedirs(os.path.join(args.save_dir, args.exp_name))
    print("args.type = ", args.type)
    if args.type == "train":
        with open(os.path.join(args.save_dir, args.exp_name, "args2.txt"), "w") as fw:
            fw.write(str(args))
        train()
    else:
        if args.batch_test:
            batch_test()
        else:
            test()
