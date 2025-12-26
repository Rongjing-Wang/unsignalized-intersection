import tensorflow as tf
import tensorflow.contrib as tc
import numpy as np
import scipy.sparse as sp


class GAT_MADDPG():
    def __init__(self, name, actor_lr, critic_lr, layer_norm=True, nb_actions=1, nb_other_action=3,
                 num_units=128, num_attention_heads=4, model="GAT_MADDPG"):
        nb_input = 4 * (nb_actions + nb_other_action)  # State input size
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.layer_norm = layer_norm
        self.nb_actions = nb_actions

        self.actor_loss_history = []  # Store Actor Loss history
        self.critic_loss_history = []  # Store Critic Loss history

        state_input = tf.placeholder(shape=[None, nb_input], dtype=tf.float32)
        action_input = tf.placeholder(shape=[None, nb_actions], dtype=tf.float32)
        other_action_input = tf.placeholder(shape=[None, nb_other_action], dtype=tf.float32)

        reward = tf.placeholder(shape=[None, 1], dtype=tf.float32)

        # Graph attention network (GAT) layers
        def gat_layer(name, input, adj, num_units, num_heads):
            with tf.variable_scope(name):
                h = input
                for _ in range(num_heads):
                    h = attention(h, adj, num_units)
                return h

        def attention(h, adj, num_units):
            with tf.variable_scope("attention"):
                # Attention mechanism for each node in the graph
                attention_weights = tf.nn.softmax(tf.matmul(h, h, transpose_b=True))  # compute attention weights
                h_new = tf.matmul(attention_weights, h)  # Apply attention weights to neighbor nodes
                return h_new

        def actor_network(name, state_input):
            with tf.variable_scope(name) as scope:
                x = state_input
                if self.layer_norm:
                    x = tc.layers.layer_norm(x, center=True, scale=True)

                x = tf.layers.dense(x, num_units)
                residual = x  # Save the input for residual connection
                x = tf.nn.relu(x)

                x = tf.layers.dense(x, num_units)
                x = x + residual  # Add residual connection
                x = tf.nn.relu(x)

                x = tf.layers.dense(x, 1)
                w_ = tf.constant(3, dtype=tf.float32)
                x = tf.multiply(tf.nn.tanh(x), w_)
            return x

        def critic_network(name, state_input, action_input, reuse=False):
            with tf.variable_scope(name) as scope:
                if reuse:
                    scope.reuse_variables()
                x = state_input
                if self.layer_norm:
                    x = tc.layers.layer_norm(x, center=True, scale=True)
                x = tf.layers.dense(x, num_units,
                                    kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3))
                if self.layer_norm:
                    x = tc.layers.layer_norm(x, center=True, scale=True)
                x = tf.nn.relu(x)

                x = tf.concat([x, action_input], axis=-1)
                x = tf.layers.dense(x, num_units,
                                    kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3))
                if self.layer_norm:
                    x = tc.layers.layer_norm(x, center=True, scale=True)
                x = tf.nn.relu(x)

                x = tf.layers.dense(x, 1, kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3))
            return x

        self.state_input = state_input
        self.action_input = action_input
        self.other_action_input = other_action_input
        self.reward = reward

        # Apply Graph Attention Network (GAT) on the state input to model interactions between agents
        self.gat_output = gat_layer(name + "gat", state_input, adj=None, num_units=num_units,
                                    num_heads=num_attention_heads)

        # Actor network output
        self.action_output = actor_network(name + "actor", state_input=self.gat_output)

        # Critic network output
        self.critic_output = critic_network(name + '_critic',
                                            action_input=tf.concat([self.action_input, self.other_action_input],
                                                                   axis=1), state_input=self.gat_output)

        self.actor_optimizer = tf.train.AdamOptimizer(self.actor_lr)
        self.critic_optimizer = tf.train.AdamOptimizer(self.critic_lr)

        # Maximize Q value
        self.actor_loss = -tf.reduce_mean(
            critic_network(name + '_critic',
                           action_input=tf.concat([self.action_output, self.other_action_input], axis=1),
                           reuse=True, state_input=self.gat_output))  # reduce_mean is the expected value
        online_var = [i for i in tf.trainable_variables() if name + "actor" in i.name]
        self.actor_train = self.actor_optimizer.minimize(self.actor_loss, var_list=online_var)

        self.actor_loss_op = tf.summary.scalar("actor_loss", self.actor_loss)

        self.target_Q = tf.placeholder(shape=[None, 1], dtype=tf.float32)
        self.critic_loss = tf.reduce_mean(
            tf.square(self.target_Q - self.critic_output))  # MSE between target Q and real Q
        self.critic_loss_op = tf.summary.scalar("critic_loss", self.critic_loss)
        self.critic_train = self.critic_optimizer.minimize(self.critic_loss)
        self.count = 0

    def train_actor(self, state, other_action, sess, summary_writer, lr):
        self.count += 1
        self.actor_lr = lr
        actor_loss_val, _, summary = sess.run(
            [self.actor_loss, self.actor_train, self.actor_loss_op],
            {self.state_input: state, self.other_action_input: other_action}
        )
        self.actor_loss_history.append(actor_loss_val)  # 记录损失值
        summary_writer.add_summary(summary, self.count)

    def train_critic(self, state, action, other_action, target, sess, summary_writer, lr):
        self.critic_lr = lr
        critic_loss_val, _, summary = sess.run(
            [self.critic_loss, self.critic_train, self.critic_loss_op],
            {self.state_input: state,
             self.action_input: action,
             self.other_action_input: other_action,
             self.target_Q: target}
        )
        self.critic_loss_history.append(critic_loss_val)  # 记录损失值
        summary_writer.add_summary(summary, self.count)

    def action(self, state, sess):
        return sess.run(self.action_output, {self.state_input: state})

    def get_loss_history(self):
        """返回Actor和Critic的损失历史记录"""
        return self.actor_loss_history, self.critic_loss_history

    def Q(self, state, action, other_action, sess):
        return sess.run(self.critic_output,
                        {self.state_input: state, self.action_input: action, self.other_action_input: other_action})

