from __future__ import print_function
from keras.api.layers import *
from keras.api.models import Model
from keras.api.regularizers import l2
from keras.api.optimizers import Adam
import keras.api.backend as K
from keras.api.models import load_model
from keras.api.models import Sequential

import numpy as np
import pickle


class PolicyValueNet():
    """policy-value network """

    def __init__(self, board_width, board_height, model_file=None):
        self.board_width = board_width
        self.board_height = board_height
        self.l2_const = 1e-4  # coef of l2 penalty
        self.create_policy_value_net(model_file)
        self._loss_train_op()

    def create_policy_value_net(self,model_file):
        self.model = load_model(model_file)
        def policy_value(state_input):
            state_input_union = np.array(state_input)
            results = self.model.predict_on_batch(state_input_union)
            return results
        self.policy_value = policy_value

    def policy_value_fn(self, board):
        legal_positions = board.availables
        current_state = board.current_state()
        act_probs, value = self.policy_value(
            current_state.reshape(-1, 5, self.board_width, self.board_height))
        
        act_probs = zip(legal_positions, act_probs.flatten()[legal_positions])
        # print(list(act_probs))
        # act_probs.flatten()[legal_positions] takes only the probs for legel_positions, others are zero
        return act_probs, value[0][0]

    def _loss_train_op(self):
        opt = Adam()
        losses = ['categorical_crossentropy', 'mean_squared_error']
        self.model.compile(optimizer=opt, loss=losses)

        def self_entropy(probs):
            return -np.mean(np.sum(probs * np.log(probs + 1e-10), axis=1))

        def train_step(state_input, mcts_probs, winner, learning_rate):
            state_input_union = np.array(state_input)
            mcts_probs_union = np.array(mcts_probs)
            winner_union = np.array(winner)
            loss = self.model.evaluate(state_input_union, [
                                       mcts_probs_union, winner_union], batch_size=len(state_input), verbose=0)
            action_probs, _ = self.model.predict_on_batch(state_input_union)
            entropy = self_entropy(action_probs)
            K.set_value(self.model.optimizer.lr, learning_rate)
            self.model.fit(state_input_union, [
                           mcts_probs_union, winner_union], batch_size=len(state_input), verbose=0)
            return loss[0], entropy

        self.train_step = train_step

    def save_model(self, model_file):
        self.model.save(model_file, save_format='h5')
        
        