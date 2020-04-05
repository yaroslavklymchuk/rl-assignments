import random
import numpy as np
from collections import deque


class ReplayBuffer(object):
    def __init__(self, capacity):
        """
        Arguments:
            capacity: Max number of elements in buffer
        """
        self._buffer = deque(maxlen=capacity)

    def push(self, s0, a, s1, r, d):
        """Push an element to the buffer.

        Arguments:
            s0: State before action
            a: Action picked by the agent
            s1: State after performing the action
            r: Reward recieved is state s1.
            d: Whether the episode terminated after in the state s1.

        If the buffer is full, start to rewrite elements
        starting from the oldest ones.
        """
        state = np.expand_dims(s0, 0)
        next_state = np.expand_dims(s1, 0)

        self._buffer.append((state, a, r, s1, d))

    def sample(self, batch_size):
        """Return `batch_size` randomly chosen elements."""
        state, action, reward, next_state, done = zip(*random.sample(self._buffer, batch_size))
        return np.concatenate(state), action, reward, np.concatenate(next_state), done

    def __len__(self):
        """Return size of the buffer."""
        return len(self._buffer)

