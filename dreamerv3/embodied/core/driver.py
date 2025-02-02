import collections

import numpy as np

from .basics import convert

import imageio

class Driver:

  _CONVERSION = {
      np.floating: np.float32,
      np.signedinteger: np.int32,
      np.uint8: np.uint8,
      bool: bool,
  }

  def __init__(self, env, mode, **kwargs):
    assert len(env) > 0
    self._env = env
    self._mode = mode
    self._kwargs = kwargs
    self._on_steps = []
    self._on_episodes = []
    self._ep_cnt = -1
    self._video_name_index = 0
    self.reset()

  def reset(self):
    self._acts = {
        k: convert(np.zeros((len(self._env),) + v.shape, v.dtype))
        for k, v in self._env.act_space.items()}
    self._acts['reset'] = np.ones(len(self._env), bool)
    self._eps = [collections.defaultdict(list) for _ in range(len(self._env))]
    self._state = None

  def on_step(self, callback):
    self._on_steps.append(callback)

  def on_episode(self, callback):
    self._on_episodes.append(callback)

  def __call__(self, policy, steps=0, episodes=0):
    step, episode = 0, 0
    while step < steps or episode < episodes:
      step, episode = self._step(policy, step, episode)

  def _step(self, policy, step, episode):
    assert all(len(x) == len(self._env) for x in self._acts.values())
    acts = {k: v for k, v in self._acts.items() if not k.startswith('log_')}
    obs = self._env.step(acts)
    
    # if self._mode == "eval":
    #   if self._ep_cnt == -1:
    #     self._writer = imageio.get_writer(f"./dreamerv3/video{self._video_name_index}.mp4", fps=20)
    #     self._ep_cnt = 0
        
    #   frame = np.squeeze(obs["agentview_image"])
    #   self._writer.append_data(frame)
    #   # print("frame saved")
    #   if obs['is_last'].any():
    #     self._writer.append_data(frame)
    #     self._writer.close()
    #     self._video_name_index += 1
    #     self._writer = imageio.get_writer(f"./dreamerv3/video{self._video_name_index}.mp4", fps=20)
    #     print(f"video saved {self._video_name_index}")
    
    obs = {k: convert(v) for k, v in obs.items()}
    assert all(len(x) == len(self._env) for x in obs.values()), obs
    acts, self._state = policy(obs, self._state, **self._kwargs)
    acts = {k: convert(v) for k, v in acts.items()}
    if obs['is_last'].any():
      mask = 1 - obs['is_last']
      acts = {k: v * self._expand(mask, len(v.shape)) for k, v in acts.items()}
    acts['reset'] = obs['is_last'].copy()
    self._acts = acts
    trns = {**obs, **acts}
    if obs['is_first'].any():
      for i, first in enumerate(obs['is_first']):
        if first:
          self._eps[i].clear()
    for i in range(len(self._env)):
      trn = {k: v[i] for k, v in trns.items()}
      [self._eps[i][k].append(v) for k, v in trn.items()]
      [fn(trn, i, **self._kwargs) for fn in self._on_steps]
      step += 1
    if obs['is_last'].any():
      for i, done in enumerate(obs['is_last']):
        if done:
          ep = {k: convert(v) for k, v in self._eps[i].items()}
          [fn(ep.copy(), i, **self._kwargs) for fn in self._on_episodes]
          episode += 1
    return step, episode

  def _expand(self, value, dims):
    while len(value.shape) < dims:
      value = value[..., None]
    return value
