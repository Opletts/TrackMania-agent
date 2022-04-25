import tmrl.config.config_constants as cfg  # constants defined in config.json
import tmrl.config.config_objects as cfg_obj  # higher-level constants
from tmrl.util import partial  # utility to partially instantiate objects

from tmrl.networking import Trainer  # main Trainer object
from tmrl.training_offline import TrainingOffline  # this is what we will customize

# =====================================================================
# USEFUL PARAMETERS
# =====================================================================
# You can change these parameters here directly,
# or you can change them in the config.json file.

# maximum number of training 'epochs':
# (you should set this to numpy.inf)
# training is checkpointed at the end of each 'epoch'
# this is also when training metrics can be logged to wandb
epochs = cfg.TMRL_CONFIG["MAX_EPOCHS"]

# number of rounds per 'epoch':
# training metrics are displayed at the end of each round
rounds = cfg.TMRL_CONFIG["ROUNDS_PER_EPOCH"]

# number of training steps per round:
steps = cfg.TMRL_CONFIG["TRAINING_STEPS_PER_ROUND"]

# minimum number of environment steps collected before training starts:
start_training = cfg.TMRL_CONFIG["ENVIRONMENT_STEPS_BEFORE_TRAINING"]

# maximum training steps / env steps ratio:
max_training_steps_per_env_step = cfg.TMRL_CONFIG["MAX_TRAINING_STEPS_PER_ENVIRONMENT_STEP"]

# number of training steps between broadcasting policy updates:
update_model_interval = cfg.TMRL_CONFIG["UPDATE_MODEL_INTERVAL"]

# number of training steps between retrieving replay memory updates:
update_buffer_interval = cfg.TMRL_CONFIG["UPDATE_BUFFER_INTERVAL"]

# training device (e.g., "cuda:0"):
# if None, the device will be selected automatically
device = None

# maximum size of the replay buffer:
memory_size = cfg.TMRL_CONFIG["MEMORY_SIZE"]

# batch size:
batch_size = cfg.TMRL_CONFIG["BATCH_SIZE"]


# =====================================================================
# ADVANCED PARAMETERS
# =====================================================================

# base class of the replay memory:
memory_base_cls = cfg_obj.MEM

# sample preprocessor for data augmentation:
sample_preprocessor = cfg_obj.SAMPLE_PREPROCESSOR

# path from where an offline dataset can be loaded:
dataset_path = cfg.DATASET_PATH


# rtgym environment class:
env_cls = cfg_obj.ENV_CLS

# observation preprocessor:
obs_preprocessor = cfg_obj.OBS_PREPROCESSOR

# number of LIDARs per observation:
imgs_obs = cfg.IMG_HIST_LEN

# number of actions in the action buffer:
act_buf_len = cfg.ACT_BUF_LEN


# =====================================================================
# MEMORY CLASS
# =====================================================================

memory_cls = partial(memory_base_cls,
                     memory_size=memory_size,
                     batch_size=batch_size,
                     obs_preprocessor=obs_preprocessor,
                     sample_preprocessor=sample_preprocessor,
                     dataset_path=cfg.DATASET_PATH,
                     imgs_obs=imgs_obs,
                     act_buf_len=act_buf_len,
                     crc_debug=False,
                     use_dataloader=False,
                     pin_memory=False)


# =====================================================================
# CUSTOM TRAINER
# =====================================================================

import torch
from ddpg_agent import SpinupDDPGAgent

training_agent_cls = partial(SpinupDDPGAgent)

training_cls = partial(TrainingOffline,
                       training_agent_cls=training_agent_cls,
                       epochs=epochs,
                       rounds=rounds,
                       steps=steps,
                       update_buffer_interval=update_buffer_interval,
                       update_model_interval=update_model_interval,
                       max_training_steps_per_env_step=max_training_steps_per_env_step,
                       start_training=start_training,
                       device=device,
                       env_cls=env_cls,
                       memory_cls=memory_cls)

my_trainer = Trainer(training_cls=training_cls)

my_trainer.run()
