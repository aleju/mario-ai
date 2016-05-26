-- Keep emulator's memory library accessible,
-- we will use "memory" for the replay memory
lsne_memory = memory

----------------------------------
-- requires
----------------------------------
require 'torch'
require 'image'
require 'nn'
require 'optim'
memory = require 'memory'
network = require 'network'
actions = require 'actions'
Action = require 'action'
util = require 'util'
states = require 'states'
State = require 'state'
rewards = require 'rewards'
Reward = require 'reward'
ForgivingMSECriterion = require 'layers.ForgivingMSECriterion'
ForgivingAbsCriterion = require 'layers.ForgivingAbsCriterion'
ok, display = pcall(require, 'display')
if not ok then print('display not found. unable to plot') end

----------------------------------
-- RNG seed
----------------------------------
SEED = 43

----------------------------------
-- GPU / cudnn
----------------------------------
GPU = 0
require 'cutorch'
require 'cunn'
require 'cudnn'
if GPU >= 0 then
    print(string.format("Using gpu device %d", GPU))
    cutorch.setDevice(GPU + 1)
    cutorch.manualSeed(SEED)

    -- Saves 40% time according to http://torch.ch/blog/2016/02/04/resnets.html
    cudnn.fastest = true
    cudnn.benchmark = true
end
math.randomseed(SEED)
torch.manualSeed(SEED)
torch.setdefaulttensortype('torch.FloatTensor')
--------------------------------

----------------------------------
-- Other settings
----------------------------------
FPS = movie.get_game_info().fps
REACT_EVERY_NTH_FRAME = 5
print(string.format("FPS: %d, Reacting every %d frames", FPS, REACT_EVERY_NTH_FRAME))

-- filepath where current game's last screenshot will be saved
-- ideally on a ramdisk (for speed and less stress on the hard drive)
SCREENSHOT_FILEPATH = "/media/ramdisk/mario-ai-screenshots/current-screen.png"

IMG_DIMENSIONS = {1, 64, 64} -- screenshots will be resized to this immediately
IMG_DIMENSIONS_Q_HISTORY = {1, 32, 32} -- size of images fed into Q (action history)
IMG_DIMENSIONS_Q_LAST = {1, 64, 64} -- size of the last state's image fed into Q
--IMG_DIMENSIONS_AE = {1, 128, 128}

BATCH_SIZE = 16
STATES_PER_EXAMPLE = 4 -- how many states (previous + last one) to use per example fed into Q

GAMMA_EXPECTED = 0.9 -- discount factor to use for future rewards anticipated by Q
GAMMA_OBSERVED = 0.9 -- discount factor to use when cascading observed direct rewards backwards through time
MAX_GAMMA_REWARD = 100 -- clamp future rewards to +/- this value

P_EXPLORE_START = 0.8 -- starting epsilon value for epsilon greedy policy
P_EXPLORE_END = 0.1 -- ending epsilon value for epsilon greedy policy
P_EXPLORE_END_AT = 400000 -- when to end at P_EXPLORE_END (number of chosen actions)

LAST_SAVE_STATE_LOAD = 0 -- last time (in number of actions) when the game has been reset to a saved state

Q_L2_NORM = 1e-6 -- L2 parameter norm for Q
Q_CLAMP = 5 -- clamp Q gradients to +/- this value

----------------------------------
-- stats per training, will be saved and reloaded when training continues
----------------------------------
STATS = {
    STATE_ID = 0, -- id of the last created state
    FRAME_COUNTER = 0, -- number of the last frame
    ACTION_COUNTER = 0, -- count of actions chosen so far
    CURRENT_DIRECT_REWARD_SUM = 0, -- no longer used?
    CURRENT_OBSERVED_GAMMA_REWARD_SUM = 0, -- no longer used?
    AVERAGE_REWARD_DATA = {}, -- plot datapoints of rewards per N states
    AVERAGE_LOSS_DATA = {}, -- plot datapoints of losses per N batches
    LAST_BEST_ACTION_VALUE = 0, -- no longer used?
    P_EXPLORE_CURRENT = P_EXPLORE_START -- current epsilon value for epsilon greedy policy
}
STATS.STATE_ID = memory.getMaxStateId(1)
