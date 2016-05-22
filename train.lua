print("------------------------")
print("START")
print("------------------------")

GPU = 0
SEED = 43
lsne_memory = memory

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
VAE = require 'VAE'
ForgivingMSECriterion = require 'layers.ForgivingMSECriterion'
ForgivingAbsCriterion = require 'layers.ForgivingAbsCriterion'
ok, display = pcall(require, 'display')
if not ok then print('display not found. unable to plot') end

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

FPS = movie.get_game_info().fps
--local fps = 100
--local reactEveryNthFrame = 6 -- roughly 26/4, i.e. every 0.25s
--REACT_EVERY_NTH_FRAME = math.floor(FPS / 5)
REACT_EVERY_NTH_FRAME = 5
print(string.format("FPS: %d, Reacting every %d frames", FPS, REACT_EVERY_NTH_FRAME))

SCREENSHOT_FILEPATH = "/media/ramdisk/mario-ai-screenshots/current-screen.png"
IMG_DIMENSIONS = {1, 64, 64}
--IMG_DIMENSIONS_Q = {1, 48, 48}
IMG_DIMENSIONS_Q_HISTORY = {1, 32, 32}
IMG_DIMENSIONS_Q_LAST = {1, 64, 64}
IMG_DIMENSIONS_AE = {1, 128, 128}
BATCH_SIZE = 16
STATES_PER_EXAMPLE = 4
GAMMA_EXPECTED = 0.9
GAMMA_OBSERVED = 0.9
MAX_GAMMA_REWARD = 100
P_EXPLORE_START = 0.8
P_EXPLORE_END = 0.1
P_EXPLORE_END_AT = 400000
STATS = {
    STATE_ID = 1,
    FRAME_COUNTER = 0,
    ACTION_COUNTER = 0,
    CURRENT_DIRECT_REWARD_SUM = 0,
    CURRENT_OBSERVED_GAMMA_REWARD_SUM = 0,
    AVERAGE_REWARD_DATA = {},
    AVERAGE_LOSS_DATA = {},
    LAST_BEST_ACTION_VALUE = 0,
    P_EXPLORE_CURRENT = P_EXPLORE_START
}
LAST_SAVE_STATE_LOAD = 0

print("Loading/Creating network...")
Q_L2_NORM = 1e-6
Q_CLAMP = 10
Q = network.createOrLoadQ()

PARAMETERS, GRAD_PARAMETERS = Q:getParameters()
--CRITERION = nn.ForgivingAbsCriterion()
CRITERION = nn.ForgivingMSECriterion()
--CRITERION = nn.MSECriterion()
OPTCONFIG = {learningRate=0.001, beta1=0.9, beta2=0.999}
DECAY = 1.0
--OPTCONFIG = {learningRate=0.001, momentum=0.9}
OPTSTATE = {}

--MODEL_AE, CRITERION_AE_LATENT, CRITERION_AE_RECONSTRUCTION, PARAMETERS_AE, GRAD_PARAMETERS_AE = VAE.createVAE()
--OPTCONFIG_AE = { learningRate=0.001 }
--OPTSTATE_AE = {}

--print("Number of parameters in AE:", network.getNumberOfParameters(MODEL_AE))
print("Number of parameters in Q:", network.getNumberOfParameters(Q))

print("Loading memory...")
memory.load()

print("Loading stats...")
util.loadStats()

print("Starting loop.")

--[[
local points = {}
for i=1,10000 do
    table.insert(points, math.random()*math.random()*math.random())
end
table.sort(points)
points2 = {}
for i=1,10000 do
    table.insert(points2, {i, points[i]})
end
display.plot(points2, {win=5, labels={'entry', 'val'}, title='Random values'})
--]]

function on_paint()
    local lastLastState = states.getEntry(-2)
    local lastState = states.getEntry(-1)

    -- last best action value
    if lastLastState ~= nil then
        gui.text(1, 1, string.format("LLRew: %.2f/%.2f", rewards.getDirectReward(lastLastState.reward), lastLastState.reward.expectedGammaReward))
        gui.text(1+175, 1, string.format("LLBAV: %.2f", STATS.LAST_BEST_ACTION_VALUE or 0))
    end
    gui.text(1+350-15, 1, string.format("Memory: %d/%d", #memory.trainData, #memory.valData))

    --[[
    local observedGammaRewards = "oGR: "
    local nStates = #states.data
    for i=1,math.min(8, nStates) do
        local s = states.data[nStates-i+1]
        local ogr = s.reward and s.reward.observedGammaReward or 0 -- the last (most recent) state does not have a reward yet
        observedGammaRewards = observedGammaRewards .. string.format("%.2f ", ogr)
    end
    gui.text(1, 15, observedGammaRewards)
    --]]
end

function on_frame_emulated()
    local lastLastState = states.getEntry(-2)
    local lastState = states.getEntry(-1)
    STATS.FRAME_COUNTER = movie.currentframe()

    -- last best action value
    --[[
    if lastLastState ~= nil then
        gui.text(1, 1, string.format("LLRew: %.2f/%.2f", rewards.getDirectReward(lastLastState.reward), rewards.getSum(lastLastState.reward)))
        gui.text(1+175, 1, string.format("LLBAV: %.2f", LAST_BEST_ACTION_VALUE or 0))
    end
    gui.text(1+350, 1, string.format("Memory: %d", #memory.data))
    --]]

    --if (STATS.FRAME_COUNTER+1) % REACT_EVERY_NTH_FRAME == 0 then
    --    actions.endAllActions()
    --    return
    if STATS.FRAME_COUNTER % REACT_EVERY_NTH_FRAME ~= 0 then
        return
    end

    STATS.ACTION_COUNTER = STATS.ACTION_COUNTER + 1

    if STATS.ACTION_COUNTER % 4000 == 0 then
        print("Garbage collection...")
        collectgarbage()
        util.sleep(2)
    end

    if lastState == nil then print("[NOTE] lastState is nil") end
    if lastLastState == nil then print("[NOTE] lastLastState is nil") end

    local state = State.new(nil, getScreenCompressed(), util.getCurrentScore(), util.getCountLifes(), util.getLevelBeatenStatus(), util.getMarioGameStatus(), util.getPlayerX(), util.getMarioImage(), util.isLevelEnding())
    states.addEntry(state) -- getLastEntries() depends on this, don't move it after the next code block
    --print("Score:", score, "Level:", util.getLevel(), "x:", playerX, "status:", marioGameStatus, "levelBeatenStatus:", levelBeatenStatus, "count lifes:", countLifes)
    --print("Mario Image", util.getMarioImage())

    -- Calculate reward
    local rew, bestAction, bestActionValue = rewards.statesToReward(states.getLastEntries(STATES_PER_EXAMPLE))
    lastState.reward = rew
    --print(string.format("[Reward] R=%.2f DR=%.2f SDR=%.2f XDR=%.2f LBR=%.2f EGR=%.2f", rewards.getSumExpected(lastState.reward), rewards.getDirectReward(lastState.reward), lastState.reward.scoreDiffReward, lastState.reward.xDiffReward, lastState.reward.levelBeatenReward, lastState.reward.expectedGammaReward))
    --print(string.format("Cascading reward %.4f", rewards.getDirectReward(lastState.reward)))
    states.cascadeBackReward(lastState.reward)
    STATS.LAST_BEST_ACTION_VALUE = bestActionValue

    -- Add to memory
    local pastStates = states.getLastEntries(STATES_PER_EXAMPLE+1)
    table.remove(pastStates) -- pop last state as that state is currently still in progress (eg no reward yet)
    local ac1000 = STATS.ACTION_COUNTER % 1000
    local validation = (ac1000 >= 750 and ac1000 < 800) or (ac1000 >= 950 and ac1000 < 1000)
    memory.addEntry(pastStates, state, validation)

    -- show state chain
    -- must happen before training as it might depend on network's current output
    display.image(states.stateChainToImage(states.getLastEntries(STATES_PER_EXAMPLE), Q), {win=17, title="Last states"})

    -- plot average rewards
    if STATS.ACTION_COUNTER % 3 == 0 then
        states.plotRewards()
    end

    --------------------

    --[[
    if STATS.ACTION_COUNTER % 250 == 0 then
        table.insert(STATS.AVERAGE_REWARD_DATA, {STATS.ACTION_COUNTER, STATS.CURRENT_DIRECT_REWARD_SUM / 50, STATS.CURRENT_OBSERVED_GAMMA_REWARD_SUM / 50})
        STATS.CURRENT_DIRECT_REWARD_SUM = 0
        STATS.CURRENT_OBSERVED_GAMMA_REWARD_SUM = 0
        plotAverageReward()
    end
    --]]
    if STATS.ACTION_COUNTER % states.MAX_SIZE == 0 then
        local directRewardSum = 0
        local observedGammaRewardSum = 0
        local expectedGammaRewardSum = 0
        for i=1,#states.dataAll do
            if states.dataAll[i].reward ~= nil then
                directRewardSum = directRewardSum + rewards.getDirectReward(states.dataAll[i].reward)
                observedGammaRewardSum = observedGammaRewardSum + states.dataAll[i].reward.observedGammaReward
                expectedGammaRewardSum = expectedGammaRewardSum + states.dataAll[i].reward.expectedGammaReward
            end
        end
        table.insert(STATS.AVERAGE_REWARD_DATA, {STATS.ACTION_COUNTER, directRewardSum / #states.dataAll, observedGammaRewardSum / #states.dataAll, expectedGammaRewardSum / #states.dataAll})
        plotAverageReward()
    end

    if STATS.ACTION_COUNTER % 10000 == 0 then
        print("Reevaluating rewards in memory...")
        memory.reevaluateRewards()
        print("Reordering memory...")
        memory.reorderByDirectReward()
        print("Plotting memory...")
        memory.plot(10)
        --print("Reordering finished")
        --print(string.format("1st: %.4f", memory.data[1][2].reward and rewards.getSumExpected(memory.data[1][2].reward) or 123.456))
        --print(string.format("2st: %.4f", memory.data[2][2].reward and rewards.getSumExpected(memory.data[2][2].reward) or 123.456))
        --print(string.format("3st: %.4f", memory.data[3][2].reward and rewards.getSumExpected(memory.data[3][2].reward) or 123.456))
        --print(string.format("4st: %.4f", memory.data[4][2].reward and rewards.getSumExpected(memory.data[4][2].reward) or 123.456))
        --print(string.format("5st: %.4f", memory.data[5][2].reward and rewards.getSumExpected(memory.data[5][2].reward) or 123.456))
        --print("Done.")
    end

    if (STATS.ACTION_COUNTER == 250 and memory.isTrainDataFull())
        or STATS.ACTION_COUNTER % 5000 == 0 then
        --print("Training AE...")
        --for i=1,25 do
        --    trainAE()
        --end

        print("Training...")
        local nTrainingBatches = 3500 --math.max(math.floor(#memory.trainData / BATCH_SIZE), 51)
        local nTrainingGroups = 50 -- number of plot points per training epoch
        local nTrainBatchesPerGroup = math.floor(nTrainingBatches / nTrainingGroups)
        local nValBatchesPerGroup = math.floor(nTrainBatchesPerGroup * 0.10)
        for i=1,nTrainingGroups do
            local sumLossTrain = 0
            local sumLossVal = 0
            for j=1,nTrainBatchesPerGroup do
                local loss = trainOneBatch()
                sumLossTrain = sumLossTrain + loss
                print(string.format("[BATCH %d/%d] loss=%.8f", (i-1)*nTrainBatchesPerGroup + j, nTrainingBatches, loss))
            end
            for j=1,nValBatchesPerGroup do
                sumLossVal = sumLossVal + valOneBatch()
            end
            table.insert(STATS.AVERAGE_LOSS_DATA, {#STATS.AVERAGE_LOSS_DATA+1, sumLossTrain/nTrainBatchesPerGroup, sumLossVal/nValBatchesPerGroup})
            plotAverageLoss()
        end

        --[[
        print("Training...")
        local nBatchesTrain = 2500
        local nBatchesVal = 250
        local sumLossTrain = 0
        local sumLossVal = 0
        for i=1,nBatchesTrain do
            local loss = trainOneBatch()
            sumLossTrain = sumLossTrain + loss
            print(string.format("[BATCH] loss=%.8f", loss))
        end
        for i=1,nBatchesVal do
            sumLossVal = sumLossVal + valOneBatch()
        end
        table.insert(STATS.AVERAGE_LOSS_DATA, {STATS.ACTION_COUNTER, sumLossTrain/nBatchesTrain, sumLossVal/nBatchesVal})
        plotAverageLoss()
        --]]

        OPTCONFIG.learningRate = OPTCONFIG.learningRate * DECAY
        print(string.format("[LEARNING RATE] %.12f", OPTCONFIG.learningRate))
    end

    --print("bestAction:", bestAction, bestAction.arrow, bestAction.button)
    state.action = chooseAction(states.getLastEntries(STATES_PER_EXAMPLE), false, bestAction)
    --states.addEntry(screen, action, bestActionValue, score, playerX, countLifes, levelBeatenStatus, marioGameStatus)


    local levelEnded = state.levelBeatenStatus == 128 or state.marioGameStatus == 2
    if levelEnded or (STATS.ACTION_COUNTER - LAST_SAVE_STATE_LOAD) > 1000 then
        -- Reload save state if level was beaten or mario died
        util.loadRandomTrainingSaveState()
        states.clear()
        LAST_SAVE_STATE_LOAD = STATS.ACTION_COUNTER
    else
        actions.endAllActions()
        actions.startAction(state.action)
    end



    -- decay exploration rate
    local pPassed = math.min(STATS.ACTION_COUNTER / P_EXPLORE_END_AT, 1.0)
    STATS.P_EXPLORE_CURRENT = (1-pPassed) * P_EXPLORE_START + pPassed * P_EXPLORE_END
    if STATS.ACTION_COUNTER % 250 == 0 then
        print(string.format("[EXPLORE P] %.2f", STATS.P_EXPLORE_CURRENT))
    end

    -- save
    if STATS.ACTION_COUNTER % 10000 == 0 then
        print("Saving stats...")
        util.saveStats()
        print("Saving network...")
        network.save()
        print("Saving memory...")
        memory.save()

        print("Clearing data and reloading...")
        states.data = {}
        states.dataAll = {}
        memory.valData = {}
        memory.trainData = {}
        collectgarbage()
        util.sleep(1)
        memory.load()
        collectgarbage()
        util.sleep(1)
        states.clear() -- refills with empty states
    end
end

function plotAverageReward()
    display.plot(STATS.AVERAGE_REWARD_DATA, {win=3, labels={'action counter', 'direct', 'observed gamma', 'expected gamma'}, title='Average rewards per N actions'})
end

function plotAverageLoss()
    display.plot(STATS.AVERAGE_LOSS_DATA, {win=4, labels={'batch group', 'training', 'validation'}, title='Average loss per batch'})
end

function getScreen()
    --------------------
    -- Estimate the reward of the chosen action,
    -- add it to memory
    --------------------
    -- Current State
    -- screenshot_bitmap() => DBITMAP
    -- DBITMAP members:
    --  blit_scaled
    --  blit_porterduff
    --  __gc
    --  draw_clip
    --  pset
    --  __newindex
    --  size
    --  hash
    --  blit_scaled_porterduff
    --  draw
    --  draw_clip_outside
    --  blit
    --  save_png
    --  adjust_transparency
    --  pget
    --  draw_outside
    --  __index
    --local screen = gui.screenshot_bitmap()
    local fp = "/media/ramdisk/mario-ai-screenshots/current-screen.jpeg"
    gui.screenshot(fp)
    local screen = image.load(fp, 3, "float"):clone()
    screen = image.scale(screen, IMG_DIMENSIONS[2], IMG_DIMENSIONS[3]):clone()
    if IMG_DIMENSIONS[1] == 1 then
        screen = util.rgb2y(screen)
    end
    return screen
end

function getScreenCompressed()
    local fp = SCREENSHOT_FILEPATH
    gui.screenshot(fp)
    return util.loadJPGCompressed(fp, IMG_DIMENSIONS[1], IMG_DIMENSIONS[2], IMG_DIMENSIONS[3])
end

function trainAE()
    local batchInput = memory.getAutoencoderBatch(BATCH_SIZE)
    VAE.train(batchInput, MODEL_AE, CRITERION_AE_LATENT, CRITERION_AE_RECONSTRUCTION, PARAMETERS_AE, GRAD_PARAMETERS_AE, OPTCONFIG_AE, OPTSTATE_AE)
end

function trainOneBatch()
    -- Train
    --FRAME_COUNTER % BATCH_SIZE == 0
    if #memory.trainData >= memory.MEMORY_TRAINING_MIN_SIZE then
        local batchInput, batchTarget = memory.getTrainingBatch(BATCH_SIZE)
        --print("Training with a batch of size " .. batchInput:size(1))
        return network.forwardBackwardBatch(batchInput, batchTarget)
    else
        return 0
    end
end

function valOneBatch()
    if #memory.valData >= BATCH_SIZE then
        local batchInput, batchTarget = memory.getValidationBatch(BATCH_SIZE)
        return network.batchToLoss(batchInput, batchTarget)
    else
        return 0
    end
end

--[[
function chooseAction(lastState, state, perfect, bestAction)
    perfect = perfect or false
    local _action, _actionValue
    if lastState == nil or state == nil or (not perfect and math.random() < STATS.P_EXPLORE_CURRENT) then
        _action = util.getRandomEntry(actions.ACTIONS_NETWORK)
        --print("Chossing action randomly:", _action)
    else
        if bestAction ~= nil then
            _action = bestAction
        else
            -- Use network to approximate action with maximal value
            _action, _actionValue = network.approximateBestAction(lastState, state)
            --print("Q approximated action:", _action, actions.ACTION_TO_BUTTON_NAME[_action])
        end
    end

    return _action
end
--]]

function chooseAction(lastStates, perfect, bestAction)
    perfect = perfect or false
    local _action, _actionValue
    if not perfect and math.random() < STATS.P_EXPLORE_CURRENT then
        if bestAction == nil or math.random() < 0.5 then
            -- randomize both
            _action = Action.new(util.getRandomEntry(actions.ACTIONS_ARROWS), util.getRandomEntry(actions.ACTIONS_BUTTONS))
        else
            -- randomize only arrow or only button
            if math.random() < 0.5 then
                _action = Action.new(util.getRandomEntry(actions.ACTIONS_ARROWS), bestAction.button)
            else
                _action = Action.new(bestAction.arrow, util.getRandomEntry(actions.ACTIONS_BUTTONS))
            end
        end
        --print("Chossing action randomly:", _action)
    else
        if bestAction ~= nil then
            _action = bestAction
        else
            -- Use network to approximate action with maximal value
            _action, _actionValue = network.approximateBestAction(lastStates)
            --print("Q approximated action:", _action, actions.ACTION_TO_BUTTON_NAME[_action])
        end
    end

    return _action
end

--setToFastSpeed()
--for lcid=1,8 do
--    local port, controller = input.lcid_to_pcid2(lcid)
--    print("controller_info:", port, controller, input.controller_info(port, controller))
--end

--print("controller type 0,0:", input.controllertype(0, 0))
--print("controller type 0,1:", input.controllertype(0, 1))
--print("controller type 1,0:", input.controllertype(1, 0))
--print("controller type 1,1:", input.controllertype(1, 1))
--print("controller_info A:", get_controller_info())
--print("controller_info B:", 1, 0, input.controller_info(1, 0))
--for i=0,128 do
--    print("input.joyget("..i.."):", input.joyget(i))
--end
--startAction(ACTION_BUTTON_START)
--print("input.raw:", input.raw())
--input.do_button_action("gamepad-1-A", 1, 1)

--endAllActions()
--startAction(ACTION_BUTTON_SELECT)
--endAllActions()
--startAction(ACTION_BUTTON_A)
--endAllActions()
--startAction(ACTION_BUTTON_B)
--endAllActions()
--startAction(ACTION_BUTTON_X)
--endAllActions()
--startAction(ACTION_BUTTON_Y)
--endAllActions()
--exit()
--movie.to_rewind("lvl1.lsmv")
actions.endAllActions()
util.loadRandomTrainingSaveState()
util.setGameSpeedToVeryFast()
states.fillWithEmptyStates()
gui.repaint()
