print("------------------------")
print("TRAINING")
print("------------------------")

require 'paths'
paths.dofile("config.lua")

Q = network.createOrLoadQ()
PARAMETERS, GRAD_PARAMETERS = Q:getParameters()
--CRITERION = nn.ForgivingAbsCriterion()
CRITERION = nn.ForgivingMSECriterion()
--CRITERION = nn.MSECriterion()
OPTCONFIG = {learningRate=0.001*0.1, beta1=0.9, beta2=0.999}
OPTSTATE = {}
DECAY = 1.0

--MODEL_AE, CRITERION_AE_LATENT, CRITERION_AE_RECONSTRUCTION, PARAMETERS_AE, GRAD_PARAMETERS_AE = VAE.createVAE()
--OPTCONFIG_AE = { learningRate=0.001 }
--OPTSTATE_AE = {}

--print("Number of parameters in AE:", network.getNumberOfParameters(MODEL_AE))
print("Number of parameters in Q:", network.getNumberOfParameters(Q))

print("Loading memory...")
memory.load()

print("Loading stats...")
util.loadStats()
STATS.STATE_ID = memory.getMaxStateId(1)

print("Starting loop.")

function on_paint()
    gui.text(1+350-15, 1, string.format("Memory: %d/%d", memory.getCountEntriesCached(false), memory.getCountEntriesCached(true)))
end

function on_frame_emulated()
    STATS.FRAME_COUNTER = movie.currentframe()

    if STATS.FRAME_COUNTER % REACT_EVERY_NTH_FRAME ~= 0 then
        return
    end

    local lastLastState = states.getEntry(-2)
    local lastState = states.getEntry(-1)
    assert(lastLastState ~= nil)
    assert(lastState ~= nil)

    STATS.ACTION_COUNTER = STATS.ACTION_COUNTER + 1

    local state = State.new(nil, util.getScreenCompressed(), util.getCurrentScore(), util.getCountLifes(), util.getLevelBeatenStatus(), util.getMarioGameStatus(), util.getPlayerX(), util.getMarioImage(), util.isLevelEnding())
    states.addEntry(state) -- getLastEntries() depends on this, don't move it after the next code block
    --print("Score:", score, "Level:", util.getLevel(), "x:", playerX, "status:", marioGameStatus, "levelBeatenStatus:", levelBeatenStatus, "count lifes:", countLifes, "Mario Image:", util.getMarioImage())

    -- Calculate reward
    local rew, bestAction, bestActionValue = rewards.statesToReward(states.getLastEntries(STATES_PER_EXAMPLE))
    lastState.reward = rew
    --print(string.format("[Reward] R=%.2f DR=%.2f SDR=%.2f XDR=%.2f LBR=%.2f EGR=%.2f", rewards.getSumExpected(lastState.reward), rewards.getDirectReward(lastState.reward), lastState.reward.scoreDiffReward, lastState.reward.xDiffReward, lastState.reward.levelBeatenReward, lastState.reward.expectedGammaReward))
    states.cascadeBackReward(lastState.reward)
    STATS.LAST_BEST_ACTION_VALUE = bestActionValue

    -- show state chain
    -- must happen before training as it might depend on network's current output
    display.image(states.stateChainsToImage({states.getLastEntries(STATES_PER_EXAMPLE)}, Q), {win=17, title="Last states"})

    -- plot average rewards
    if STATS.ACTION_COUNTER % 3 == 0 then
        states.plotRewards()
    end

    --------------------

    -- plot average rewards per N actions
    if STATS.ACTION_COUNTER % 1000 == 0 then
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
        util.plotAverageReward(STATS.AVERAGE_REWARD_DATA)
    end

    if (STATS.ACTION_COUNTER == 250 and memory.isTrainDataFull())
        or STATS.ACTION_COUNTER % 5000 == 0 then
        print("Preparing state ids cache...")
        memory.prepareStateIdsCache()

        print("Training...")
        local nTrainingBatches = 2500 --math.max(math.floor(#memory.trainData / BATCH_SIZE), 51)
        local nTrainingGroups = 50 -- number of plot points per training epoch
        local nTrainBatchesPerGroup = math.floor(nTrainingBatches / nTrainingGroups)
        local nValBatchesPerGroup = math.floor(nTrainBatchesPerGroup * 0.10)
        assert(nTrainBatchesPerGroup >= 1)
        assert(nValBatchesPerGroup >= 1)
        for i=1,nTrainingGroups do
            local sumLossTrain = 0
            local sumLossVal = 0
            local batchStart = (i-1)*nTrainBatchesPerGroup
            local batchEnd = math.min(batchStart + nTrainBatchesPerGroup, nTrainingBatches)
            sumLossTrain = trainBatches(batchStart, batchEnd, nTrainingBatches)
            --for j=1,nTrainBatchesPerGroup do
            --    local loss = trainBatches()
            --    sumLossTrain = sumLossTrain + loss
            --end
            for j=1,nValBatchesPerGroup do
                sumLossVal = sumLossVal + valOneBatch()
            end
            table.insert(STATS.AVERAGE_LOSS_DATA, {#STATS.AVERAGE_LOSS_DATA+1, sumLossTrain, sumLossVal/nValBatchesPerGroup})
            network.plotAverageLoss(STATS.AVERAGE_LOSS_DATA)
        end

        OPTCONFIG.learningRate = OPTCONFIG.learningRate * DECAY
        print(string.format("[LEARNING RATE] %.12f", OPTCONFIG.learningRate))
    end

    -- choose action
    state.action = actions.chooseAction(states.getLastEntries(STATES_PER_EXAMPLE), false, bestAction)

    -- reset to last save state or perform action
    local levelEnded = state.levelBeatenStatus == 128 or state.marioGameStatus == 2
    if levelEnded or (STATS.ACTION_COUNTER - LAST_SAVE_STATE_LOAD) > 750 then
        print("Reloading saved gamestate and saving states...")
        states.addToMemory()
        states.clear()

        -- Reload save state if level was beaten or mario died
        util.loadRandomTrainingSaveState()
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
    if STATS.ACTION_COUNTER % 5000 == 0 then
        print("Saving stats...")
        util.saveStats()
        print("Saving network...")
        network.save()
    end
end


--[[
function trainAE()
    local batchInput = memory.getAutoencoderBatch(BATCH_SIZE)
    VAE.train(batchInput, MODEL_AE, CRITERION_AE_LATENT, CRITERION_AE_RECONSTRUCTION, PARAMETERS_AE, GRAD_PARAMETERS_AE, OPTCONFIG_AE, OPTSTATE_AE)
end
--]]

function trainBatches(batchStart, batchEnd, batchEndAll)
    local nBatches = batchEnd - batchStart
    if memory.getCountEntriesCached(false) >= memory.MEMORY_TRAINING_MIN_SIZE then
        local sumLoss = 0
        local Q_clone = Q:clone()

        for i=1,nBatches do
            -- takes ~40% of the time
            --    <1% load random state ids
            --    10% load states by id
            --    30% reevaluate rewards
            --local timeStart = os.clock()
            local batchInput, batchTarget, stateChains = memory.getBatch(BATCH_SIZE, false, true, Q_clone)
            --print(string.format("getBatch: %.8f", os.clock()-timeStart))

            -- takes ~52% of the time
            --local timeStart = os.clock()
            local loss = network.forwardBackwardBatch(batchInput, batchTarget)
            sumLoss = sumLoss + loss
            --print(string.format("fwbw: %.8f", os.clock()-timeStart))
            print(string.format("[BATCH %d/%d] loss=%.8f", batchStart+i-1, batchEndAll, loss))

            -- takes ~7% of the time
            --local timeStart = os.clock()
            display.image(states.stateChainsToImage(stateChains, Q, 1, 1), {win=18, title="Last training batch, 1st example"})
            --print(string.format("plot: %.8f", os.clock()-timeStart))
        end

        Q_clone = nil

        return sumLoss / nBatches
    else
        return 0
    end
end

function valOneBatch()
    if memory.getCountEntriesCached(true) >= BATCH_SIZE then
        local batchInput, batchTarget = memory.getBatch(BATCH_SIZE, true, true)
        return network.batchToLoss(batchInput, batchTarget)
    else
        return 0
    end
end

--[[
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
--]]

if STATS.ACTION_COUNTER > 0 then
    states.plotRewards()
    util.plotAverageReward(STATS.AVERAGE_REWARD_DATA)
    network.plotAverageLoss(STATS.AVERAGE_LOSS_DATA)
end

actions.endAllActions()
util.loadRandomTrainingSaveState()
util.setGameSpeedToVeryFast()
states.fillWithEmptyStates()
gui.repaint()
