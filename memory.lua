require 'torch'
require 'paths'

local memory = {}
memory.valData = {}
memory.trainData = {}
memory.MEMORY_MAX_SIZE_TRAINING = 120000
memory.MEMORY_MAX_SIZE_VALIDATION = 10000
memory.MEMORY_TRAINING_MIN_SIZE = 100

function memory.load()
    local fp = "learned/memory.th7"
    if paths.filep(fp) then
        local savedData = torch.load(fp)
        memory.valData = savedData.valData
        memory.trainData = savedData.trainData
    else
        print("[INFO] Could not load previously saved memory, file does not exist.")
    end
end

function memory.save()
    local fp = "learned/memory.th7"
    if paths.filep(fp) then
        os.execute(string.format("mv %s %s.old", fp, fp))
    end
    torch.save(fp, {valData=memory.valData, trainData=memory.trainData})
end

function memory.isTrainDataFull()
    return #memory.trainData == memory.MEMORY_MAX_SIZE_TRAINING
end

-- reorder states so that those with highest absolute(!) reward are at the top,
-- i.e. very high (level finished, killed mobs) or very low (death) rewards
--[[
function memory.reorderByReward()
    local comparer = function(statePair1, statePair2)
        -- we are only interested in the rewards of the outcome, i.e. after state 2
        local a = statePair1[2]
        local b = statePair2[2]
        local aReward = 0
        local bReward = 0
        if a.reward ~= nil then aReward = math.abs(rewards.getSumForTraining(a.reward)) end
        if b.reward ~= nil then bReward = math.abs(rewards.getSumForTraining(b.reward)) end
        return aReward > bReward
    end
    table.sort(memory.valData, comparer) -- table sort works in-place
    table.sort(memory.trainData, comparer) -- table sort works in-place
end
--]]

function memory.reevaluateRewards()
    print("[REEVALUATE] Started...")
    local batchSize = BATCH_SIZE * 6
    local reevaluate = function(data)
        --print("batchSize:", batchSize)
        collectgarbage()

        for i=1,#data,batchSize do
            --print(string.format("at %d", i))
            local batchStateChains = {}
            for j=i,math.min(i+batchSize-1, #data) do
                table.insert(batchStateChains, data[j].nextStateChain)
            end
            --print("batch size:", #batchStateChains)

            local bestActions = network.approximateBestActionsBatch(batchStateChains)
            --print("#actionValue:", #actionValues)
            for j=1,#batchStateChains do
                local entry = data[i+j-1]
                local nextState = entry.nextStateChain[#entry.nextStateChain]
                local lastState = entry.currentStateChain[#entry.currentStateChain]
                local oldReward = lastState.reward
                --local arrowActionValue = actionValues[j][lastState.action.arrow]
                --local buttonActionValue = actionValues[j][lastState.action.button]
                --local actionValue = (arrowActionValue + buttonActionValue)/2
                --local newReward = rewards.statesToReward(stateChain, lastState.action, actionValue)
                --print("curr size:", #entry.currentStateChain, "next size:", #entry.nextStateChain)
                local newReward = rewards.statesToReward(entry.nextStateChain, bestActions[j].action, bestActions[j].value)

                if oldReward.expectedGammaReward == nil then
                    print("[REEVALUATE] received nil as expected gamma reward in old reward")
                end
                if newReward.expectedGammaReward == nil then
                    print("[REEVALUATE] received nil as expected gamma reward in new reward")
                end

                --for key, value in pairs(actionValues[j]) do
                --    print("aV ", key, string.format("%.4f", value))
                --end
                --print(string.format("[A] Changed from %.4f to %.4f | action=%d", oldReward.expectedGammaReward, newReward.expectedGammaReward, lastState.action))
                --local bAV = network.approximateActionValue(stateChain, lastState.action)
                --print(string.format("[A] bAV=%.4f", bAV))
                newReward.observedGammaReward = oldReward.observedGammaReward
                lastState.reward = newReward
            end

            if (i-1) % (batchSize*100) == 0 then
                collectgarbage()
                print(string.format("[REEVALUATE] at %d of %d...", i, #data))
            end
        end
    end
    reevaluate(memory.valData)
    reevaluate(memory.trainData)
    collectgarbage()
end

function memory.reorderByDirectReward()
    local func = function (r)
        return math.abs(rewards.getDirectReward(r))
    end
    memory.reorderBy(func)
end

function memory.reorderBySurprise()
    local func = rewards.getSurprise
    memory.reorderBy(func)
end

function memory.reorderBy(func)
    local comparer = function(entry1, entry2)
        local stateChain1 = entry1.currentStateChain
        local stateChain2 = entry2.currentStateChain
        -- we are only interested in the rewards of the outcome, i.e. after the last state
        local a = stateChain1[#stateChain1]
        local b = stateChain2[#stateChain2]
        local aSurprise = 0
        local bSurprise = 0
        if a == nil then print("[warn] a is nil in memory.reorderBy()")
        elseif a.reward == nil then print("[warn] a.reward is nil in memory.reorderBy()")
        elseif a.reward.expectedGammaReward == nil then print("[warn] a.reward.expectedGammaReward is nil in memory.reorderBy()") end
        if b == nil then print("[warn] b is nil in memory.reorderBy()")
        elseif b.reward == nil then print("[warn] b.reward is nil in memory.reorderBy()")
        elseif b.reward.expectedGammaReward == nil then print("[warn] b.reward.expectedGammaReward is nil in memory.reorderBy()") end

        --if a.reward ~= nil then aSurprise = rewards.getSurprise(a.reward) end
        --if b.reward ~= nil then bSurprise = rewards.getSurprise(b.reward) end
        if a.reward ~= nil then aValue = func(a.reward) end
        if b.reward ~= nil then bValue = func(b.reward) end
        return aValue > bValue
    end
    table.sort(memory.valData, comparer) -- table sort works in-place
    table.sort(memory.trainData, comparer) -- table sort works in-place
end

--[[
function reevaluateExpectedGammaRewards()
    local data = memory.trainData
    for i=1,#data do
        local state1, state2 = unpack(data[i])

    end
end
--]]

--[[
function memory.addEntry(lastState, lastAction, state, action, reward)
    --print("Adding entry to memory (new size:" .. (#memory.data+1) .. ")...")
    table.insert(memory.data,
        {lastState = lastState, lastAction = lastAction, state = state, action = action, reward = reward}
    )
    if #memory.data > memory.MEMORY_MAX_SIZE then
        --print("Removing entry from memory...")
        memory.removeRandomEntry()
    end
end
--]]
function memory.addEntry(stateChain, nextState, validation)
    assert(validation == true or validation == false)
    local data
    local sizeLimit
    if validation then
        data = memory.valData
        sizeLimit = memory.MEMORY_MAX_SIZE_VALIDATION
    else
        data = memory.trainData
        sizeLimit = memory.MEMORY_MAX_SIZE_TRAINING
    end

    local nextStateChain = {}
    for x=2,#stateChain do
        table.insert(nextStateChain, stateChain[x])
    end
    table.insert(nextStateChain, nextState)

    local entry = {currentStateChain = stateChain, nextStateChain = nextStateChain}

    if #data <= 2 then
        table.insert(data, entry)
    elseif #data >= sizeLimit then
        local pos = memory.getRandomWeightedIndex("balanced", validation)
        data[pos] = entry
    else
        --local pos = math.floor(#data / 2)
        --table.insert(data, pos, entry)
        table.insert(data, entry)
    end

    memory.reduceToMaxSizes()
end

function memory.reduceToMaxSizes()
    local data = memory.trainData
    local size = #data
    local maxSize = memory.MEMORY_MAX_SIZE_TRAINING
    if size > maxSize then
        local nRemove = size - maxSize
        memory.removeRandomEntries(nRemove, false)
    end

    local data = memory.valData
    local size = #data
    local maxSize = memory.MEMORY_MAX_SIZE_VALIDATION
    if size > maxSize then
        local nRemove = size - maxSize
        memory.removeRandomEntries(nRemove, true)
    end
end

--[[
function memory.removeRandomEntry()
    local entryIdx = math.random(#memory.data)
    local reward = memory.data[entryIdx].reward
    if reward >= -1.0 and reward <= 1.0 then
        table.remove(memory.data, entryIdx)
    elseif math.random() < 0.05 then
        table.remove(memory.data, entryIdx)
    else
        memory.removeRandomEntry()
    end
end
--]]
--[[
function memory.removeRandomEntry()
    local entryIdx = math.random(#memory.data)
    local states = memory.data[entryIdx]
    local state1, state2 = states[1], states[2]
    local reward = rewards.getSum(state2.reward)
    if reward >= -5.0 and reward <= 5.0 then
        table.remove(memory.data, entryIdx)
    elseif math.random() < 0.05 then
        table.remove(memory.data, entryIdx)
    else
        memory.removeRandomEntry()
    end
end
--]]

function memory.removeRandomEntry(validation)
    memory.removeRandomEntries(1)
end

function memory.removeRandomEntries(nRemove, validation, skew)
    assert(validation == true or validation == false)
    skew = skew or "balanced"
    local data
    if validation then
        data = memory.valData
    else
        data = memory.trainData
    end

    for i=1,math.min(nRemove, #data) do
        local entryIdx = memory.getRandomWeightedIndex(skew, validation)
        table.remove(data, entryIdx)
    end
end

function memory.getRandomWeightedIndex(skew, validation, skewStrength)
    assert(skew == "balanced" or skew == "top" or skew == "bottom", "Skew must be 'balanced' or 'top' or 'bottom'")
    assert(validation == true or validation == false)
    skewStrength = skewStrength or 2
    assert(skewStrength >= 1)

    local data
    if validation then
        data = memory.valData
    else
        data = memory.trainData
    end
    local size = #data

    if skew == "balanced" then
        return math.random(size)
    end

    -- combine N random values between 0 and 1,
    -- the result should be skewed towards 0
    local randomFloat = 1
    for i=1,skewStrength do
        randomFloat = randomFloat * math.random()
    end

    -- if we prefer values with small absolute rewards, then just make
    -- it skewed towards 1 by (1 - p)
    -- towards 1 because we will do p*#entries, higher p's result in higher
    -- end values, which means later indexes/entries
    if skew == "bottom" then
        randomFloat = 1 - randomFloat
    end

    -- convert to the integer index of an entry
    local randomIdx = math.floor(randomFloat * size)

    -- no clue if randomIdx can really hit the exactly 1st and last entries,
    -- so we just add +1 or -1 randomly here and then clip the resulting value,
    -- just to be sure
    if math.random() < 0.5 then
        randomIdx = randomIdx - 1
    else
        randomIdx = randomIdx + 1
    end
    if randomIdx > size then
        randomIdx = size
    elseif randomIdx < 1 then
        randomIdx = 1
    end

    return randomIdx
end


--[[
function memory.getRandomBatch(batchSize)
    local c, h, w = IMG_DIMENSIONS[1], IMG_DIMENSIONS[2], IMG_DIMENSIONS[3]
    local batchInput = torch.zeros(batchSize, c*2, h, w)
    local batchTarget = torch.zeros(batchSize, #actions.ACTIONS_NETWORK)
    for i=1,batchSize do
        local idx = math.random(#memory.data)
        local entry = memory.data[idx]
        local actionIdx = actions.getIndexOfAction(entry.action)
        batchInput[i] = network.imagesToInput(entry.lastState, entry.state)
        batchTarget[{i, actionIdx}] = entry.reward:getSum()
        --print("reward:", entry.reward)
    end
    return batchInput, batchTarget
end
--]]
function memory.getTrainingBatch(batchSize)
    return memory.getBatch(batchSize, false)
end

function memory.getValidationBatch(batchSize)
    return memory.getBatch(batchSize, true)
end

-- TODO replace as far as possible with network.stateChainsToBatch()
--[[
function memory.getBatch(batchSize, validation)
    assert(validation == true or validation == false)
    local data
    if validation then
        data = memory.valData
    else
        data = memory.trainData
    end

    local c, h, w = IMG_DIMENSIONS_Q[1], IMG_DIMENSIONS_Q[2], IMG_DIMENSIONS_Q[3]
    --local batchInput = torch.zeros(batchSize, c, h, w*2)
    local batchInput = torch.zeros(batchSize, STATES_PER_EXAMPLE, IMG_DIMENSIONS_Q[2], IMG_DIMENSIONS_Q[3])
    local batchTarget = torch.zeros(batchSize, #actions.ACTIONS_NETWORK)
    for i=1,batchSize do
        --local idx = math.random(#memory.data)
        local idx = memory.getRandomWeightedIndex("top", validation)
        --print("Choose index", idx, "of", #memory.data)
        --local state12 = data[idx] -- dont call this states, as states is reserved for states.lua
        --local state1, state2 = state12[1], state12[2]
        --local screen1 = states.decompressScreen(state1.screen)
        --local screen2 = states.decompressScreen(state2.screen)
        --local actionIdx = actions.getIndexOfAction(state2.action)
        --batchInput[i] = network.imagesToInput(screen1, screen2)
        local stateChain = data[idx]
        batchInput[i] = network.statesToInput(stateChain)
        --print("state1", state1.reward)
        --print("state2", state2.reward)
        --batchTarget[{i, actionIdx}] = rewards.getSumObserved(state2.reward)
        batchTarget[{i, actionIdx}] = network.statesToTarget(stateChain)
        --print("reward:", entry.reward)
    end
    return batchInput, batchTarget
end
--]]

function memory.getBatch(batchSize, validation)
    assert(validation == true or validation == false)
    local data
    if validation then
        data = memory.valData
    else
        data = memory.trainData
    end

    local stateChains = {}
    for i=1,batchSize do
        --local idx = memory.getRandomWeightedIndex("top", validation)
        local idx = memory.getRandomWeightedIndex("balanced", validation)
        local stateChain = data[idx].currentStateChain
        table.insert(stateChains, stateChain)
    end
    local batchInput, batchTarget = network.stateChainsToBatch(stateChains)
    return batchInput, batchTarget
end


function memory.getAutoencoderBatch(batchSize)
    local c, h, w = IMG_DIMENSIONS[1], IMG_DIMENSIONS[2], IMG_DIMENSIONS[3]
    local cAE, hAE, wAE = IMG_DIMENSIONS_AE[1], IMG_DIMENSIONS_AE[2], IMG_DIMENSIONS_AE[3]
    local batchInput = torch.zeros(batchSize, 1, h, w)
    for i=1,batchSize do
        local idx = memory.getRandomWeightedIndex("top")
        local state12 = memory.data[idx] -- dont call this states, as states is reserved for states.lua
        local state2 = state12[2]
        local screen2 = states.decompressScreen(state2.screen)
        if c ~= cAE then
            screen2 = util.rgb2y(screen2)
        end
        screen2 = image.scale(screen2, hAE, wAE)
        batchInput[i] = screen2
    end
    return batchInput
end

function memory.plot(subsampling)
    subsampling = subsampling or 1
    local points = {}
    for i=1,#memory.trainData,subsampling do
        local stateChain = memory.trainData[i].currentStateChain
        local lastState = stateChain[#stateChain]
        if lastState.reward ~= nil then
            --print("reward = ", rewards.getSum(pair[2].reward))
            --table.insert(points, {i, rewards.getSumForTraining(pair[2].reward)})
            --table.insert(points, {i, rewards.getSurprise(lastState.reward)})
            table.insert(points, {i, rewards.getDirectReward(lastState.reward)})
        else
            --print("reward = ", nil)
            table.insert(points, {i, 0})
        end
    end
    --display.plot(points, {win=2, labels={'entry', 'reward'}, title='Index to surprise in memory (training data)'})
    display.plot(points, {win=2, labels={'entry', 'reward'}, title='Index to direct reward in memory (training data)'})
end

return memory
