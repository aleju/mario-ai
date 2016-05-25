-- Functions to manage the history of recent states.
-- This is not the same as the replay memory, which is more permanent.

local states = {
    data = {}, -- only the recent states of the current run, cleared after each death/level-finish
    dataAll = {} -- all recent states
}

states.MAX_SIZE = 5000 -- max size of the history of recent states.
states.CASCADE_BACK = 100 -- how far backwards (in states) to cascade direct rewards to calculate observed future rewards.

-------------------------
-- State object member functions
-------------------------
function states.decompressScreen(screen)
    return util.decompressJPG(screen)
end

-- Creates an empty/dummy state.
function states.createEmptyState()
    local screen = torch.zeros(3, IMG_DIMENSIONS[2], IMG_DIMENSIONS[3])
    screen = image.drawText(screen, string.format("F%d", STATS.FRAME_COUNTER), 0, 0, {color={255,255,255}})
    if IMG_DIMENSIONS[1] == 1 then
        screen = util.rgb2y(screen)
    end
    local screenCompressed = util.compressJPG(screen)
    local score = 0
    local countLifes = 0
    local levelBeatenStatus = 0
    local marioGameStatus = 0
    local playerX = 0
    local marioImage = 0
    local isLevelEnding = false
    local action = actions.createRandomAction()
    local reward = Reward.new(0, 0, 0, 0, 0, 0)
    local s = State.new(nil, screenCompressed, score, countLifes, levelBeatenStatus, marioGameStatus, playerX, marioImage, isLevelEnding, action, reward)
    s.isDummy = true
    return s
end


-------------------------
-- state history functions
-------------------------

-- Add a state to the history of recent states.
function states.addEntry(pState)
    table.insert(states.data, pState)
    table.insert(states.dataAll, pState)
    if #states.data > states.MAX_SIZE then
        table.remove(states.data, 1)
    end
    if #states.dataAll > states.MAX_SIZE then
        table.remove(states.dataAll, 1)
    end
end

-- Returns the last entry in the history.
function states.getLastEntry()
    if #states.data > 0 then
        return states.data[#states.data]
    else
        return nil
    end
end

-- Returns the state at the given index.
function states.getEntry(index)
    if index <= 0 then
        error("index<=0")
    elseif index > 0 then
        return states.data[index]
    elseif index < 0 then
        --print("#states.data", #states.data)
        --print("returning index ", #states.data - math.abs(index+1))
        --print("is nil:", states.data[#states.data - math.abs(index+1)] == nil)
        return states.data[#states.data - math.abs(index+1)]
    end
end

-- Returns the last N states.
function states.getLastEntries(nEntries)
    assert(#states.data >= nEntries, string.format("%d requested, %d available", nEntries, #states.data))
    local result = {}
    for i=1,nEntries do
        table.insert(result, states.data[#states.data-nEntries+i])
    end
    return result
end

-- Cascades a (direct) reward backwards through the history of recent states.
-- This is used to calculate observed future rewards.
-- This only cascades during the current run, i.e. stops at deaths/level-finishes.
-- @param reward Reward object.
function states.cascadeBackReward(reward)
    --print(string.format("cascading reward %.2f", rewards.getDirectReward(reward)))
    local epsilon = 0.0001
    local last = math.max(#states.data - 1, 1)
    local first = math.max(last - states.CASCADE_BACK + 1, 1)
    local exp = last - first + 1
    local direct = rewards.getDirectReward(reward)
    -- TODO start at the last state, move towards the first state, break if propagated reward is lower than epsilon,
    --      should be faster
    if direct > epsilon or direct < (-1)*epsilon then
        for i=first,last do
            local s = states.data[i]
            if not s.isDummy then
                local oldGamma = s.reward.observedGammaReward
                local cascadedGamma = torch.pow(GAMMA_OBSERVED, exp) * direct
                local newGamma = oldGamma + cascadedGamma
                --print(string.format("Cascade %.2f at i=%d/c=%d from %.4f to %.4f by %.4f", direct, i, exp, oldGamma, newGamma, cascadedGamma))
                s.reward.observedGammaReward = newGamma
            end
            exp = exp - 1
        end
    end
end

-- Adds the last states to the replay memory, if they haven't been added already.
function states.addToMemory()
    print("start addToMemory")
    -- -1 because the last state doesnt have a reward and action yet
    local statesTrain = {}
    local statesVal = {}
    for i=1,#states.data-1 do
        local state = states.data[i]
        local id = state.id

        -- Add some states to the validation set and others to the training set.
        local id1000 = id % 1000
        local val = false
        if (id1000 >= 750 and id1000 < 800) or (id1000 >= 950 and id1000 < 1000) then
            table.insert(statesVal, state)
        else
            table.insert(statesTrain, state)
        end
    end
    memory.addStates(statesTrain, false)
    memory.addStates(statesVal, true)
    print("end addToMemory")
end

-- Clear the history of recent states of the current run.
-- The states remain in the history states.dataAll.
-- Call this after a death or level-finish.
-- @param refill Whether to add dummy states to the history of the current run.
function states.clear(refill)
    states.data = {}
    if refill or refill == nil then
        states.fillWithEmptyStates()
    end
end

-- Adds dummy states to the history of the current run,
-- until a certain number of states is reached.
-- @param minNumberOfStates Fill until this number of states is reached
function states.fillWithEmptyStates(minNumberOfStates)
    minNumberOfStates = minNumberOfStates or (STATES_PER_EXAMPLE+1)
    -- TODO shouldn't there be a (minNumberOfStates-#states.data) here?
    for i=1,minNumberOfStates do
        states.addEntry(states.createEmptyState())
    end
end

-- Plot the rewards of the last N states.
function states.plotRewards(nBackMax)
    nBackMax = nBackmax or 200
    local points = {}
    for i=math.max(#states.dataAll-nBackMax, 1),#states.dataAll do
        local state = states.dataAll[i]
        local r = state.reward
        -- the newest added state should not have any reward yet
        if r == nil then
            table.insert(points, {i, 0, 0, 0, 0})
        else
            table.insert(points, {i, rewards.getDirectReward(r), r.observedGammaReward, r.expectedGammaReward, r.expectedGammaRewardRaw})
        end
    end
    display.plot(points, {win=21, labels={'State', 'Direct Reward', 'OGR', 'EGR (after gamma multiply)', 'EGR (before gamma multiply)'}, title='Reward per state (direct reward, observed/expected gamma reward)'})
end

--[=[
function states.stateChainToImage(stateChain, net)
    local batchSize = 1
    local lastState = stateChain[#stateChain]
    local previousStates = {}
    local previousScreens = {}
    for i=1,#stateChain-1 do
        table.insert(previousStates, stateChain[i])
        local screen = states.decompressScreen(stateChain[i].screen)
        local screenWithAction = torch.zeros(screen:size(1), screen:size(2)+16, screen:size(3))
        screenWithAction[{{1,screen:size(1)}, {1,screen:size(2)}, {1, screen:size(3)}}] = screen
        if screen:size(1) == 1 then
            screenWithAction = torch.repeatTensor(screenWithAction, 3, 1, 1)
        end
        local actionStr = actions.actionToString(stateChain[i].action)
        local x = 2
        local y = screen:size(2) + 2
        screenWithAction = image.drawText(screenWithAction, actionStr, x, y, {color={255,255,255}})
        screenWithAction = image.scale(screenWithAction, IMG_DIMENSIONS_Q_LAST[2], IMG_DIMENSIONS_Q_LAST[3])
        table.insert(previousScreens, screenWithAction)
    end

    local lastScreen = states.decompressScreen(lastState.screen)

    if net ~= nil then
        -- Get the transformation matrix from the AffineTransformMatrixGenerator
        local transfo = nil
        local function findTransformer(m)
            local name = torch.type(m)

            if name:find('AffineTransformMatrixGenerator') then
                transfo = m
            end
        end
        net:apply(findTransformer)

        if transfo ~= nil then
            transfo = transfo.output:float()

            local corners = torch.Tensor{{-1,-1,1},{-1,1,1},{1,-1,1},{1,1,1}}
            -- Compute the positions of the corners in the original image
            local points = torch.bmm(corners:repeatTensor(batchSize,1,1), transfo:transpose(2,3))
            -- Ensure these points are still in the image
            local imageSize = lastScreen:size(2)

            points = torch.floor((points+1)*imageSize/2)
            points:clamp(1,imageSize-1)

            for batch=1,batchSize do
                for pt=1,4 do
                    local point = points[batch][pt]
                    --print(string.format("p2 %.4f %.4f", point[1], point[2]))
                    for chan=1,IMG_DIMENSIONS_Q_LAST[1] do
                        local max_value = lastScreen[chan]:max()*1.1
                        -- We add 4 white pixels because one can disappear in image rescaling
                        lastScreen[chan][point[1]][point[2]] = max_value
                        lastScreen[chan][point[1]+1][point[2]] = max_value
                        lastScreen[chan][point[1]][point[2]+1] = max_value
                        lastScreen[chan][point[1]+1][point[2]+1] = max_value
                    end
                end
            end
        end
    end

    if lastScreen:size(1) == 1 then
        lastScreen = torch.repeatTensor(lastScreen, 3, 1, 1)
    end
    table.insert(previousScreens, lastScreen)

    local result = image.toDisplayTensor{input=previousScreens, nrow=#previousScreens, padding=1}

    return result
end
--]=]

-- Converts multiple state chains to an image showing what the network "sees" (i.e. receives).
-- If possible, it adds the area of focus of the spatial transformer.
-- However, that will only work if these exact state chains were the last ones forwarded through the network.
-- @param stateChains List of state chains to visualize.
-- @param net Network which to check for the spatial transformer.
-- @param startAt Index (1,N) of the state chain at which to start.
-- @param endAt Index (1,N) or the state chain at which to stop.
-- @returns Tensor (image)
function states.stateChainsToImage(stateChains, net, startAt, endAt)
    startAt = startAt or 1
    endAt = endAt or #stateChains
    local batchSize = endAt - startAt + 1
    local allScreens = {}

    for sc=startAt,endAt do
        local stateChain = stateChains[sc]
        local lastState = stateChain[#stateChain]
        local previousStates = {}
        local previousScreens = {}
        -- Collect previous state images (everything before the last state).
        -- They are upscaled to around the size of the last image.
        -- Additionally the get a text showing the chosen action (buttons).
        for i=1,#stateChain-1 do
            -- collect previous state's images
            table.insert(previousStates, stateChain[i])
            local screen = states.decompressScreen(stateChain[i].screen)
            local screenWithAction = torch.zeros(screen:size(1), screen:size(2)+16, screen:size(3)) -- leave some space below the image for the action text
            screenWithAction[{{1,screen:size(1)}, {1,screen:size(2)}, {1, screen:size(3)}}] = screen
            if screen:size(1) == 1 then
                screenWithAction = torch.repeatTensor(screenWithAction, 3, 1, 1)
            end

            -- add action text
            local actionStr = actions.actionToString(stateChain[i].action)
            local x = 2
            local y = screen:size(2) + 2
            screenWithAction = image.drawText(screenWithAction, actionStr, x, y, {color={255,255,255}})

            -- upscale to last state's image size
            screenWithAction = image.scale(screenWithAction, IMG_DIMENSIONS_Q_LAST[2], IMG_DIMENSIONS_Q_LAST[3])

            table.insert(previousScreens, screenWithAction)
        end

        local lastScreen = states.decompressScreen(lastState.screen)

        -- Mark Spatial Transformer's area of focus in the image.
        -- Code is a modified version of https://github.com/Moodstocks/gtsrb.torch/blob/master/plot.lua
        if net ~= nil then
            -- Get the transformation matrix from the AffineTransformMatrixGenerator
            local transfo = nil
            local function findTransformer(m)
                local name = torch.type(m)

                if name:find('AffineTransformMatrixGenerator') then
                    transfo = m
                end
            end
            net:apply(findTransformer)

            -- Stop if the network doesn't contain a spatial transformer
            if transfo ~= nil then
                transfo = transfo.output:float()
                transfo = transfo[{{sc}, {}, {}}] -- must have nDims=3, transfo[sc] reduces to nDims=2
                --print(transfo:size(1), transfo:size(2))

                --[[
                print("transfo size", transfo:size(1), transfo:size(2), transfo:size(3))
                print("Transformation matrix values:")
                for a=1,transfo:size(1) do
                    for b=1,transfo:size(2) do
                        for c=1,transfo:size(3) do
                            print(string.format("%d %d %d = %.4f", a, b, c, transfo[a][b][c]))
                        end
                    end
                end
                --]]

                local corners = torch.Tensor{{-1,-1,1},{-1,1,1},{1,-1,1},{1,1,1}}
                -- Compute the positions of the corners in the original image
                local points = torch.bmm(corners:repeatTensor(1,1,1), transfo:transpose(2,3))
                -- Ensure these points are still in the image
                local imageSize = lastScreen:size(2)

                --[[
                print("Corner points before fix:")
                for batch=1,batchSize do
                    for pt=1,4 do
                        local point = points[batch][pt]
                        print(string.format("(%.4f, %.4f)", point[1], point[2]))
                    end
                end
                --]]

                points = torch.floor((points+1)*imageSize/2)
                points:clamp(1,imageSize-1)

                --[[
                print("Corner points after fix:")
                for batch=1,batchSize do
                    for pt=1,4 do
                        local point = points[batch][pt]
                        print(string.format("(%.4f, %.4f)", point[1], point[2]))
                    end
                end
                --]]

                local batch = 1
                for pt=1,4 do
                    local point = points[batch][pt]
                    --print(string.format("p2 %.4f %.4f", point[1], point[2]))
                    for chan=1,IMG_DIMENSIONS_Q_LAST[1] do
                        local max_value = lastScreen[chan]:max()*1.1
                        -- We add 4 white pixels because one can disappear in image rescaling
                        lastScreen[chan][point[1]][point[2]] = max_value
                        lastScreen[chan][point[1]+1][point[2]] = max_value
                        lastScreen[chan][point[1]][point[2]+1] = max_value
                        lastScreen[chan][point[1]+1][point[2]+1] = max_value
                    end
                end
            end
        end

        if lastScreen:size(1) == 1 then
            lastScreen = torch.repeatTensor(lastScreen, 3, 1, 1)
        end

        for i=1,#previousScreens do
            table.insert(allScreens, previousScreens[i])
        end
        table.insert(allScreens, lastScreen)
    end

    local result = image.toDisplayTensor{input=allScreens, nrow=#stateChains[1], padding=1}

    return result
end

return states
