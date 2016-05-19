local states = {
    data = {},
    dataAll = {}
}

states.MAX_SIZE = 5000
states.CASCADE_BACK = 100
--states.CASCADE_STOP = 0.05
--states.CASCADE_INFLUENCE = 0.25
--states.idx = 0

--[[
function states.addEntry(screen, action, bestActionValue, score, playerX, countLifes, levelBeatenStatus, marioGameStatus)
    states.idx = states.idx + 1
    local entry = {
        idx=states.idx,
        screen=screen,
        action=action,
        bestActionValue=bestActionValue,
        score=score,
        playerX=playerX,
        countLifes=countLifes,
        levelBeatenStatus=levelBeatenStatus,
        marioGameStatus=marioGameStatus,
        reward=nil
    }
    table.insert(states.data, entry)
    --print("Added entry ", entry.idx, " to states, now", #states.data)
    --print("First entry idx:", states.data[1].idx)
    --print("Last entry idx:", states.data[#states.data].idx)
    if #states.data > states.MAX_SIZE then
        table.remove(states.data, 1)
        --print("Removed entry from states, now", #states.data)
    end
    return entry
end
--]]
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

--[[
function states.setLastReward(reward)
    local entry = states.getLastEntry()
    if entry == nil then
        print("[INFO] could not set reward of last state, because there is no last state")
    else
        --print("Setting reward of ", entry.idx, " to ", reward)
        entry.reward = reward
    end
end
--]]

function states.getLastEntry()
    if #states.data > 0 then
        return states.data[#states.data]
    else
        return nil
    end
end

function states.getEntry(index)
    if index == 0 then
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

function states.getLastEntries(nEntries)
    assert(#states.data >= nEntries, string.format("%d requested, %d available", nEntries, #states.data))
    local result = {}
    for i=1,nEntries do
        table.insert(result, states.data[#states.data-nEntries+i])
    end
    return result
end

--[[
function states.cascadeBackReward(reward)
    --print(string.format("cascading reward %.2f", rewards.getDirectReward(reward)))
    local last = math.max(#states.data - 1, 1)
    local first = math.max(last - states.CASCADE_BACK + 1, 1)
    local counter = last - first
    local direct = rewards.getDirectReward(reward)
    for i=first,last do
        local s = states.data[i]
        if not s.isDummy then
            local oldGamma = s.reward.observedGammaReward
            local cascadedGamma = torch.pow(GAMMA_OBSERVED, counter+1) * direct
            --local cascadedGamma = GAMMA_OBSERVED * 1/(counter+1) * direct
            --local newGamma = states.CASCADE_INFLUENCE * (cascadedGamma) + (1 - states.CASCADE_INFLUENCE) * oldGamma
            --local newGamma = oldGamma + cascadedGamma
            local newGamma = oldGamma + cascadedGamma
            print(string.format("Cascade %.2f at i=%d/c=%d from %.4f to %.4f by %.4f", direct, i, counter, oldGamma, newGamma, cascadedGamma))
            s.reward.observedGammaReward = newGamma
            counter = counter - 1
            --if math.abs(cascadedGamma) < states.CASCADE_STOP then
            --    break
            --end
        end
    end
end
--]]

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
                --local cascadedGamma = GAMMA_OBSERVED * 1/(counter+1) * direct
                --local newGamma = states.CASCADE_INFLUENCE * (cascadedGamma) + (1 - states.CASCADE_INFLUENCE) * oldGamma
                --local newGamma = oldGamma + cascadedGamma
                local newGamma = oldGamma + cascadedGamma
                --print(string.format("Cascade %.2f at i=%d/c=%d from %.4f to %.4f by %.4f", direct, i, exp, oldGamma, newGamma, cascadedGamma))
                s.reward.observedGammaReward = newGamma
                --if math.abs(cascadedGamma) < states.CASCADE_STOP then
                --    break
                --end
            end
            exp = exp - 1
        end
    end
end

function states.decompressScreen(screen)
    return util.decompressJPG(screen)
end

function states.addToMemory()
    print("start addToMemory")
    -- -1 because the last state doesnt have a reward and action yet
    for i=1,#states.data-1 do
        local state = states.data[i]
        local id = state.id
        local id1000 = id % 1000
        local val = false
        if (id1000 >= 750 and id1000 < 800) or (id1000 >= 950 and id1000 < 1000) then
            val = true
        end
        memory.addState(state, val)
    end
    print("end addToMemory")
end

function states.clear(refill)
    states.data = {}
    if refill or refill == nil then
        states.fillWithEmptyStates()
    end
end

function states.fillWithEmptyStates(minNumberOfStates)
    minNumberOfStates = minNumberOfStates or (STATES_PER_EXAMPLE+1)
    for i=1,minNumberOfStates do
        --table.insert(states.data, states.createEmptyState())
        states.addEntry(states.createEmptyState())
    end
end

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

--[[
function states.plotRewards()
    local points = {}
    for i=1,#states.dataAll do
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
--]]

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

--[[
function states.plotStateChain(stateChain, windowId, title, width)
    windowId = windowId or 20
    title = title or "State Chain"

    local imgsDisp = torch.zeros(#stateChain, IMG_DIMENSIONS_Q_HISTORY[1], IMG_DIMENSIONS_Q_HISTORY[2], IMG_DIMENSIONS_Q_HISTORY[3])
    for i=1,#stateChain do
        imgsDisp[i] = states.decompressScreen(stateChain[i].screen)
    end

    local out = image.toDisplayTensor{input=imgsDisp, nrow=#stateChain, padding=1}

    if width then
        display.image(out, {win=windowId, width=width, title=title})
    else
        display.image(out, {win=windowId, title=title})
    end
end
--]]

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
            local points = torch.bmm(corners:repeatTensor(batchSize,1,1), transfo:transpose(2,3))
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
    --local result = image.toDisplayTensor{input={lastScreen}, nrow=1, padding=1}
    --[[
    local c = math.max(IMG_DIMENSIONS_Q_LAST[1], IMG_DIMENSIONS_Q_HISTORY[1])
    local h = math.max(prev:size(3)+16, IMG_DIMENSIONS_Q_LAST[2])
    local w = prev:size(2)+IMG_DIMENSIONS_Q_LAST[3]
    local result = torch.zeros(c, h, w)
    result[{{1,c}, {1,prev:size(2)}, {1,prev:size(3)}}] = prev
    for i=1,#previousStates do
        image.draw()
    end
    --]]

    return result
end

return states
