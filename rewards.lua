local rewards = {}

function rewards.getSumObserved(reward)
    --print(string.format("osg: %.4f", reward.observedGammaReward))
    return rewards.getDirectReward(reward) + reward.observedGammaReward
end

function rewards.getSumExpected(reward)
    return rewards.getDirectReward(reward) + reward.expectedGammaReward
end

function rewards.getSumForTraining(reward)
    --return rewards.getDirectReward(reward) + 0.75 * reward.expectedGammaReward + 0.25 * reward.observedGammaReward
    --return rewards.getDirectReward(reward) + reward.expectedGammaReward
    --return rewards.getSumObserved(reward)
    return rewards.getSumExpected(reward)
end

function rewards.getDirectReward(reward)
    return reward.scoreDiffReward + reward.xDiffReward + reward.levelBeatenReward
end

function rewards.getSurprise(reward)
    return math.abs(reward.expectedGammaReward - reward.observedGammaReward)
end

--[[
function rewards.statesToReward(state1, state2)
    assert(state1 ~= nil and state2 ~= nil)

    local scoreDiffReward = 0
    local xDiffReward = 0
    local levelBeatenReward = 0
    local expectedGammaReward = 0

    if levelBeatenStatus ~= 128 then
        scoreDiffReward = state2.score - state1.score
        --scoreDiffReward = scoreDiffReward / 2

        if state2.playerX > state1.playerX then
            xDiffReward = 1.0
        else
            xDiffReward = -1.5
        end
    end

    local bestAction, bestActionValue = network.approximateBestAction(state1, state2)
    -- no NaN, no INF
    if bestActionValue ~= bestActionValue or bestActionValue == math.isinf then
        print("[WARNING] Encountered NaN or INF when approximating best action.")
        bestActionValue = 0
    end
    expectedGammaReward = GAMMA_EXPECTED * bestActionValue

    if state2.levelBeatenStatus == 128 and state1.levelBeatenStatus == 0 then
        if state2.countLifes < state1.countLifes then
            levelBeatenReward = -1000
        else
            levelBeatenReward = 10000
        end
    end

    scoreDiffReward = math.min(scoreDiffReward, 100)

    local rew = Reward.new(scoreDiffReward / 100, xDiffReward / 100, levelBeatenReward / 100, expectedGammaReward)

    return rew, bestAction, bestActionValue
end
--]]

function rewards.statesToReward(stateChain, bestAction, bestActionValue)
    return rewards.statesToReward3(stateChain, bestAction, bestActionValue)
end

function rewards.statesToReward1(stateChain, bestAction, bestActionValue)
    assert(#stateChain == STATES_PER_EXAMPLE)
    local state1 = stateChain[#stateChain-1]
    local state2 = stateChain[#stateChain]

    local scoreDiffReward = 0
    local xDiffReward = 0
    local levelBeatenReward = 0
    local exepctedGammaRewardRaw = 0
    local expectedGammaReward = 0

    --print(string.format("[REWARD] Score before: %.4f, score after: %.4f", state1.score, state2.score))

    -- dont give points right after the start (dummy states have score 0)
    --print("dummy: ", state1.isDummy, state2.isDummy)
    --print("lbs: ", state1.levelBeatenStatus, state2.levelBeatenStatus)
    --print(string.format("lifes: %d, %d", state1.countLifes, state2.countLifes))
    --print(string.format("scores: %d %d", state1.score, state2.score))
    --print(string.format("mgs: %d, %d", state1.marioGameStatus, state2.marioGameStatus))
    --print(string.format("x: %d, %d", state1.playerX, state2.playerX))

    if not state1.isDummy and not state2.isDummy then
        if state2.levelBeatenStatus ~= 128 then
            if state2.score > state1.score then
                scoreDiffReward = 1.0
            end
            --scoreDiffReward = state2.score - state1.score
            --scoreDiffReward = scoreDiffReward / 2
            --if scoreDiffReward > 0 then print(string.format("[REWARD] score diff %.4f", scoreDiffReward)) end

            --[[
            local xDiff = state2.playerX - state1.playerX
            if xDiff > 0 then
                xDiffReward = math.min(xDiff/100, 0.1)
            else
                xDiffReward = -0.1
            end
            --]]
            --local xDiff = state2.playerX - state1.playerX
            --xDiffReward = math.max(math.min(xDiff/100, 0.1), 0)
            local xDiff = state2.playerX - state1.playerX
            if xDiff > 0 then
                xDiffReward = 1.0
            else
                xDiffReward = -1.0
            end

            --[[
            local xDiff = state2.playerX - state1.playerX
            if xDiff <= 0 then
                xDiffReward = math.min(xDiff/100, 0.1)
            else
                xDiffReward = -0.1
            end
            --]]
        elseif state2.levelBeatenStatus == 128 and state1.levelBeatenStatus == 0 then
            if state2.countLifes < state1.countLifes then
                --levelBeatenReward = -2
                levelBeatenReward = -1.0
            else
                -- FIXME this reward doesnt seem to work, though the score increases by finishing the level anyways
                levelBeatenReward = 2
            end
        end

        --[[
        if state1.marioImage ~= 62 and state2.marioImage == 62 then
            levelBeatenReward = -2
        end
        --]]
        if state1.marioImage == 62 then
            levelBeatenReward = -1.0
        end
    end

    if bestAction ~= nil then
        if bestActionValue == nil then
            bestActionValue = network.approximateActionValue(stateChain, bestActionIdx)
        end
    else
        bestAction, bestActionValue = network.approximateBestAction(stateChain)
    end

    -- no NaN, no INF
    if bestActionValue == nil then
        print("[WARNING] bestActionValue was nil")
        bestActionValue = 0
    elseif bestActionValue ~= bestActionValue or bestActionValue == math.isinf then
        print("[WARNING] Encountered NaN or INF when approximating best action.")
        bestActionValue = 0
    end
    expectedGammaRewardRaw = bestActionValue
    expectedGammaReward = GAMMA_EXPECTED * expectedGammaRewardRaw

    --scoreDiffReward = math.min(scoreDiffReward, 1000)

    local rew = Reward.new(scoreDiffReward, xDiffReward, levelBeatenReward, expectedGammaReward, expectedGammaRewardRaw)

    return rew, bestAction, bestActionValue
end

function rewards.statesToReward2(stateChain, bestAction, bestActionValue)
    assert(#stateChain == STATES_PER_EXAMPLE)
    local state1 = stateChain[#stateChain-1]
    local state2 = stateChain[#stateChain]

    local scoreDiffReward = 0
    local xDiffReward = 0
    local levelBeatenReward = 0
    local exepctedGammaRewardRaw = 0
    local expectedGammaReward = 0

    --print(string.format("[REWARD] Score before: %.4f, score after: %.4f", state1.score, state2.score))

    -- dont give points right after the start (dummy states have score 0)
    --print("dummy: ", state1.isDummy, state2.isDummy)
    --print("lbs: ", state1.levelBeatenStatus, state2.levelBeatenStatus)
    --print(string.format("lifes: %d, %d", state1.countLifes, state2.countLifes))
    --print(string.format("scores: %d %d", state1.score, state2.score))
    --print(string.format("mgs: %d, %d", state1.marioGameStatus, state2.marioGameStatus))
    --print(string.format("x: %d, %d", state1.playerX, state2.playerX))

    if not state1.isDummy and not state2.isDummy then
        if state2.levelBeatenStatus ~= 128 then
            if state2.score > state1.score then
                scoreDiffReward = 2.5
            end

            -- roughly set so that running fast towards the right is around +1.5 reward
            local xDiff = state2.playerX - state1.playerX
            xDiffReward = xDiff/10
            if xDiff > 0 then
                xDiffReward = math.min(xDiffReward, 1.5)
                xDiffReward = xDiffReward + 0.1
            elseif xDiff < 0 then
                xDiffReward = math.max(xDiffReward, -2.5)
                xDiffReward = 1.25*xDiffReward - 0.25
            end

            --[[
            local xDiff = state2.playerX - state1.playerX
            if xDiff <= 0 then
                xDiffReward = math.min(xDiff/100, 0.1)
            else
                xDiffReward = -0.1
            end
            --]]
        elseif state2.levelBeatenStatus == 128 and state1.levelBeatenStatus == 0 then
            if state2.countLifes < state1.countLifes then
                levelBeatenReward = -3.0
            end
        end

        if state2.isLevelEnding then
            levelBeatenReward = 2.0
        end

        --[[
        if state1.marioImage ~= 62 and state2.marioImage == 62 then
            levelBeatenReward = -2
        end
        --]]
        if state1.marioImage == 62 then
            levelBeatenReward = -3.0
        end
    end

    if bestAction ~= nil then
        if bestActionValue == nil then
            bestActionValue = network.approximateActionValue(stateChain, bestActionIdx)
        end
    else
        bestAction, bestActionValue = network.approximateBestAction(stateChain)
    end

    -- no NaN, no INF
    if bestActionValue == nil then
        print("[WARNING] bestActionValue was nil")
        bestActionValue = 0
    elseif bestActionValue ~= bestActionValue or bestActionValue == math.isinf then
        print("[WARNING] Encountered NaN or INF when approximating best action.")
        bestActionValue = 0
    end
    expectedGammaRewardRaw = bestActionValue
    expectedGammaReward = GAMMA_EXPECTED * expectedGammaRewardRaw

    --scoreDiffReward = math.min(scoreDiffReward, 1000)

    local rew = Reward.new(scoreDiffReward, xDiffReward, levelBeatenReward, expectedGammaReward, expectedGammaRewardRaw)

    return rew, bestAction, bestActionValue
end

function rewards.statesToReward3(stateChain, bestAction, bestActionValue)
    assert(#stateChain == STATES_PER_EXAMPLE)
    local state1 = stateChain[#stateChain-1]
    local state2 = stateChain[#stateChain]

    local scoreDiffReward = 0
    local xDiffReward = 0
    local levelBeatenReward = 0
    local exepctedGammaRewardRaw = 0
    local expectedGammaReward = 0

    --print(string.format("[REWARD] Score before: %.4f, score after: %.4f", state1.score, state2.score))

    -- dont give points right after the start (dummy states have score 0)
    --print("dummy: ", state1.isDummy, state2.isDummy)
    --print("lbs: ", state1.levelBeatenStatus, state2.levelBeatenStatus)
    --print(string.format("lifes: %d, %d", state1.countLifes, state2.countLifes))
    --print(string.format("scores: %d %d", state1.score, state2.score))
    --print(string.format("mgs: %d, %d", state1.marioGameStatus, state2.marioGameStatus))
    --print(string.format("x: %d, %d", state1.playerX, state2.playerX))

    if not state1.isDummy and not state2.isDummy then
        if state2.levelBeatenStatus ~= 128 then
            if state2.score > state1.score then
                scoreDiffReward = 1.0
            end

            -- roughly set so that running fast towards the right is around +1.5 reward
            local xDiff = state2.playerX - state1.playerX
            if xDiff >= 8 then
                xDiffReward = 1.0
            elseif xDiff > 0 then
                xDiffReward = 0.5
            elseif xDiff > -8 then
                xDiffReward = -1.0
            else
                xDiffReward = -1.5
            end

            --[[
            local xDiff = state2.playerX - state1.playerX
            if xDiff <= 0 then
                xDiffReward = math.min(xDiff/100, 0.1)
            else
                xDiffReward = -0.1
            end
            --]]
        elseif state2.levelBeatenStatus == 128 and state1.levelBeatenStatus == 0 then
            if state2.countLifes < state1.countLifes then
                levelBeatenReward = -3.0
            end
        end

        if state2.isLevelEnding then
            levelBeatenReward = 2.0
        end

        --[[
        if state1.marioImage ~= 62 and state2.marioImage == 62 then
            levelBeatenReward = -2
        end
        --]]
        if state1.marioImage == 62 then
            levelBeatenReward = -3.0
        end
    end

    if bestAction ~= nil then
        if bestActionValue == nil then
            bestActionValue = network.approximateActionValue(stateChain, bestActionIdx)
        end
    else
        bestAction, bestActionValue = network.approximateBestAction(stateChain)
    end

    -- no NaN, no INF
    if bestActionValue == nil then
        print("[WARNING] bestActionValue was nil")
        bestActionValue = 0
    elseif bestActionValue ~= bestActionValue or bestActionValue == math.isinf then
        print("[WARNING] Encountered NaN or INF when approximating best action.")
        bestActionValue = 0
    end

    expectedGammaRewardRaw = bestActionValue
    if expectedGammaRewardRaw > MAX_GAMMA_REWARD then
        expectedGammaRewardRaw = MAX_GAMMA_REWARD
    elseif expectedGammaRewardRaw < (-1)*MAX_GAMMA_REWARD then
        expectedGammaRewardRaw = (-1)*MAX_GAMMA_REWARD
    end

    expectedGammaReward = GAMMA_EXPECTED * expectedGammaRewardRaw

    --scoreDiffReward = math.min(scoreDiffReward, 1000)

    local rew = Reward.new(scoreDiffReward, xDiffReward, levelBeatenReward, expectedGammaReward, expectedGammaRewardRaw)

    return rew, bestAction, bestActionValue
end

return rewards
