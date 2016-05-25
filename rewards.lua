-- Functions to deal with rewards.

local rewards = {}

-------------------------
-- Reward object member functions
-------------------------

-- For a reward, calculate the sum of direct reward + observed future rewards.
function rewards.getSumObserved(reward)
    return rewards.getDirectReward(reward) + reward.observedGammaReward
end

-- For a reward, calculate the sum of direct reward + discounted predicted future rewards.
function rewards.getSumExpected(reward)
    return rewards.getDirectReward(reward) + reward.expectedGammaReward
end

-- For a reward, get the reward sum to use for training.
function rewards.getSumForTraining(reward)
    return rewards.getSumExpected(reward)
end

-- For a reward, get the directly received reward (no future rewards).
function rewards.getDirectReward(reward)
    return reward.scoreDiffReward + reward.xDiffReward + reward.levelBeatenReward
end

-- For a reward, get the absolute difference between predicted future rewards and observed future rewards.
function rewards.getSurprise(reward)
    return math.abs(reward.expectedGammaReward - reward.observedGammaReward)
end


-------------------------
-- General reward functions
-------------------------

-- Calculate the reward for a given chain of states.
-- If the chain contains states (i, i+1, ..., i+N-1, i+N), the reward is calculated for state i+N-1.
-- @param stateChain The list of consecutive states.
-- @param bestAction Action object, optional.
-- @param bestActionValue Float value, optional.
-- @returns Reward
function rewards.statesToReward(stateChain, bestAction, bestActionValue)
    assert(#stateChain == STATES_PER_EXAMPLE)
    local state1 = stateChain[#stateChain-1]
    local state2 = stateChain[#stateChain]

    local scoreDiffReward = 0
    local xDiffReward = 0
    local levelBeatenReward = 0
    local exepctedGammaRewardRaw = 0
    local expectedGammaReward = 0

    -- No rewards for dummy states.
    if not state1.isDummy and not state2.isDummy then
        if state2.levelBeatenStatus ~= 128 then
            -- Reward if score increased.
            if state2.score > state1.score then
                scoreDiffReward = 1.0
            end

            -- Reward for moving to the right, more if faster.
            -- Negative reward for moving to the left.
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
        elseif state2.levelBeatenStatus == 128 and state1.levelBeatenStatus == 0 then
            -- Negative reward when Mario has died.
            if state2.countLifes < state1.countLifes then
                levelBeatenReward = -3.0
            end
        end

        -- Reward for finishing the level.
        if state2.isLevelEnding then
            levelBeatenReward = 2.0
        end

        -- Reward for Mario dying (death animation playing).
        if state1.marioImage == 62 then
            levelBeatenReward = -3.0
        end
    end

    -- Re-approximate the Q-value of that state chain, if necessary.
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

    -- Clamp/truncate predicted future reward values.
    expectedGammaRewardRaw = bestActionValue
    if expectedGammaRewardRaw > MAX_GAMMA_REWARD then
        expectedGammaRewardRaw = MAX_GAMMA_REWARD
    elseif expectedGammaRewardRaw < (-1)*MAX_GAMMA_REWARD then
        expectedGammaRewardRaw = (-1)*MAX_GAMMA_REWARD
    end

    expectedGammaReward = GAMMA_EXPECTED * expectedGammaRewardRaw

    local rew = Reward.new(scoreDiffReward, xDiffReward, levelBeatenReward, expectedGammaReward, expectedGammaRewardRaw)

    return rew, bestAction, bestActionValue
end

return rewards
