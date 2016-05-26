print("------------------------")
print("TESTING")
print("------------------------")

require 'paths'
paths.dofile('config.lua')
ADD_STATE_EVERY_NTH_FRAME = REACT_EVERY_NTH_FRAME
--REACT_EVERY_NTH_FRAME = 1

VIDEO = true

print("Loading network...")
Q = network.load("learned/q11b-network.th7")
assert(Q ~= nil)
Q:evaluate()

-- count parameters
--print("Number of parameters in AE:", network.getNumberOfParameters(MODEL_AE))
print("Number of parameters in Q:", network.getNumberOfParameters(Q))

print("Loading memory...")
memory.load()

print("Loading stats...")
util.loadStats()
STATS.P_EXPLORE_CURRENT = 0.25 -- epsilon in epsilon-greedy policy. Introducing a medium amount of noise here seems to help the agent a bit.

print("Starting loop.")

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
    local lastEntries = states.getLastEntries(STATES_PER_EXAMPLE-1)
    table.insert(lastEntries, state)
    if STATS.FRAME_COUNTER % ADD_STATE_EVERY_NTH_FRAME == 0 then
        states.addEntry(state) -- getLastEntries() depends on this, don't move it after the next code block
        --print("Score:", score, "Level:", util.getLevel(), "x:", playerX, "status:", marioGameStatus, "levelBeatenStatus:", levelBeatenStatus, "count lifes:", countLifes, "Mario Image", util.getMarioImage())

        -- Calculate reward
        local rew, bestAction, bestActionValue = rewards.statesToReward(lastEntries)
        lastState.reward = rew
        --print(string.format("[Reward] R=%.2f DR=%.2f SDR=%.2f XDR=%.2f LBR=%.2f EGR=%.2f", rewards.getSumExpected(lastState.reward), rewards.getDirectReward(lastState.reward), lastState.reward.scoreDiffReward, lastState.reward.xDiffReward, lastState.reward.levelBeatenReward, lastState.reward.expectedGammaReward))
        states.cascadeBackReward(lastState.reward)
    end

    -- plot rewards of last states
    if STATS.ACTION_COUNTER % 1 == 0 then
        if VIDEO then
            states.plotRewardsVideo(100)
        else
            states.plotRewards()
        end
    end

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

    --------------------

    state.action = actions.chooseAction(lastEntries, false, bestAction)

    local levelEnded = state.levelBeatenStatus == 128 or state.marioGameStatus == 2
    if levelEnded then
        print("Reloading saved gamestate...")
        states.clear()

        -- Reload save state if level was beaten or mario died
        util.loadRandomTestSaveState()
        LAST_SAVE_STATE_LOAD = STATS.ACTION_COUNTER
    else
        actions.endAllActions()
        actions.startAction(state.action)
    end

    -- show state chain
    display.image(states.stateChainsToImage({lastEntries}, Q), {win=17, title="What the model sees (White area is focused by spatial transformer)"})
end

actions.endAllActions()
util.loadRandomTestSaveState()
util.setGameSpeedToNormal()
states.fillWithEmptyStates()
gui.repaint()
