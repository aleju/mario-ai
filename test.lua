print("------------------------")
print("TESTING")
print("------------------------")

paths.dofile('config.lua')

print("Loading network...")
Q = network.load()
assert(Q ~= nil)

-- count parameters
--print("Number of parameters in AE:", network.getNumberOfParameters(MODEL_AE))
print("Number of parameters in Q:", network.getNumberOfParameters(Q))

print("Loading memory...")
memory.load()

print("Loading stats...")
util.loadStats()
STATS.P_EXPLORE_CURRENT = 0.0

print("Starting loop.")

function on_frame_emulated()
    local lastLastState = states.getEntry(-2)
    local lastState = states.getEntry(-1)
    STATS.FRAME_COUNTER = movie.currentframe()

    if STATS.FRAME_COUNTER % REACT_EVERY_NTH_FRAME ~= 0 then
        return
    end

    STATS.ACTION_COUNTER = STATS.ACTION_COUNTER + 1

    local state = State.new(nil, util.getScreenCompressed(), util.getCurrentScore(), util.getCountLifes(), util.getLevelBeatenStatus(), util.getMarioGameStatus(), util.getPlayerX(), util.getMarioImage(), util.isLevelEnding())
    states.addEntry(state) -- getLastEntries() depends on this, don't move it after the next code block
    --print("Score:", score, "Level:", util.getLevel(), "x:", playerX, "status:", marioGameStatus, "levelBeatenStatus:", levelBeatenStatus, "count lifes:", countLifes, "Mario Image", util.getMarioImage())

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
    if STATS.ACTION_COUNTER % 1 == 0 then
        states.plotRewards()
    end

    --------------------

    state.action = actions.chooseAction(states.getLastEntries(STATES_PER_EXAMPLE), false, bestAction)

    local levelEnded = state.levelBeatenStatus == 128 or state.marioGameStatus == 2
    if levelEnded then
        print("Reloading saved gamestate...")
        states.clear()

        -- Reload save state if level was beaten or mario died
        util.loadRandomTrainingSaveState()
        LAST_SAVE_STATE_LOAD = STATS.ACTION_COUNTER
    else
        actions.endAllActions()
        actions.startAction(state.action)
    end
end

actions.endAllActions()
util.loadRandomTrainingSaveState()
util.setGameSpeedToVeryFast()
states.fillWithEmptyStates()
gui.repaint()
