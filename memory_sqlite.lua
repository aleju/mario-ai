require 'torch'
require 'paths'
local driver = require 'luasql.sqlite3'

local env = assert(driver.sqlite3())
-- connect to data source
local con = assert(env:connect("learned/memory.sqlite"))

assert(con:execute[[
    CREATE TABLE IF NOT EXISTS states(
        id INTEGER(12) PRIMARY KEY,
        screen_jpg BLOB,
        score INTEGER(12),
        count_lifes INTEGER(6),
        level_beaten_status INTEGER(6),
        mario_game_status INTEGER(6),
        player_x INTEGER(6),
        mario_image INTEGER(6),
        is_level_ending INTEGER(1),
        action_arrow INTEGER(2),
        action_button INTEGER(2),
        reward_score_diff REAL,
        reward_x_diff REAL,
        reward_level_beaten REAL,
        reward_expected_gamma REAL,
        reward_expected_gamma_raw REAL,
        reward_observed_gamma REAL,
        is_dummy INTEGER(1),
        is_validation INTEGER(1)
    )
]])

--res = assert(con:execute[[
--    CREATE TABLE memories(
--        state_chain_id INT(12),
--        state_chain_pos INT(4),
--        state_id INT(12)
--    )
--]])

local memory = {}
memory.MEMORY_MAX_SIZE_TRAINING = 120000
memory.MEMORY_MAX_SIZE_VALIDATION = 10000
memory.MEMORY_TRAINING_MIN_SIZE = 100

function memory.load()
end

function memory.save()
end

function memory.getCountEntries(validation)
    assert(validation == true or validation == false)
    local val_int = 0
    if validation then val_int = 1 end
    local cur = assert(con:execute(string.format("SELECT COUNT(*) AS c FROM states WHERE is_validation = %d", val_int)))
    local row = cur:fetch({}, "a")
    local count = row.c
    return count
end

function memory.getCountAllEntries()
    local cur = assert(con:execute(string.format("SELECT COUNT(*) AS c FROM states")))
    local row = cur:fetch({}, "a")
    local count = row.c
    return count
end

function memory.getMaxStateId(defaultVal)
    if memory.getCountAllEntries() == 0 then
        return defaultVal
    else
        local cur = assert(con:execute("SELECT MAX(id) AS id FROM states"))
        local row = cur:fetch({}, "a")
        local id = row.id
        return id
    end
end

function memory.isTrainDataFull()
    return memory.getCountEntries(false) == memory.MEMORY_MAX_SIZE_TRAINING
end

function memory.reevaluateRewards()
end

function memory.reorderByDirectReward()
end

function memory.reorderBySurprise()
end

function memory.reorderBy(func)
end

function memory.addEntry(stateChain, nextState, validation)
    --[[
    assert(validation == true or validation == false)
    for i=1,#stateChain do
        memory.insertState(stateChain[i], validation, false)
    end
    memory.insertState(nextState, validation, false)

    if torch.rand() < 1/1000 do
        memory.reduceToMaxSizes()
    end
    --]]
end

function memory.addState(state, validation)
    assert(validation == true or validation == false)
    memory.insertState(state, validation, false)
    if math.random() < 1/1000 then
        print("Reducing to max size...")
        memory.reduceToMaxSizes()
    end
end

function memory.insertState(state, validation, updateIfExists)
    assert(state.action ~= nil)
    assert(state.reward ~= nil)
    --print("X")
    local ifExistsCommand = "IGNORE"
    if updateIfExists == true then ifExistsCommand = "UPDATE" end
    local screen_jpg_serialized = torch.serialize(state.screen, "ascii")
    local isLevelEnding_int = 0
    if state.isLevelEnding then isLevelEnding_int = 1 end
    local dummy_int = 0
    if state.isDummy then dummy_int = 1 end
    local val_int = 0
    if validation then val_int = 1 end

    --print("A")
    local query = string.format(
        [[
            INSERT OR %s INTO states (
                id,
                screen_jpg,
                score,
                count_lifes,
                level_beaten_status,
                mario_game_status,
                player_x,
                mario_image,
                is_level_ending,
                action_arrow,
                action_button,
                reward_score_diff,
                reward_x_diff,
                reward_level_beaten,
                reward_expected_gamma,
                reward_expected_gamma_raw,
                reward_observed_gamma,
                is_dummy,
                is_validation
            )
            VALUES
            (
                %d,
                '%s',
                %d,
                %d,
                %d,
                %d,
                %d,
                %d,
                %d,
                %d,
                %d,
                %.8f,
                %.8f,
                %.8f,
                %.8f,
                %.8f,
                %.8f,
                %d,
                %d
            )
        ]],
        ifExistsCommand,
        state.id,
        con:escape(screen_jpg_serialized),
        state.score,
        state.countLifes,
        state.levelBeatenStatus,
        state.marioGameStatus,
        state.playerX,
        state.marioImage,
        isLevelEnding_int,
        state.action.arrow,
        state.action.button,
        state.reward.scoreDiffReward,
        state.reward.xDiffReward,
        state.reward.levelBeatenReward,
        state.reward.expectedGammaReward,
        state.reward.expectedGammaRewardRaw,
        state.reward.observedGammaReward,
        dummy_int,
        val_int
    )
    --print("B")

    --local query2 = con:prepare(string.format(
        --[[
            INSERT OR %s states (
                id,
                screen_jpg,
                score,
                count_lifes,
                level_beaten_status,
                mario_game_status,
                player_x,
                mario_image,
                is_level_ending,
                action_arrow,
                action_button,
                reward_score_diff,
                reward_x_diff,
                reward_level_beaten,
                reward_expected_gamma,
                reward_expected_gamma_raw,
                reward_observed_gamma,
                is_dummy,
                is_validation
            )
            VALUES
            (
                ?,
                ?,
                ?,
                ?,
                ?,
                ?,
                ?,
                ?,
                ?,
                ?,
                ?,
                ?,
                ?,
                ?,
                ?,
                ?,
                ?,
                ?,
                ?
            )
        --]]--, ifExistsCommand)
    --)

    --[[
    query2:bind({"INTEGER", state.id})
    query2:bind({"BLOB", screen_jpg_serialized})
    query2:bind({"INTEGER", state.score})
    query2:bind({"INTEGER", state.countLifes})
    query2:bind({"INTEGER", state.levelBeatenStatus})
    query2:bind({"INTEGER", state.marioGameStatus})
    query2:bind({"INTEGER", state.playerX})
    query2:bind({"INTEGER", state.marioImage})
    query2:bind({"INTEGER", isLevelEnding_int})
    query2:bind({"INTEGER", state.action.arrow})
    query2:bind({"INTEGER", state.action.button})
    query2:bind({"REAL", state.reward.scoreDiffReward})
    query2:bind({"REAL", state.reward.xDiffReward})
    query2:bind({"REAL", state.reward.levelBeatenReward})
    query2:bind({"REAL", state.reward.expectedGammaReward})
    query2:bind({"REAL", state.reward.expectedGammaRewardRaw})
    query2:bind({"REAL", state.reward.observedGammaReward})
    query2:bind({"INTEGER", dummy_int})
    query2:bind({"INTEGER", val_int})
    --]]

    --print(query)
    --if STATS.ACTION_COUNTER < 10 then
        assert(con:execute(query))
    --end
    --print("C")
end

function memory.reduceToMaxSizes()
    local function deleteSubf(count, maxCount, validation)
        local val_int = 0
        if validation then val_int = 1 end
        local toDelete = count - maxCount
        print(string.format("count=%d, max count=%d, toDelete=%d, val_int=%d", count, maxCount, toDelete, val_int))
        if toDelete > 0 then
            local cur = assert(con:execute(string.format("SELECT id FROM states WHERE is_validation = %d ORDER BY id ASC LIMIT %d", val_int, toDelete)))
            local row = cur:fetch({}, "a")
            while row do
                assert(con:execute(string.format("DELETE FROM states WHERE id = %d", row.id)))
                row = cur:fetch(row, "a")
            end
        end
    end

    deleteSubf(memory.getCountEntries(false), memory.MEMORY_MAX_SIZE_TRAINING, false)
    deleteSubf(memory.getCountEntries(true), memory.MEMORY_MAX_SIZE_VALIDATION, true)
end

function memory.removeRandomEntry(validation)
end

function memory.removeRandomEntries(nRemove, validation, skew)
end

function memory.getRandomWeightedIndex(skew, validation, skewStrength)

end

function memory.getRandomStateChain(length, validation)
    assert(memory.getCountEntries(validation) >= length)
    local val_int = 0
    if validation then val_int = 1 end
    local cur = assert(con:execute(string.format("SELECT id FROM states WHERE is_validation = %d ORDER BY random() LIMIT 1", val_int)))
    local row = cur:fetch({}, "a")
    local id = row.id
    local indices = {}
    for i=1,length do table.insert(indices, id + i -1) end
    local states = memory.getStatesByIndices(indices)
    if #states < length then
        return memory.getRandomStateChain(length, validation)
    else
        return states
    end
end

function memory.getStatesByIndices(indices)
    local indicesStr = ""
    local indexToState = {}
    for i=1,#indices do
        if i == 1 then
            indicesStr = "" .. indices[i]
        else
            indicesStr = indicesStr .. ", " .. indices[i]
        end
        indexToState[indices[i]] = false
    end
    local cur = assert(con:execute(string.format("SELECT * FROM states WHERE id IN (%s)", indicesStr)))
    local row = cur:fetch({}, "a")
    while row do
        local state = memory.rowToState(row)
        assert(indexToState[state.id] ~= nil)
        indexToState[state.id] = state
        row = cur:fetch(row, "a")
    end
    local states = {}
    for k,v in pairs(indexToState) do
        table.insert(states, v)
    end
    return states
end

function memory.getTrainingBatch(batchSize)
    return memory.getBatch(batchSize, false)
end

function memory.getValidationBatch(batchSize)
    return memory.getBatch(batchSize, true)
end

function memory.getBatch(batchSize, validation, reevaluate)
    assert(validation == true or validation == false)
    assert(reevaluate == true or reevaluate == false)
    local stateChains = {}
    local length = STATES_PER_EXAMPLE
    if reevaluate then length = length + 1 end
    for i=1,batchSize do
        --local idx = memory.getRandomWeightedIndex("top", validation)
        local stateChain = memory.getRandomStateChain(length, validation)
        table.insert(stateChains, stateChain)
    end

    if not reevaluate then
        local batchInput, batchTarget = network.stateChainsToBatch(stateChains)
        return batchInput, batchTarget
    else
        local stateChainsSlim = {}
        local stateChainsNext = {}
        for i=1,#stateChain do
            local stateChain = stateChains[i]
            local stateChainCurrent = {}
            local stateChainNext = {}
            for j=1,length-1 do
                table.insert(stateChainCurrent, stateChain[j])
                table.insert(stateChainNext, stateChain[j+1])
            end
            table.insert(stateChainsCurrent, stateChainCurrent)
            table.insert(stateChainsNext, stateChainNext)
        end
        local bestActions = network.approximateBestActionsBatch(stateChainsNext)
        for i=1,#stateChains do
            local oldReward = stateChainsCurrent[i][length].reward
            local newReward = rewards.statesToReward(stateChainsNext[i], bestActions[i].action, bestActions[i].value)
            newReward.observedGammaReward = oldReward.observedGammaReward
            stateChainsCurrent[i][length].reward = newReward
        end

        local batchInput, batchTarget = network.stateChainsToBatch(stateChainsCurrent)
        return batchInput, batchTarget, stateChainsCurrent
    end
end

function memory.rowToState(row)
    local screen_jpg = torch.deserialize(row.screen_jpg, "ascii")
    local isDummy = false
    if row.is_dummy == 1 then isDummy = true end
    local action = Action.new(row.action_arrow, row.action_button)
    local reward = Reward.new(row.reward_score_diff, row.reward_x_diff, row.reward_level_beaten, row.reward_expected_gamma, row.reward_expected_gamma_raw, row.reward_observed_gamma)
    local state = State.new(row.id, row.score, row.count_lifes, row.level_beaten_status, row.mario_game_status, row.player_x, row.mario_image, row.is_level_ending, action, reward)
    state.isDummy = isDummy
    return state
end

function memory.plot(subsampling)

end

return memory
