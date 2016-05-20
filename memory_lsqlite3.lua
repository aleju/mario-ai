require 'torch'
require 'paths'
local driver = require 'lsqlite3'

--local db = sqlite3.open_memory()
local db = sqlite3.open("learned/memory.sqlite")

assert(db:exec[[
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

local memory = {}
memory.MEMORY_MAX_SIZE_TRAINING = 250000
memory.MEMORY_MAX_SIZE_VALIDATION = 25000
memory.MEMORY_TRAINING_MIN_SIZE = 100

memory.cache = {}

function memory.resetCache()
    memory.cache = {
        count_train = nil,
        count_val = nil,
        state_ids_train = nil,
        state_ids_val = nil
    }
end
memory.resetCache()

function memory.load()
end

function memory.save()
end

function memory.getCountEntries(validation)
    assert(validation == true or validation == false)
    local val_int = 0
    if validation then val_int = 1 end
    for row in db:nrows(string.format("SELECT COUNT(*) AS c FROM states WHERE is_validation = %d", val_int)) do
        return row.c
    end
end

function memory.getCountEntriesCached(validation)
    assert(validation == true or validation == false)
    if validation then
        if memory.cache.count_val == nil then
            memory.cache.count_val = memory.getCountEntries(validation)
        end
        return memory.cache.count_val
    else
        if memory.cache.count_train == nil then
            memory.cache.count_train = memory.getCountEntries(validation)
        end
        return memory.cache.count_train
    end
end

function memory.getCountAllEntries()
    for row in db:nrows(string.format("SELECT COUNT(*) AS c FROM states")) do
        return row.c
    end
end

function memory.getMaxStateId(defaultVal)
    if memory.getCountAllEntries() == 0 then
        return defaultVal
    else
        for row in db:nrows(string.format("SELECT MAX(id) AS id FROM states")) do
            return row.id
        end
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
    memory.addStates({state}, validation)
end

function memory.addStates(states, validation)
    assert(validation == true or validation == false)
    memory.insertStatesBatched(states, validation, false)
    if math.random() < 1/100 then
        print("Reducing to max size...")
        memory.reduceToMaxSizes()
    end
end

function memory.insertState(state, validation, updateIfExists)
    memory.insertStates({state}, validation, updateIfExists)
end

function memory.insertStates(states, validation, updateIfExists)
    memory.resetCache()

    local ifExistsCommand = "IGNORE"
    if updateIfExists == true then ifExistsCommand = "UPDATE" end
    local val_int = 0
    if validation then val_int = 1 end

    --print("A")
    local stmt = db:prepare(
        string.format([[
            INSERT OR %s INTO states (
                screen_jpg,
                id,
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
                :screenjpg,
                :id,
                :score,
                :countlifes,
                :levelbeatenstatus,
                :mariogamestatus,
                :playerx,
                :marioimage,
                :islevelending,
                :actionarrow,
                :actionbutton,
                :rewardscorediff,
                :rewardxdiff,
                :rewardlevelbeaten,
                :rewardexpectedgamma,
                :rewardexpectedgammaraw,
                :rewardobservedgamma,
                :isdummy,
                :isvalidation
            )
        ]], ifExistsCommand)
    )

    for i=1,#states do
        local state = states[i]

        assert(state.action ~= nil)
        assert(state.reward ~= nil)
        --print("X")

        local screen_jpg_serialized = torch.serialize(state.screen, "ascii")
        --print("screen serialized length", #screen_jpg_serialized, torch.type(screen_jpg_serialized))
        --print("screen length serialized:", #screen_jpg_serialized)
        --print("screen length serialized escaped:", #con:escape(screen_jpg_serialized))
        local isLevelEnding_int = 0
        if state.isLevelEnding then isLevelEnding_int = 1 end
        local dummy_int = 0
        if state.isDummy then dummy_int = 1 end

        --screenjpg = con:escape(screen_jpg_serialized)
        --screenjpg = "0",)
        --stmt:bind_blob(0, screen_jpg_serialized)
        stmt:bind_names{
            screenjpg = screen_jpg_serialized,
            id = state.id,
            score = state.score,
            countlifes = state.countLifes,
            levelbeatenstatus = state.levelBeatenStatus,
            mariogamestatus = state.marioGameStatus,
            playerx = state.playerX,
            marioimage = state.marioImage,
            islevelending = isLevelEnding_int,
            actionarrow = state.action.arrow,
            actionbutton = state.action.button,
            rewardscorediff = state.reward.scoreDiffReward,
            rewardxdiff = state.reward.xDiffReward,
            rewardlevelbeaten = state.reward.levelBeatenReward,
            rewardexpectedgamma = state.reward.expectedGammaReward,
            rewardexpectedgammaraw = state.reward.expectedGammaRewardRaw,
            rewardobservedgamma = state.reward.observedGammaReward,
            isdummy = dummy_int,
            isvalidation = val_int
        }
        --print("xDiffReward", state.reward.xDiffReward)

        stmt:step()
        --stmt:finalize()
        stmt:reset()
    end
    stmt:finalize()
end

function memory.insertStatesBatched(states, validation, updateIfExists)
    local batchSize = 32
    memory.resetCache()

    local ifExistsCommand = "IGNORE"
    if updateIfExists == true then ifExistsCommand = "UPDATE" end
    local val_int = 0
    if validation then val_int = 1 end

    for i=1,#states,batchSize do
        local sql = string.format([[
            INSERT OR %s INTO states (
                screen_jpg,
                id,
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
        ]], ifExistsCommand)

        local data = {}
        for j=i,math.min(i+batchSize, #states) do
            local state = states[j]

            assert(state.action ~= nil)
            assert(state.reward ~= nil)
            --print("X")

            local screen_jpg_serialized = torch.serialize(state.screen, "ascii")
            --print("screen serialized length", #screen_jpg_serialized, torch.type(screen_jpg_serialized))
            --print("screen length serialized:", #screen_jpg_serialized)
            --print("screen length serialized escaped:", #con:escape(screen_jpg_serialized))
            local isLevelEnding_int = 0
            if state.isLevelEnding then isLevelEnding_int = 1 end
            local dummy_int = 0
            if state.isDummy then dummy_int = 1 end

            if j > i then
                sql = sql .. ", "
            end

            local values = [[(
                :screenjpg#,
                :id#,
                :score#,
                :countlifes#,
                :levelbeatenstatus#,
                :mariogamestatus#,
                :playerx#,
                :marioimage#,
                :islevelending#,
                :actionarrow#,
                :actionbutton#,
                :rewardscorediff#,
                :rewardxdiff#,
                :rewardlevelbeaten#,
                :rewardexpectedgamma#,
                :rewardexpectedgammaraw#,
                :rewardobservedgamma#,
                :isdummy#,
                :isvalidation#
            )]]
            values = values:gsub("#", j)
            sql = sql .. values
            data["screenjpg" .. j] = screen_jpg_serialized
            data["id" .. j] = state.id
            data["score" .. j] = state.score
            data["countlifes" .. j] = state.countLifes
            data["levelbeatenstatus" .. j] = state.levelBeatenStatus
            data["mariogamestatus" .. j] = state.marioGameStatus
            data["playerx" .. j] = state.playerX
            data["marioimage" .. j] = state.marioImage
            data["islevelending" .. j] = isLevelEnding_int
            data["actionarrow" .. j] = state.action.arrow
            data["actionbutton" .. j] = state.action.button
            data["rewardscorediff" .. j] = state.reward.scoreDiffReward
            data["rewardxdiff" .. j] = state.reward.xDiffReward
            data["rewardlevelbeaten" .. j] = state.reward.levelBeatenReward
            data["rewardexpectedgamma" .. j] = state.reward.expectedGammaReward
            data["rewardexpectedgammaraw" .. j] = state.reward.expectedGammaRewardRaw
            data["rewardobservedgamma" .. j] = state.reward.observedGammaReward
            data["isdummy" .. j] = dummy_int
            data["isvalidation" .. j] = val_int
        end

        --print(sql)
        local stmt = db:prepare(sql)
        stmt:bind_names(data)
        stmt:step()
        stmt:reset()
    end
end

function memory.getCountHoles()
    for row in db:nrows(string.format("SELECT COUNT(s1.id) AS c FROM states as s1 LEFT JOIN states as s2 ON s2.id=s1.id-1 WHERE s2.id IS NULL ORDER BY s1.id ASC;")) do
        return row.c
    end
end

function memory.reduceToMaxSizes()
    memory.resetCache()

    print(string.format("Count Holes (before reduce): %d", memory.getCountHoles()))

    local function deleteSubf(count, maxCount, validation)
        local val_int = 0
        if validation then val_int = 1 end
        local toDelete = count - maxCount
        --print(string.format("count=%d, max count=%d, toDelete=%d, val_int=%d", count, maxCount, toDelete, val_int))
        if toDelete > 0 then
            --[[
            local indices = {}
            for row in db:nrows(string.format("SELECT id FROM states WHERE is_validation = %d ORDER BY id ASC LIMIT %d", val_int, toDelete)) do
                table.insert(indices, row.id)
            end

            for i=1,#indices do
                db:exec(string.format("DELETE FROM states WHERE id = %d", indices[i]))
            end
            --]]
            db:exec(string.format("DELETE FROM states WHERE id IN (SELECT id FROM states WHERE is_validation=%d ORDER BY id ASC LIMIT %d)", val_int, toDelete))
        end
    end

    deleteSubf(memory.getCountEntries(false), memory.MEMORY_MAX_SIZE_TRAINING, false)
    deleteSubf(memory.getCountEntries(true), memory.MEMORY_MAX_SIZE_VALIDATION, true)
    print(string.format("Count Holes (after reduce): %d", memory.getCountHoles()))
end

function memory.removeRandomEntry(validation)
end

function memory.removeRandomEntries(nRemove, validation, skew)
end

function memory.getRandomWeightedIndex(skew, validation, skewStrength)

end

function memory.prepareStateIdsCache()
    local idsTrain = {}
    local idsVal = {}
    local val_int = 0
    local sql = string.format("SELECT id, is_validation FROM states WHERE is_dummy=0 ORDER BY id ASC")
    for row in db:nrows(sql) do
        if row.is_validation == 1 then
            table.insert(idsVal, row.id)
        else
            table.insert(idsTrain, row.id)
        end
    end
    memory.cache.state_ids_val = idsVal
    memory.cache.state_ids_train = idsTrain
end

function memory.getStateIdsCache(validation)
    assert(validation == true or validation == false)
    if validation then
        if memory.cache.state_ids_val == nil then
            memory.prepareStateIdsCache()
        end
        return memory.cache.state_ids_val
    else
        if memory.cache.state_ids_train == nil then
            memory.prepareStateIdsCache()
        end
        return memory.cache.state_ids_train
    end
end

function memory.getRandomStateChains(n, length, validation)
    assert(validation == true or validation == false)
    --assert(memory.getCountEntries(validation) >= length)
    local val_int = 0
    if validation then val_int = 1 end

    --[[
    local timeStart = os.clock()
    local lastStatesIndices = {}
    local sql = string.format("SELECT id FROM states WHERE is_validation = %d and is_dummy=0 and id > (select MIN(id) from states)+%d ORDER BY random() LIMIT %d", val_int, length, n)
    for row in db:nrows(sql) do
        table.insert(lastStatesIndices, row.id)
    end
    assert(#lastStatesIndices == n)
    print(string.format("getRandomStateChains load ids: %.8f", os.clock()-timeStart))
    --]]

    --local timeStart = os.clock()
    local ids = memory.getStateIdsCache(validation)
    local lastStatesIndices = {}
    local nbIds = #ids
    for i=1,n do
        --print(string.format("i=%d val=%d length=%d nbIds=%d", i, val_int, length, nbIds))
        table.insert(lastStatesIndices, ids[math.random(length, nbIds)])
    end
    --print(string.format("getRandomStateChains load ids: %.8f", os.clock()-timeStart))

    local stateChains = {}
    for s=1,#lastStatesIndices do
        local id = lastStatesIndices[s]
        local indices = {}
        for i=1,length do table.insert(indices, id - length + i) end
        local stateChain = memory.getStatesByIndices(indices)
        if #stateChain ~= length then
            -- load another chain if a chain couldn't be loaded completely
            --print(string.format("[WARNING] Missing states in state chain %d, found %d expected %d; loading recursively", s, #stateChain, length))
            stateChain = memory.getRandomStateChains(1, length, validation)
            stateChain = stateChain[1]
        end
        table.insert(stateChains, stateChain)
    end
    return stateChains
end

function memory.getStatesByIndices(indices)
    --print("start getStatesByIndices")
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

    for row in db:nrows(string.format("SELECT * FROM states WHERE id IN (%s)", indicesStr)) do
        local state = memory.rowToState(row)
        assert(indexToState[state.id] ~= nil)
        indexToState[state.id] = state
    end
    --print("Found ", found, "of ", #indices)

    local notFound = 0
    for i=1,#indices do
        if indexToState[indices[i]] == false then
            --print(string.format("[WARNING] Could not find state with index %d in getStatesByIndices() (all searched indices: %s)", indices[i], indicesStr))
            notFound = notFound + 1
        end
    end

    local states = {}
    --[[
    for k,v in ipairs(indexToState) do
        print(k, v)
        table.insert(states, v)
    end
    --]]
    for k,v in pairs(indexToState) do
        if v ~= false then
            table.insert(states, v)
        end
    end
    table.sort(states, function(a, b) return a.id < b.id end)
    --print("#states", #states)
    --print(states[1].id, states[2].id)
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

    local length = STATES_PER_EXAMPLE
    local loadLength = length
    if reevaluate then loadLength = loadLength + 1 end

    --local timeStart = os.clock()
    local stateChains = memory.getRandomStateChains(batchSize, loadLength, validation)
    --print(string.format("getRandomStateChains: %.8f", os.clock()-timeStart))

    --for i=1,batchSize do
        --local idx = memory.getRandomWeightedIndex("top", validation)
    --    local stateChain = memory.getRandomStateChain(loadLength, validation)
    --    table.insert(stateChains, stateChain)
    --end

    if not reevaluate then
        local batchInput, batchTarget = network.stateChainsToBatch(stateChains)
        return batchInput, batchTarget, stateChains
    else
        local stateChainsCurrent = {}
        local stateChainsNext = {}
        for i=1,#stateChains do
            local stateChain = stateChains[i]
            local stateChainCurrent = {}
            local stateChainNext = {}
            for j=1,length do
                table.insert(stateChainCurrent, stateChain[j])
                table.insert(stateChainNext, stateChain[j+1])
            end
            table.insert(stateChainsCurrent, stateChainCurrent)
            table.insert(stateChainsNext, stateChainNext)
        end

        --local timeStart = os.clock()
        local bestActions = network.approximateBestActionsBatch(stateChainsNext)
        --print(string.format("approximateBestActionsBatch: %.8f", os.clock()-timeStart))

        --local timeStart = os.clock()
        for i=1,#stateChains do
            local oldReward = stateChainsCurrent[i][length].reward
            local newReward = rewards.statesToReward(stateChainsNext[i], bestActions[i].action, bestActions[i].value)
            newReward.observedGammaReward = oldReward.observedGammaReward
            stateChainsCurrent[i][length].reward = newReward
        end
        --print(string.format("statesToReward: %.8f", os.clock()-timeStart))

        local batchInput, batchTarget = network.stateChainsToBatch(stateChainsCurrent)
        return batchInput, batchTarget, stateChainsCurrent
    end
end

function memory.rowToState(row)
    local screen_jpg
    if row.screen_jpg == nil then
        print("[WARNING] screen was nil on state", row.id)
        screen_jpg = torch.zeros(IMG_DIMENSIONS[1], IMG_DIMENSIONS[2], IMG_DIMENSIONS[3])
        screen_jpg = util.compressJPG(screen_jpg)
    else
        --print("Deserializing...")
        --print(row.screen_jpg)
        screen_jpg = torch.deserialize(row.screen_jpg, "ascii")
        --print("Deserialized.")
    end

    --local screen_jpg = nil
    --print(screen_jpg:size(1), screen_jpg:size(2), screen_jpg:size(3))
    --print("row", row)

    --[[
    print("---------------------")
    print("row.action_arrow", row.action_arrow)
    print("row.action_button", row.action_button)
    print("row.reward_score_diff", row.reward_score_diff)
    print("row.reward_x_diff", row.reward_x_diff)
    print("row.reward_level_beaten", row.reward_level_beaten)
    print("row.reward_expected_gamma", row.reward_expected_gamma)
    print("row.reward_expected_gamma_raw", row.reward_expected_gamma_raw)
    print("row.reward_observed_gamma", row.reward_observed_gamma)
    print("row.id", row.id)
    print("row.score", row.score)
    print("row.count_lifes", row.count_lifes)
    print("row.level_beaten_status", row.level_beaten_status)
    print("row.mario_game_status", row.mario_game_status)
    print("row.player_x", row.player_x)
    print("row.mario_image", row.mario_image)
    print("row.is_level_ending", row.is_level_ending)
    print("row.is_dummy", row.is_dummy)
    --]]

    local isDummy = false
    if row.is_dummy == 1 then isDummy = true end
    local action = Action.new(row.action_arrow, row.action_button)
    local reward = Reward.new(row.reward_score_diff, row.reward_x_diff, row.reward_level_beaten, row.reward_expected_gamma, row.reward_expected_gamma_raw, row.reward_observed_gamma)
    local state = State.new(row.id, screen_jpg, row.score, row.count_lifes, row.level_beaten_status, row.mario_game_status, row.player_x, row.mario_image, row.is_level_ending == 1, action, reward)
    state.isDummy = isDummy
    return state
end

function memory.plot(subsampling)

end

return memory
