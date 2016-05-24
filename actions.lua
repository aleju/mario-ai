local actions = {}

actions.ACTION_BUTTON_B = 0
actions.ACTION_BUTTON_Y = 1
actions.ACTION_BUTTON_SELECT = 2
actions.ACTION_BUTTON_START = 3
actions.ACTION_BUTTON_UP = 4
actions.ACTION_BUTTON_DOWN = 5
actions.ACTION_BUTTON_LEFT = 6
actions.ACTION_BUTTON_RIGHT = 7
actions.ACTION_BUTTON_A = 8
actions.ACTION_BUTTON_X = 9
actions.ACTION_BUTTON_L = 10
actions.ACTION_BUTTON_R = 11
actions.ACTIONS_ALL = {
    actions.ACTION_BUTTON_B, actions.ACTION_BUTTON_Y,
    actions.ACTION_BUTTON_SELECT, actions.ACTION_BUTTON_START,
    actions.ACTION_BUTTON_UP, actions.ACTION_BUTTON_DOWN,
    actions.ACTION_BUTTON_LEFT, actions.ACTION_BUTTON_RIGHT,
    actions.ACTION_BUTTON_A, actions.ACTION_BUTTON_X,
    actions.ACTION_BUTTON_L, actions.ACTION_BUTTON_R
}
--[[actions.ACTIONS_NETWORK = {
    actions.ACTION_BUTTON_B, --actions.ACTION_BUTTON_Y,
    actions.ACTION_BUTTON_LEFT, actions.ACTION_BUTTON_RIGHT,
    actions.ACTION_BUTTON_A, --actions.ACTION_BUTTON_X
}--]]
actions.ACTIONS_NETWORK = {
    actions.ACTION_BUTTON_B, actions.ACTION_BUTTON_Y,
    actions.ACTION_BUTTON_UP, actions.ACTION_BUTTON_DOWN,
    actions.ACTION_BUTTON_LEFT, actions.ACTION_BUTTON_RIGHT,
    actions.ACTION_BUTTON_A, actions.ACTION_BUTTON_X
}
actions.ACTIONS_ARROWS = {
    actions.ACTION_BUTTON_UP,actions.ACTION_BUTTON_DOWN,
    actions.ACTION_BUTTON_LEFT, actions.ACTION_BUTTON_RIGHT
}
actions.ACTIONS_BUTTONS = {
    actions.ACTION_BUTTON_B, actions.ACTION_BUTTON_Y,
    --actions.ACTION_BUTTON_SELECT, actions.ACTION_BUTTON_START,
    actions.ACTION_BUTTON_A, actions.ACTION_BUTTON_X,
    --actions.ACTION_BUTTON_L, actions.ACTION_BUTTON_R
}
actions.ACTION_TO_BUTTON_NAME = {}
actions.ACTION_TO_BUTTON_NAME[0] = "gamepad-1-B"
actions.ACTION_TO_BUTTON_NAME[1] = "gamepad-1-Y"
actions.ACTION_TO_BUTTON_NAME[2] = "gamepad-1-select"
actions.ACTION_TO_BUTTON_NAME[3] = "gamepad-1-start"
actions.ACTION_TO_BUTTON_NAME[4] = "gamepad-1-up"
actions.ACTION_TO_BUTTON_NAME[5] = "gamepad-1-down"
actions.ACTION_TO_BUTTON_NAME[6] = "gamepad-1-left"
actions.ACTION_TO_BUTTON_NAME[7] = "gamepad-1-right"
actions.ACTION_TO_BUTTON_NAME[8] = "gamepad-1-A"
actions.ACTION_TO_BUTTON_NAME[9] = "gamepad-1-X"
actions.ACTION_TO_BUTTON_NAME[10] = "gamepad-1-L"
actions.ACTION_TO_BUTTON_NAME[11] = "gamepad-1-R"

actions.ACTION_TO_SHORT_NAME = {}
actions.ACTION_TO_SHORT_NAME[0] = "B"
actions.ACTION_TO_SHORT_NAME[1] = "Y"
actions.ACTION_TO_SHORT_NAME[2] = "s"
actions.ACTION_TO_SHORT_NAME[3] = "S"
actions.ACTION_TO_SHORT_NAME[4] = "AU"
actions.ACTION_TO_SHORT_NAME[5] = "AD"
actions.ACTION_TO_SHORT_NAME[6] = "AL"
actions.ACTION_TO_SHORT_NAME[7] = "AR"
actions.ACTION_TO_SHORT_NAME[8] = "A"
actions.ACTION_TO_SHORT_NAME[9] = "X"
actions.ACTION_TO_SHORT_NAME[10] = "L"
actions.ACTION_TO_SHORT_NAME[11] = "R"


function actions.isArrowsActionIdx(actionIdx)
    for i=1,#actions.ACTIONS_ARROWS do
        if actionIdx == actions.ACTIONS_ARROWS[i] then
            return true
        end
    end
    return false
end

function actions.isButtonsActionIdx(actionIdx)
    for i=1,#actions.ACTIONS_BUTTONS do
        if actionIdx == actions.ACTIONS_BUTTONS[i] then
            return true
        end
    end
    return false
end

function actions.actionToString(action)
    if action == nil then
        return "nil"
    else
        return actions.ACTION_TO_SHORT_NAME[action.arrow] .. "+" .. actions.ACTION_TO_SHORT_NAME[action.button]
    end
end

function actions.createRandomAction()
    local arrow = actions.ACTIONS_ARROWS[math.random(#actions.ACTIONS_ARROWS)]
    local button = actions.ACTIONS_BUTTONS[math.random(#actions.ACTIONS_BUTTONS)]
    return Action.new(arrow, button)
end

function actions.endAllActions()
    --local lcid = 1
    --local port, controller = input.lcid_to_pcid2(lcid)
    --local controller = 0
    --input.set2(port, controller, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    for i=1,#actions.ACTIONS_ALL do
        local newstate = 0 -- 1 = pressed, 0 = released
        local mode = 3 -- 1 = autohold, 2 = framehold, others = press/release
        input.do_button_action(actions.ACTION_TO_BUTTON_NAME[actions.ACTIONS_ALL[i]], newstate, mode)
    end
end

--[[
function endAction(action)
    local lcid = 1
    local port, controller = input.lcid_to_pcid2(lcid)
    --local controller = 0
    local value = 0 -- 0 = release, 1 = press
    input.set2(port, controller, action, value)
end
--]]

function actions.startAction(action)
    assert(action ~= nil)
    --for lcid=1,8 do
    --print(port, controller)
    --local controller = 0
    --local value = 1 -- 0 = release, 1 = press
    --input.set2(port, controller, action, value)
    --end
    --setJoypad2({action})
    --print("Starting action!", action)
    local newstate = 1 -- 1 = pressed, 0 = released
    local mode = 3 -- 1 = autohold, 2 = framehold, others = press/release
    --if action == ACTION_BUTTON_B or action == ACTION_BUTTON_A then
    --    mode = 2
    --end
    local arrowAction = actions.ACTION_TO_BUTTON_NAME[action.arrow]
    local buttonAction = actions.ACTION_TO_BUTTON_NAME[action.button]
    assert(arrowAction ~= nil)
    assert(buttonAction ~= nil)
    input.do_button_action(arrowAction, newstate, mode)
    input.do_button_action(buttonAction, newstate, mode)
end

function actions.setJoypad(actions)
   print("set joypad")
   local lcid = 1
   local port, controller = input.lcid_to_pcid2(lcid)
   local value = 1 -- 0 = release, 1 = press
   --input.set2(port, controller, ACTION_BUTTON_A, value)
   input.set2(port, controller, 0, 0)
   --for i=0,32000 do
   --    input.set2(port, controller, i, 1)
   --end
end

function actions.setJoypad2(actions)
    local lcid = 1
    local port, controller = input.lcid_to_pcid2(lcid)
    --[[
    local table = {
        B = false, Y = false, select = false, start = false,
        up = false, down = false, left = false, right = false,
        A = false,  X = false,
        L = false, R = false
    }

    for i=1,#actions do
        local action = actions[i]
        if action == ACTION_BUTTON_B then table.B = true end
        if action == ACTION_BUTTON_Y then table.Y = true end
        if action == ACTION_BUTTON_SELECT then table.select = true end
        if action == ACTION_BUTTON_START then table.start = true end
        if action == ACTION_BUTTON_UP then table.up = true end
        if action == ACTION_BUTTON_DOWN then table.down = true end
        if action == ACTION_BUTTON_LEFT then table.left = true end
        if action == ACTION_BUTTON_RIGHT then table.right = true end
        if action == ACTION_BUTTON_A then table.A = true end
        if action == ACTION_BUTTON_X then table.X = true end
        if action == ACTION_BUTTON_L then table.L = true end
        if action == ACTION_BUTTON_R then table.R = true end
    end
    --]]
    local table = {}
    table["P1 B"] = true
    table["P1 Y"] = true
    table["P1 select"] = true
    table["P1 start"] = true
    table["P1 up"] = true
    table["P1 down"] = true
    table["P1 left"] = true
    table["P1 right"] = true
    table["P1 A"] = true
    table["P1 X"] = true
    table["P1 L"] = true
    table["P1 R"] = true
    local table2 = {}
    table2["B"] = true
    table2["Y"] = false
    table2["select"] = false
    table2["start"] = false
    table2["up"] = false
    table2["down"] = false
    table2["left"] = false
    table2["right"] = false
    table2["A"] = true
    table2["X"] = false
    table2["L"] = false
    table2["R"] = false
    local table3 = {}
    for i=1,12 do
        table3[i] = false
        if i==1 and math.random()<0.1 then table3[i] = true end
    end
    print("Sending to joyset...", table)
    --for i=1,1 do
        input.joyset(1, table3)
    --end
end

function chooseAction(lastStates, perfect, bestAction, pExplore)
    perfect = perfect or false
    pExplore = pExplore or STATS.P_EXPLORE_CURRENT
    local _action, _actionValue
    if not perfect and math.random() < pExplore then
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

return actions
