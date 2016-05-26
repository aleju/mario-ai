-- Functions and constants dealing with the choice and application of actions,
-- i.e. pressing buttons on the controller.
-- Functions to find optimal actions are in network.lua .

local actions = {}

-- Action ids used by the emulator (?)
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

-- List of all action ids.
actions.ACTIONS_ALL = {
    actions.ACTION_BUTTON_B, actions.ACTION_BUTTON_Y,
    actions.ACTION_BUTTON_SELECT, actions.ACTION_BUTTON_START,
    actions.ACTION_BUTTON_UP, actions.ACTION_BUTTON_DOWN,
    actions.ACTION_BUTTON_LEFT, actions.ACTION_BUTTON_RIGHT,
    actions.ACTION_BUTTON_A, actions.ACTION_BUTTON_X,
    actions.ACTION_BUTTON_L, actions.ACTION_BUTTON_R
}

-- List of action ids that the network can use (i.e. for which it predicts rewards).
-- Note that the order is important, the first action id is the action that is
-- represented by the first output neuron of the network.
actions.ACTIONS_NETWORK = {
    actions.ACTION_BUTTON_B, actions.ACTION_BUTTON_Y,
    actions.ACTION_BUTTON_UP, actions.ACTION_BUTTON_DOWN,
    actions.ACTION_BUTTON_LEFT, actions.ACTION_BUTTON_RIGHT,
    actions.ACTION_BUTTON_A, actions.ACTION_BUTTON_X
}

-- List of arrow actions (up, down, left, right).
actions.ACTIONS_ARROWS = {
    actions.ACTION_BUTTON_UP,actions.ACTION_BUTTON_DOWN,
    actions.ACTION_BUTTON_LEFT, actions.ACTION_BUTTON_RIGHT
}

-- List of "other" button actions (A, B, X, Y).
actions.ACTIONS_BUTTONS = {
    actions.ACTION_BUTTON_B, actions.ACTION_BUTTON_Y,
    --actions.ACTION_BUTTON_SELECT, actions.ACTION_BUTTON_START,
    actions.ACTION_BUTTON_A, actions.ACTION_BUTTON_X,
    --actions.ACTION_BUTTON_L, actions.ACTION_BUTTON_R
}

-- Action names used by the emulator.
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

-- Short string names for each action, used for string conversions.
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

-- Returns whether a certain action index represents an arrow action (up, down, left, right).
function actions.isArrowsActionIdx(actionIdx)
    for i=1,#actions.ACTIONS_ARROWS do
        if actionIdx == actions.ACTIONS_ARROWS[i] then
            return true
        end
    end
    return false
end

-- Returns whether a certain action index represents a button action (A, B, X, Y).
function actions.isButtonsActionIdx(actionIdx)
    for i=1,#actions.ACTIONS_BUTTONS do
        if actionIdx == actions.ACTIONS_BUTTONS[i] then
            return true
        end
    end
    return false
end

-- Transforms an action (arrow action index + button action index) to a short, readable string.
function actions.actionToString(action)
    if action == nil then
        return "nil"
    else
        return actions.ACTION_TO_SHORT_NAME[action.arrow] .. "+" .. actions.ACTION_TO_SHORT_NAME[action.button]
    end
end

-- Returns a new, random Action object.
function actions.createRandomAction()
    local arrow = actions.ACTIONS_ARROWS[math.random(#actions.ACTIONS_ARROWS)]
    local button = actions.ACTIONS_BUTTONS[math.random(#actions.ACTIONS_BUTTONS)]
    return Action.new(arrow, button)
end

-- Resets all buttons (to "not pressed").
function actions.endAllActions()
    for i=1,#actions.ACTIONS_ALL do
        local newstate = 0 -- 1 = pressed, 0 = released
        local mode = 3 -- 1 = autohold, 2 = framehold, others = press/release
        input.do_button_action(actions.ACTION_TO_BUTTON_NAME[actions.ACTIONS_ALL[i]], newstate, mode)
    end
end

-- Starts an action.
-- @param action An Action object.
function actions.startAction(action)
    assert(action ~= nil)
    local newstate = 1 -- 1 = pressed, 0 = released
    local mode = 3 -- 1 = autohold, 2 = framehold, others = press/release
    local arrowAction = actions.ACTION_TO_BUTTON_NAME[action.arrow]
    local buttonAction = actions.ACTION_TO_BUTTON_NAME[action.button]
    assert(arrowAction ~= nil)
    assert(buttonAction ~= nil)
    input.do_button_action(arrowAction, newstate, mode)
    input.do_button_action(buttonAction, newstate, mode)
end

-- Chooses an action based on a chain of states.
-- @param lastStates List of State objects.
-- @param perfect Boolean, sets exploration prob. to 0.0 (not really necessary anymore with pExplore).
-- @param bestAction Optionally an Action object for epsilon-greedy policy, otherwise the best action will be approximated.
-- @param pExplore Exploration probability for epsilon-greedy policy.
function actions.chooseAction(lastStates, perfect, bestAction, pExplore)
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
