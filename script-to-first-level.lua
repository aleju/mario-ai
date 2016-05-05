memory = require 'memory'
network = require 'network'
actions = require 'actions'
util = require 'util'

counter = 0
do_actions = {
    actions.ACTION_BUTTON_B,
    -1,
    actions.ACTION_BUTTON_B,
    -1,
    actions.ACTION_BUTTON_B,
    -1,
    -1,
    -1,
    -1,
    -1,
    -1,
    -1,
    -1,
    -1,
    -1,
    -1,
    -1,
    -1,
    -1,
    -1,
    -1,
    -1,
    actions.ACTION_BUTTON_B,
    -1,
    actions.ACTION_BUTTON_UP,
    -1,
    -1,
    actions.ACTION_BUTTON_LEFT,
    -1,
    -1,
    actions.ACTION_BUTTON_B,
    -1,
    -1,
    -1,
    -1,
    -1,
    -1,
    actions.ACTION_BUTTON_B,
    -1
}

function on_paint()
    if movie.currentframe() % 50 ~= 0 then
        return
    end

    counter = counter + 1
    if counter > #do_actions then
        return
    end

    if counter == 1 then
        util.setGameSpeedToVeryFast()
    end

    actions.endAllActions()
    if do_actions[counter] ~= -1 then
        actions.startAction(do_actions[counter])
    end

    if counter == #do_actions then
        util.setGameSpeedToNormal()
    end
end

actions.endAllActions()
gui.repaint()
