-- This script can be used to deactivate all buttons, if any are still pressed.
memory = require 'memory'
network = require 'network'
actions = require 'actions'
util = require 'util'

function on_paint()
    actions.endAllActions()
end

gui.repaint()
