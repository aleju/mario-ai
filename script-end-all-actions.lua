memory = require 'memory'
network = require 'network'
actions = require 'actions'
util = require 'util'

function on_paint()
    actions.endAllActions()
end

gui.repaint()
