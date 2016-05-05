local Action = {}
Action.__index = Action

function Action.new(arrowAction, buttonAction)
    local self = setmetatable({}, Action)
    self.arrow = arrowAction
    self.button = buttonAction
    return self
end

return Action
