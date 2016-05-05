local State = {}
State.__index = State

function State.new(screen, score, countLifes, levelBeatenStatus, marioGameStatus, playerX, marioImage, isLevelEnding, action, reward)
    local self = setmetatable({}, State)
    self.screen = screen
    self.score = score
    self.countLifes = countLifes
    self.levelBeatenStatus = levelBeatenStatus
    self.marioGameStatus = marioGameStatus
    self.playerX = playerX
    self.marioImage = marioImage
    self.isLevelEnding = isLevelEnding
    self.action = action
    self.reward = reward
    self.isDummy = false
    return self
end

return State
