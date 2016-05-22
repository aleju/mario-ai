local State = {}
State.__index = State

function State.new(id, screen, score, countLifes, levelBeatenStatus, marioGameStatus, playerX, marioImage, isLevelEnding, action, reward)
    local self = setmetatable({}, State)
    if id == nil then
        id = STATS.STATE_ID
        STATS.STATE_ID = STATS.STATE_ID + 1
    end
    self.id = id
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
