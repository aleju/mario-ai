-- This object represents a state.
local State = {}
State.__index = State

-- @param id The id number of the state.
-- @param screen The screenshot of the game, expected to be jpg-compressed.
-- @param countLifes The number of lifes of Mario.
-- @param levelBeatenStatus Level-beaten-status, read from the game memory. Should be 128 if level ends / before switching to overworld.
-- @param marioGameStatus TODO Is this still used for anything?
-- @param playerX X-coordinate of Mario.
-- @param marioImage Current sprite of Mario, should be 62 if death animation is playing.
-- @param isLevelEnding True if the ending animation plays, after walking through the flag pole.
-- @param action The action chosen at this state.
-- @param reward The reward received at this state.
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

-- Objects here have no members functions, because those seemed to be gone
-- after torch.save() and then torch.load().

return State
