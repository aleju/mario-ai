-- Object representing a received reward.
local Reward = {}
Reward.__index = Reward

-- scoreDiffReward = reward from score changes between states
-- xDiffReward = reward from Mario moving to the right
-- levelBeatenReward = reward from finishing the level or Mario dying
-- expectedGammaReward = discounted Q-value reward
-- expectedGammaRewardRaw = undiscounted Q-value reward
-- observedGammaReward = received direct rewards from future states, cascaded backwards (and discounted)
function Reward.new(scoreDiffReward, xDiffReward, levelBeatenReward, expectedGammaReward, expectedGammaRewardRaw, observedGammaReward)
    local self = setmetatable({}, Reward)
    self.scoreDiffReward = scoreDiffReward
    self.xDiffReward = xDiffReward
    self.levelBeatenReward = levelBeatenReward
    self.expectedGammaReward = expectedGammaReward
    self.expectedGammaRewardRaw = expectedGammaRewardRaw
    self.observedGammaReward = observedGammaReward or 0
    return self
end

-- Objects here have no members functions, because those seemed to be gone
-- after torch.save() and then torch.load().
-- See rewards.lua for some functions.

return Reward
