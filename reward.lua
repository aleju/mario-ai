local Reward = {}
Reward.__index = Reward

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

--[[
function Reward.getSum(self)
    return self:getDirectReward() + self.gammaReward
end

function Reward.getDirectReward(self)
    return self.scoreDiffReward + self.xDiffReward + self.levelBeatenReward
end
--]]

return Reward
