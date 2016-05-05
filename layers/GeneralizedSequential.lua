-- from http://stackoverflow.com/questions/33648796/add-my-custom-loss-function-to-torch
local GeneralizedSequential, _ = torch.class('nn.GeneralizedSequential', 'nn.Sequential')

function GeneralizedSequential:forward(input, target)
    return self:updateOutput(input, target)
end

function GeneralizedSequential:updateOutput(input, target)
    local currentOutput = input
    for i=1,#self.modules do
        currentOutput = self.modules[i]:updateOutput(currentOutput, target)
    end
    self.output = currentOutput
    return currentOutput
end
