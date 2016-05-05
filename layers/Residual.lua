require 'nn'
require 'cudnn'
require 'cunn'

local Residual, parent = torch.class('nn.Residual', 'nn.Container')

function Residual:__init(modNet, skipNet)
    parent.__init(self)
    self.gradInput = torch.Tensor()
    self.train = true

    self.net = modNet

    if skipNet == nil then
        self.skip = nn.Sequential()
        self.skip:add(nn.Identity())
    else
        self.skip = skipNet
    end

    self.modules = {self.net, self.skip}
end

function Residual:updateOutput(input)
    local skip_forward = self.skip:forward(input)
    self.output:resizeAs(skip_forward):copy(skip_forward)
    self.output:add(self.net:forward(input))
    return self.output
end

function Residual:updateGradInput(input, gradOutput)
    self.gradInput = self.gradInput or input.new()
    self.gradInput:resizeAs(input):copy(self.skip:updateGradInput(input, gradOutput))
    self.gradInput:add(self.net:updateGradInput(input, gradOutput))
    return self.gradInput
end

function Residual:accGradParameters(input, gradOutput, scale)
    scale = scale or 1
    self.net:accGradParameters(input, gradOutput, scale)
end

function Residual:__tostring__()
    local tab = '  '
    local line = '\n'
    local next = ' -> '
    local str = 'nn.Residual'
    str = str .. ' {' .. line .. tab .. '[input'
    for i=1,#self.modules do
        str = str .. next .. '(' .. i .. ')'
    end
    str = str .. next .. 'output]'
    for i=1,#self.modules do
        str = str .. line .. tab .. '(' .. i .. '): ' .. tostring(self.modules[i]):gsub(line, line .. tab)
    end
    str = str .. line .. '}'
    return str
end
