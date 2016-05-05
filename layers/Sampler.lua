-- Based on JoinTable module

require 'nn'

local Sampler, parent = torch.class('nn.Sampler', 'nn.Module')

function Sampler:__init()
    parent.__init(self)
    self.gradInput = {}
end

function Sampler:updateOutput(input)
    self.eps = self.eps or input[1].new()
    self.eps:resizeAs(input[1]):copy(torch.randn(input[1]:size()))

    self.ouput = self.output or self.output.new()
    self.output:resizeAs(input[2]):copy(input[2])
    self.output:mul(0.5):exp():cmul(self.eps)

    self.output:add(input[1])

    return self.output
end

function Sampler:updateGradInput(input, gradOutput)
    self.gradInput[1] = self.gradInput[1] or input[1].new()
    self.gradInput[1]:resizeAs(gradOutput):copy(gradOutput)

    self.gradInput[2] = self.gradInput[2] or input[2].new()
    self.gradInput[2]:resizeAs(gradOutput):copy(input[2])

    self.gradInput[2]:mul(0.5):exp():mul(0.5):cmul(self.eps)
    self.gradInput[2]:cmul(gradOutput)

    return self.gradInput
end
