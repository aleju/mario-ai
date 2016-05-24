local L2Penalty, parent = torch.class('nn.L2Penalty','nn.Module')

--This module acts as an L2 latent state regularizer, adding the
--[gradOutput] to the gradient of the L2 loss. The [input] is copied to
--the [output].

function L2Penalty:__init(l2weight, sizeAverage, provideOutput)
    parent.__init(self)
    self.l2weight = l2weight
    self.sizeAverage = sizeAverage or false
    if provideOutput == nil then
       self.provideOutput = true
    else
       self.provideOutput = provideOutput
    end
end

function L2Penalty:updateOutput(input)
    local m = self.l2weight
    if self.sizeAverage == true then
      m = m/input:nElement()
    end
    local loss = m * input:norm(2) --pow(2):sum()
    self.loss = loss
    self.output = input
    return self.output
end

function L2Penalty:updateGradInput(input, gradOutput)
    local m = self.l2weight
    if self.sizeAverage == true then
      m = m/input:nElement()
    end

    self.gradInput:resizeAs(input):copy(input):mul(2):mul(m)

    if self.provideOutput == true then
        self.gradInput:add(gradOutput)
    end

    return self.gradInput
end
