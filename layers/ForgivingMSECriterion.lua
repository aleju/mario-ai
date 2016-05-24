local ForgivingMSECriterion, parent = torch.class('nn.ForgivingMSECriterion', 'nn.Criterion')

function ForgivingMSECriterion:__init(forgiveness, sizeAverage)
    parent.__init(self)
    if sizeAverage ~= nil then
        self.sizeAverage = sizeAverage
    else
        self.sizeAverage = true
    end
    self.forgiveness = forgiveness or 1.0
end

--[[
function ForgivingAbsCriterion:updateOutput(input, target)
  local delta = target - input
  local loss = delta:abs():apply(
    function(x)
      if x < 1 then
        return 0.5 * x * x
      else
        return x - 0.5
      end
    end
  ):sum()
  if self.sizeAverage then
    loss = loss / input:nElement()
  end
  return loss
end
--]]

function ForgivingMSECriterion:updateOutput(input, target)
    assert(#input:size() == 2 and #target:size() == 2)
    local epsilon = 1e-6
    local delta = target - input
    for exampleIdx=1,target:size(1) do
        for actionIdx=1,target:size(2) do
            if target[exampleIdx][actionIdx] > 0 - epsilon and target[exampleIdx][actionIdx] < 0 + epsilon then
                delta[exampleIdx][actionIdx] = (1-self.forgiveness) * delta[exampleIdx][actionIdx]
            end
        end
    end
    local loss = delta:pow(2):sum()
    if self.sizeAverage then
        loss = loss / input:nElement()
    end
    return loss
end

function ForgivingMSECriterion:updateGradInput(input, target)
    local norm = self.sizeAverage and 2/input:nElement() or 2

    self.gradInput:resizeAs(input):copy(input):add(-1, target):mul(norm)

    local epsilon = 1e-6
    --[[
    print("Grads BEFORE:")
    for exampleIdx=1,2 do
        print("e",exampleIdx)
        for actionIdx=1,target:size(2) do
            print(string.format("%.2f", self.gradInput[exampleIdx][actionIdx]))
        end
    end
    --]]
    for exampleIdx=1,target:size(1) do
        for actionIdx=1,target:size(2) do
            if target[exampleIdx][actionIdx] > 0 - epsilon and target[exampleIdx][actionIdx] < 0 + epsilon then
                self.gradInput[exampleIdx][actionIdx] = (1-self.forgiveness) * self.gradInput[exampleIdx][actionIdx]
            end
        end
    end
    --[[
    print("Grads AFTER:")
    for exampleIdx=1,2 do
        print("e",exampleIdx)
        for actionIdx=1,target:size(2) do
            print(string.format("%.2f", self.gradInput[exampleIdx][actionIdx]))
        end
    end
    --]]
    return self.gradInput
end

--[[
function SmoothL1Criterion:updateGradInput(input, target)
  local norm = self.sizeAverage and 1.0 / input:nElement() or 1.0
  self.gradInput:resizeAs(input):copy(input):add(-1, target)
  self.gradInput:apply(
    function(x)
      if math.abs(x) < 1 then
        return norm * x
      elseif x > 0 then
        return norm
      else
        return -norm
      end
    end
  )
  return self.gradInput
end
--]]
