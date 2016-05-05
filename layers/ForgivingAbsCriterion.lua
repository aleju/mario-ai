local ForgivingAbsCriterion, parent = torch.class('nn.ForgivingAbsCriterion', 'nn.Criterion')

function ForgivingAbsCriterion:__init(sizeAverage)
    parent.__init(self)
    if sizeAverage ~= nil then
        self.sizeAverage = sizeAverage
    else
        self.sizeAverage = false
    end
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

function ForgivingAbsCriterion:updateOutput(input, target)
    assert(#input:size() == 2 and #target:size() == 2)
    local epsilon = 1e-6
    local delta = target - input
    for exampleIdx=1,target:size(1) do
        for actionIdx=1,target:size(2) do
            if target[exampleIdx][actionIdx] > 0 - epsilon and target[exampleIdx][actionIdx] < 0 + epsilon then
                delta[exampleIdx][actionIdx] = 0
            end
        end
    end
    local loss = delta:abs():sum()
    if self.sizeAverage then
        loss = loss / input:nElement()
    end
    return loss
end

function ForgivingAbsCriterion:updateGradInput(input, target)
    local norm = self.sizeAverage and 1/input:nElement() or 1

    self.gradInput:resizeAs(input) --:copy(input):add(-1, target):mul(norm)

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
            local target_i = target[exampleIdx][actionIdx]
            if target_i > 0 - epsilon and target_i < 0 + epsilon then
                self.gradInput[exampleIdx][actionIdx] = 0
            else
                local input_i = input[exampleIdx][actionIdx]
                local delta_i = input_i - target_i

                if delta_i > 0 then
                    self.gradInput[exampleIdx][actionIdx] = norm
                else
                    self.gradInput[exampleIdx][actionIdx] = (-1) * norm
                end
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
