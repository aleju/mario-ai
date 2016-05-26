-- Used in the Spatial Transformer. Adds a constant tensor to another tensor.
-- Is adapted to the specific situation and might not work in other cases, i.e.
-- with other tensor sizes.
local AddConstantTensor, parent = torch.class('nn.AddConstantTensor', 'nn.Module')

function AddConstantTensor:__init(tnsr, ip)
  parent.__init(self)
  assert(tnsr:nDimension() == 1)
  self.constant_tensor = tnsr
end

function AddConstantTensor:updateOutput(input)
  assert(input:nDimension() == self.constant_tensor:nDimension() or input:nDimension() == self.constant_tensor:nDimension()+1)
  if input:nDimension() == self.constant_tensor:nDimension() then
      self.output:resizeAs(input)
      self.output:copy(input)
      self.output:add(self.constant_tensor)
  else
      local constant_tensor_batch = torch.repeatTensor(self.constant_tensor, input:size(1), 1)
      self.output:resizeAs(input)
      self.output:copy(input)
      self.output:add(constant_tensor_batch)
  end

  --[[
  if self.inplace then
    input:add(self.constant_tensor)
    self.output:set(input)
  else
    self.output:resizeAs(input)
    self.output:copy(input)
    self.output:add(self.constant_tensor)
  end
  --]]
  return self.output
end

function AddConstantTensor:updateGradInput(input, gradOutput)
  self.gradInput:resizeAs(gradOutput)
  self.gradInput:copy(gradOutput)
  return self.gradInput
end
