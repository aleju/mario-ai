-- from dpnn, adapted to also work in lsnes
local PrintSize, parent = torch.class('nn.PrintSize', 'nn.Module')

function PrintSize:__init(prefix)
   parent.__init(self)
   self.prefix = prefix or "PrintSize"
end

function PrintSize:updateOutput(input)
   self.output = input
   local size = PrintSize.sizeToString(input)
   print(self.prefix..":input\n", size)
   return self.output
end


function PrintSize:updateGradInput(input, gradOutput)
   local size = PrintSize.sizeToString(gradOutput)
   print(self.prefix..":gradOutput\n", size)
   self.gradInput = gradOutput
   return self.gradInput
end

function PrintSize.sizeToString(input)
    local size
    if torch.type(input) == 'table' then
        local s = ""
        for i=1,#input do
            s = s .. "[" .. i .. "] => ("
            s = s .. PrintSize.sizeToString(input[i])
            s = s .. ") "
        end
        size = s
    elseif torch.type(input) == 'nil' then
       size = 'missing size'
    else
       local s = ""
       for i=1,#input:size() do
           if i > 1 then
               --s = s .. " [" .. i .. " of " .. input:nDimension() .. "] " .. input:size(i)
               s = s .. ", " .. input:size(i)
           else
               --s = s .. " [" .. i .. " of " .. input:nDimension() .. "] " .. input:size(i)
               s = s .. "" .. input:size(i)
           end
       end
       size = s
    end
    return size
end
