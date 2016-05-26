-- Network functions.
-- E.g.:
--   - Creating the model
--   - Converting chains of states to training batches
--   - Forward/backward of batches
--   - Using the model to approximate good actions for a chain of states
require 'torch'
require 'paths'
require 'nn'
require 'layers.Residual'
require 'layers.PrintSize'
require 'layers.L2Penalty'
require 'layers.AddConstantTensor'
require 'stn'

local network = {}

-- Load a saved model or return nil.
function network.load(fp)
    fp = fp or "learned/network.th7"
    if paths.filep(fp) then
        local savedData = torch.load(fp)
        return savedData
    else
        print("[INFO] Could not load previously saved network, file does not exist.")
        return nil
    end
end

-- Save the model to the save file.
function network.save(fp)
    fp = fp or "learned/network.th7"
    network.prepareNetworkForSave(Q)
    torch.save(fp, Q)
end

-- Tries to load the network from the save file. If that fails, it creates a new network.
function network.createOrLoadQ()
    local loaded = network.load()
    if loaded == nil then
        return network.createQ10()
    else
        return loaded
    end
end

-- Create the model.
function network.createQ10()
    function conv(nbInputPlanes, nbOutputPlanes, ks, stride)
        return nn.Sequential()
                :add(cudnn.SpatialConvolution(nbInputPlanes, nbOutputPlanes, ks, ks, stride, stride, (ks-1)/2, (ks-1)/2))
                :add(nn.SpatialBatchNormalization(nbOutputPlanes))
                :add(nn.LeakyReLU(0.2, true))
    end

    local cH, hH, wH = unpack(IMG_DIMENSIONS_Q_HISTORY)
    local cL, hL, wL = unpack(IMG_DIMENSIONS_Q_LAST)
    local net = nn.Sequential()

    -- Action history branch. deals with previously chosen action(-ids).
    local actionHistory = nn.Sequential()
    if GPU then actionHistory:add(nn.Copy('torch.FloatTensor', 'torch.CudaTensor', true, true)) end
    actionHistory:add(nn.Reshape(STATES_PER_EXAMPLE * #actions.ACTIONS_NETWORK))
    actionHistory:add(nn.Linear(STATES_PER_EXAMPLE * #actions.ACTIONS_NETWORK, 32))
    actionHistory:add(nn.LeakyReLU(0.2, true))
    local actionHistorySize = 32

    -- State history branch. Deals with previously seen states (as small images).
    -- Note that this branch also retrieves the current state as a small image.
    local imageHistory = nn.Sequential()
    if GPU then imageHistory:add(nn.Copy('torch.FloatTensor', 'torch.CudaTensor', true, true)) end
    imageHistory:add(conv(STATES_PER_EXAMPLE, 64, 3, 1))
    imageHistory:add(conv(64, 64, 5, 2))
    imageHistory:add(conv(64, 64, 5, 4))
    local imageHistorySize = 64 * hH/2/4 * wH/2/4
    imageHistory:add(nn.Reshape(imageHistorySize))

    -- Last image branch. Sees only the current state as a larger image.
    local lastImage = nn.Sequential()
    if GPU then lastImage:add(nn.Copy('torch.FloatTensor', 'torch.CudaTensor', true, true)) end
    lastImage:add(conv(cL, 256, 5, 1))
    lastImage:add(nn.SpatialMaxPooling(2, 2))
    lastImage:add(conv(256, 64, 3, 1))

    local liParallel = nn.Concat(2)

    -- Subbranch with Spatial Transformer.
    local localizedNet = nn.Sequential()
    localizedNet:add(network.createSpatialTransformer(false, true, true, hL/2, 64, GPU))
    localizedNet:add(conv(64, 64, 3, 2))
    localizedNet:add(conv(64, 32, 3, 1))
    local localizedNetSize = 32*hL/2/2*wL/2/2
    localizedNet:add(nn.Reshape(localizedNetSize))

    -- Subbranch without Spatial Transformer.
    local unlocalizedNet = nn.Sequential()
    unlocalizedNet:add(conv(64, 64, 3, 2))
    unlocalizedNet:add(conv(64, 128, 3, 2))
    unlocalizedNet:add(conv(128, 128, 3, 2))
    local unlocalizedNetSize = 128*hL/2/2/2/2*wL/2/2/2/2
    unlocalizedNet:add(nn.Reshape(unlocalizedNetSize))

    liParallel:add(localizedNet):add(unlocalizedNet)
    lastImage:add(liParallel)
    local lastImageSize = localizedNetSize + unlocalizedNetSize
    lastImage:add(nn.Reshape(lastImageSize))

    -- Merge all three branches.
    local parallel = nn.ParallelTable():add(actionHistory):add(imageHistory):add(lastImage)
    net:add(parallel)
    net:add(nn.JoinTable(1, 1))
    net:add(nn.Dropout(0.25)) -- maybe not needed

    -- Apply a linear hidden layer to the merged branche's results.
    net:add(nn.Linear(actionHistorySize + imageHistorySize + lastImageSize, 512))
    net:add(nn.BatchNormalization(512))
    net:add(nn.Tanh()) -- maybe replaceable by LReLU
    net:add(nn.L2Penalty(1e-8)) -- maybe not needed
    net:add(nn.Dropout(0.5)) -- maybe not needed

    -- Predict rewards by action.
    net:add(nn.Linear(512, #actions.ACTIONS_NETWORK))
    net:add(nn.L1Penalty(1e-8, false)) -- maybe not needed

    if GPU then
        net:add(nn.Copy('torch.CudaTensor', 'torch.FloatTensor', true, true))
        net:cuda()
    end

    -- Function to initialize the weights and biases.
    local function weights_init(m)
        -- dontInitialize flag is set for one linear layer of the Spatial Transformer
        -- (though nowadays probably not necessary anymore)
        if m.dontInitialize == nil then
            local name = torch.type(m)

            if name:find('Convolution') then
                m.weight:normal(0.0, 0.05)
                -- check if layer is unbiased (:noBias())
                if m.bias ~= nil then
                    m.bias:normal(0.0, 0.05)
                end
            elseif name:find('Linear') then
                m.weight:normal(0.0, 0.05)
                -- check if layer is unbiased (:noBias())
                if m.bias ~= nil then
                    m.bias:normal(0.0, 0.05)
                end
            elseif name:find('BatchNormalization') then
                if m.weight then m.weight:normal(0.3, 0.03) end
                if m.bias then m.bias:fill(0) end
            end
        end
    end
    net:apply(weights_init)

    return net
end

function network.createQ11()
    function conv(nbInputPlanes, nbOutputPlanes, ks, stride)
        return nn.Sequential()
                :add(cudnn.SpatialConvolution(nbInputPlanes, nbOutputPlanes, ks, ks, stride, stride, (ks-1)/2, (ks-1)/2))
                :add(nn.SpatialBatchNormalization(nbOutputPlanes))
                :add(nn.LeakyReLU(0.2, true))
    end

    local cH, hH, wH = unpack(IMG_DIMENSIONS_Q_HISTORY)
    local cL, hL, wL = unpack(IMG_DIMENSIONS_Q_LAST)
    local net = nn.Sequential()

    -- Action history branch. deals with previously chosen action(-ids).
    local actionHistory = nn.Sequential()
    if GPU then actionHistory:add(nn.Copy('torch.FloatTensor', 'torch.CudaTensor', true, true)) end
    actionHistory:add(nn.Reshape(STATES_PER_EXAMPLE * #actions.ACTIONS_NETWORK))
    actionHistory:add(nn.Linear(STATES_PER_EXAMPLE * #actions.ACTIONS_NETWORK, 32))
    actionHistory:add(nn.LeakyReLU(0.2, true))
    local actionHistorySize = 32

    -- State history branch. Deals with previously seen states (as small images).
    -- Note that this branch also retrieves the current state as a small image.
    local imageHistory = nn.Sequential()
    if GPU then imageHistory:add(nn.Copy('torch.FloatTensor', 'torch.CudaTensor', true, true)) end
    imageHistory:add(conv(STATES_PER_EXAMPLE, 64, 3, 1))
    imageHistory:add(conv(64, 64, 5, 2))
    imageHistory:add(conv(64, 64, 5, 4))
    local imageHistorySize = 64 * hH/2/4 * wH/2/4
    imageHistory:add(nn.Reshape(imageHistorySize))

    -- Last image branch. Sees only the current state as a larger image.
    local lastImage = nn.Sequential()
    if GPU then lastImage:add(nn.Copy('torch.FloatTensor', 'torch.CudaTensor', true, true)) end
    lastImage:add(conv(cL, 256, 5, 1))
    lastImage:add(nn.SpatialMaxPooling(2, 2))
    lastImage:add(conv(256, 64, 3, 1))

    local liParallel = nn.Concat(2)

    -- Subbranch with Spatial Transformer.
    local localizedNet = nn.Sequential()
    localizedNet:add(network.createSpatialTransformer2(false, true, true, hL/2, 64, GPU))
    localizedNet:add(conv(64, 64, 3, 2))
    localizedNet:add(conv(64, 32, 3, 1))
    local localizedNetSize = 32*hL/2/2*wL/2/2
    localizedNet:add(nn.Reshape(localizedNetSize))

    -- Subbranch without Spatial Transformer.
    local unlocalizedNet = nn.Sequential()
    unlocalizedNet:add(conv(64, 64, 3, 2))
    unlocalizedNet:add(conv(64, 128, 3, 2))
    unlocalizedNet:add(conv(128, 128, 3, 2))
    local unlocalizedNetSize = 128*hL/2/2/2/2*wL/2/2/2/2
    unlocalizedNet:add(nn.Reshape(unlocalizedNetSize))

    liParallel:add(localizedNet):add(unlocalizedNet)
    lastImage:add(liParallel)
    local lastImageSize = localizedNetSize + unlocalizedNetSize
    lastImage:add(nn.Reshape(lastImageSize))

    -- Merge all three branches.
    local parallel = nn.ParallelTable():add(actionHistory):add(imageHistory):add(lastImage)
    net:add(parallel)
    net:add(nn.JoinTable(1, 1))
    --net:add(nn.Dropout(0.25))

    -- Apply a linear hidden layer to the merged branche's results.
    net:add(nn.Linear(actionHistorySize + imageHistorySize + lastImageSize, 512))
    net:add(nn.BatchNormalization(512))
    net:add(nn.LeakyReLU(0.2, true))
    --net:add(nn.L2Penalty(1e-8))
    --net:add(nn.Dropout(0.5))

    -- Predict rewards by action.
    net:add(nn.Linear(512, #actions.ACTIONS_NETWORK))
    --net:add(nn.L1Penalty(1e-8, false))

    if GPU then
        net:add(nn.Copy('torch.CudaTensor', 'torch.FloatTensor', true, true))
        net:cuda()
    end

    -- Function to initialize the weights and biases.
    local function weights_init(m)
        -- dontInitialize flag is set for one linear layer of the Spatial Transformer
        -- (though nowadays probably not necessary anymore)
        if m.dontInitialize == nil then
            local name = torch.type(m)

            if name:find('Convolution') then
                m.weight:normal(0.0, 0.05)
                -- check if layer is unbiased (:noBias())
                if m.bias ~= nil then
                    m.bias:normal(0.0, 0.05)
                end
            elseif name:find('Linear') then
                m.weight:normal(0.0, 0.05)
                -- check if layer is unbiased (:noBias())
                if m.bias ~= nil then
                    m.bias:normal(0.0, 0.05)
                end
            elseif name:find('BatchNormalization') then
                if m.weight then m.weight:normal(0.3, 0.03) end
                if m.bias then m.bias:fill(0) end
            end
        end
    end
    net:apply(weights_init)

    return net
end

-- Perform a forward/backward training pass for a given batch.
function network.forwardBackwardBatch(batchInput, batchTarget)
    local loss
    Q:training()
    local feval = function(x)
        local input = batchInput
        local target = batchTarget

        GRAD_PARAMETERS:zero() -- reset gradients
        -- forward pass
        local batchOutput = Q:forward(input)
        local err = CRITERION:forward(batchOutput, target)
        --  backward pass
        local df_do = CRITERION:backward(batchOutput, target)
        Q:backward(input, df_do)
        --errG = network.l1(PARAMETERS, GRAD_PARAMETERS, err, 1e-6)
        err = network.l2(PARAMETERS, GRAD_PARAMETERS, err, Q_L2_NORM)
        network.clamp(GRAD_PARAMETERS, Q_CLAMP)

        loss = err
        return err, GRAD_PARAMETERS
    end
    optim.adam(feval, PARAMETERS, OPTCONFIG, OPTSTATE)
    --optim.adagrad(feval, PARAMETERS, {}, OPTSTATE)
    --optim.sgd(feval, PARAMETERS, {learningRate=0.00001}, OPTSTATE)
    --optim.sgd(feval, PARAMETERS, OPTCONFIG, OPTSTATE)
    --optim.rmsprop(feval, PARAMETERS, {}, OPTSTATE)
    Q:evaluate()
    return loss
end

-- Compute the loss of a given batch without training on it.
function network.batchToLoss(batchInput, batchTarget)
    Q:evaluate()
    local batchOutput = Q:forward(batchInput)
    local err = CRITERION:forward(batchOutput, batchTarget)
    err = network.l2(PARAMETERS, nil, err, Q_L2_NORM)
    return err
end

-- Approximate the Q-value of a specific action for a given state chain.
function network.approximateActionValue(stateChain, action)
    assert(action ~= nil)
    local values = network.approximateActionValues(stateChain)
    --return {arrows = values[action.arrows], buttons = values[action.buttons]}
    return (values[action.arrow] + values[action.button])/2
end

-- Approximate the Q-values of all actions for a list of state chains.
-- TODO fully replace this function with approximateActionValuesBatch().
function network.approximateActionValues(stateChain)
    assert(#stateChain == STATES_PER_EXAMPLE)

    local out = network.approximateActionValuesBatch({stateChain})
    out = out[1]

    return out
end

-- Approximate Q-values (for all actions) for many chains of states.
function network.approximateActionValuesBatch(stateChains, net)
    net = net or Q
    net:evaluate()
    local batchInput = network.stateChainsToBatchInput(stateChains)
    local result = net:forward(batchInput)
    local out = {}
    for i=1,result:size(1) do
        out[i] = network.networkVectorToActionValues(result[i])
    end

    local plotPoints = {}
    for i=1,result[1]:size(1) do
        table.insert(plotPoints, {i, result[1][i]})
    end
    display.plot(plotPoints, {win=41, labels={'Action', 'Q(s,a)'}, title='Q(s,a) using network output action positions'})

    return out
end

-- Predict the best action (maximal reward) for a chain of states.
-- @returns tuple (Action, action value)
function network.approximateBestAction(stateChain)
    local values = network.approximateActionValues(stateChain)

    local bestArrowIdx = nil
    local bestArrowValue = nil
    for key, value in pairs(values) do
        if actions.isArrowsActionIdx(key) then
            if bestArrowIdx == nil or value > bestArrowValue then
                bestArrowIdx = key
                bestArrowValue = value
            end
        end
    end

    local bestButtonIdx = nil
    local bestButtonValue = nil
    for key, value in pairs(values) do
        if actions.isButtonsActionIdx(key) then
            if bestButtonIdx == nil or value > bestButtonValue then
                bestButtonIdx = key
                bestButtonValue = value
            end
        end
    end

    -- dont use pairs() here for iteration, because order of items is important for display.plot()
    local plotPointsArrows = {}
    for i=1,#actions.ACTIONS_ARROWS do
        local key = actions.ACTIONS_ARROWS[i]
        table.insert(plotPointsArrows, {key, values[key]})
    end
    display.plot(plotPointsArrows, {win=39, labels={'Action', 'Q(s,a)'}, title='Q(s,a) using emulator action IDs (Arrows)'})

    local plotPointsButtons = {}
    for i=1,#actions.ACTIONS_BUTTONS do
        local key = actions.ACTIONS_BUTTONS[i]
        table.insert(plotPointsButtons, {key, values[key]})
    end
    display.plot(plotPointsButtons, {win=40, labels={'Action', 'Q(s,a)'}, title='Q(s,a) using emulator action IDs (Buttons)'})

    return Action.new(bestArrowIdx, bestButtonIdx), (bestArrowValue+bestButtonValue)/2
end

-- Predicts the best actions (maximal reward) for many chains of states.
-- @returns List of (Action, action value)
function network.approximateBestActionsBatch(stateChains, net)
    net = net or Q
    local result = {}
    local valuesBatch = network.approximateActionValuesBatch(stateChains, net)
    for i=1,#valuesBatch do
        local values = valuesBatch[i]

        local bestArrowIdx = nil
        local bestArrowValue = nil
        for key, value in pairs(values) do
            if actions.isArrowsActionIdx(key) then
                if bestArrowIdx == nil or value > bestArrowValue then
                    bestArrowIdx = key
                    bestArrowValue = value
                end
            end
        end

        local bestButtonIdx = nil
        local bestButtonValue = nil
        for key, value in pairs(values) do
            if actions.isButtonsActionIdx(key) then
                if bestButtonIdx == nil or value > bestButtonValue then
                    bestButtonIdx = key
                    bestButtonValue = value
                end
            end
        end

        local oneResult = {action = Action.new(bestArrowIdx, bestButtonIdx), value = (bestArrowValue+bestButtonValue)/2}
        table.insert(result, oneResult)
    end
    return result
end

-- Converts many chains of states to a batch for training/validation.
-- @returns tuple (input/X, target/Y)
function network.stateChainsToBatch(stateChains)
    local batchInput = network.stateChainsToBatchInput(stateChains)
    local batchTarget = network.stateChainsToBatchTarget(stateChains)
    return batchInput, batchTarget
end

-- Converts many chains of states to the input/x of a batch.
-- @returns Table {action history tensor, state history tensor, last state tensor}
function network.stateChainsToBatchInput(stateChains)
    local batchSize = #stateChains
    local batchInput = {
        torch.zeros(#stateChains, STATES_PER_EXAMPLE, #actions.ACTIONS_NETWORK),
        torch.zeros(#stateChains, STATES_PER_EXAMPLE, IMG_DIMENSIONS_Q_HISTORY[2], IMG_DIMENSIONS_Q_HISTORY[3]),
        torch.zeros(#stateChains, IMG_DIMENSIONS_Q_LAST[1], IMG_DIMENSIONS_Q_LAST[2], IMG_DIMENSIONS_Q_LAST[3])
    }
    for i=1,#stateChains do
        local stateChain = stateChains[i]
        local example = network.stateChainToInput(stateChain)
        batchInput[1][i] = example[1]
        batchInput[2][i] = example[2]
        batchInput[3][i] = example[3]
    end

    return batchInput
end

-- Converts many chains of states to their batch targets (Y).
-- @returns Tensor
function network.stateChainsToBatchTarget(stateChains)
    local batchSize = #stateChains
    local batchTarget = torch.zeros(batchSize, #actions.ACTIONS_NETWORK)
    for i=1,#stateChains do
        local stateChain = stateChains[i]
        batchTarget[i] = network.stateChainToTarget(stateChain)
    end

    return batchTarget
end

-- Converts a single state chain to a batch input.
-- @returns {action history tensor, image history tensor, last image tensor}
function network.stateChainToInput(stateChain)
    assert(#stateChain == STATES_PER_EXAMPLE)
    local actionChain = torch.zeros(#stateChain, #actions.ACTIONS_NETWORK)
    for i=1,#stateChain do
        if stateChain[i].action ~= nil then
            actionChain[i] = network.actionToNetworkVector(stateChain[i].action)
        end
    end

    local imageHistory = torch.zeros(#stateChain, IMG_DIMENSIONS_Q_HISTORY[2], IMG_DIMENSIONS_Q_HISTORY[3])
    for i=1,#stateChain do
        local screenDec = states.decompressScreen(stateChain[i].screen)
        screenDec = util.toImageDimensions(screenDec, IMG_DIMENSIONS_Q_HISTORY)
        imageHistory[i] = screenDec
    end

    local lastImage = util.toImageDimensions(states.decompressScreen(stateChain[#stateChain].screen), IMG_DIMENSIONS_Q_LAST)

    local example = {actionChain, imageHistory, lastImage}

    return example
end

-- Converts a single state chain to a batch target.
-- @returns Tensor
function network.stateChainToTarget(stateChain)
    local lastState = stateChain[#stateChain]
    local action = lastState.action
    local vec = network.actionToNetworkVector(action)
    vec:mul(rewards.getSumForTraining(lastState.reward))
    return vec
end

-- Converts an Action object to a two-hot-vector that can be used as target for a batch.
-- (Two, because there are two choices: Arrow and other button.)
-- @returns Tensor
function network.actionToNetworkVector(action)
    local vec = torch.zeros(#actions.ACTIONS_NETWORK)
    vec[network.getNetworkPositionOfActionIdx(action.arrow)] = 1
    vec[network.getNetworkPositionOfActionIdx(action.button)] = 1
    return vec
end

-- Converts a network output to a table [action index => reward].
-- @returns Table
function network.networkVectorToActionValues(vec)
    local out = {}
    for i=1,vec:size(1) do
        out[actions.ACTIONS_NETWORK[i]] = vec[i]
    end
    return out
end

-- Returns the position (1..N) of an action (specified by its index) among the output neurons of the network.
-- @returns integer
function network.getNetworkPositionOfActionIdx(actionIdx)
    assert(actionIdx ~= nil)
    for i=1,#actions.ACTIONS_NETWORK do
        if actions.ACTIONS_NETWORK[i] == actionIdx then
            return i
        end
    end
    error("action not found: " .. actionIdx)
end

-- Clamps/truncates gradient values.
function network.clamp(gradParameters, clampValue)
    if clampValue ~= 0 then
        gradParameters:clamp((-1)*clampValue, clampValue)
    end
end

-- Applies a L1 norm to the parameters of the network.
function network.l1(parameters, gradParameters, lossValue, l1weight)
    if l1weight ~= 0 then
        lossValue = lossValue + l1weight * torch.norm(parameters, 1)
        if gradParameters ~= nil then
            gradParameters:add(torch.sign(parameters):mul(l1Weight))
        end
    end
    return lossValue
end

-- Applies a L2 norm to the parameters of the network.
function network.l2(parameters, gradParameters, lossValue, l2weight)
    if l2weight ~= 0 then
        lossValue = lossValue + l2weight * torch.norm(parameters, 2)^2/2
        if gradParameters ~= nil then
            gradParameters:add(parameters:clone():mul(l2weight))
        end
    end
    return lossValue
end

-- Returns the number of parameters/weights in a network.
function network.getNumberOfParameters(net)
    local nparams = 0
    local dModules = net:listModules()
    for i=1,#dModules do
        if dModules[i].weight ~= nil then
            nparams = nparams + dModules[i].weight:nElement()
        end
    end
    return nparams
end

-- Displays a batch of images.
-- TODO does this still work? is this still used?
function network.displayBatch(images, windowId, title, width)
    --print("network.displayBatch start")
    local nExamples, nStates, h, w = images:size(1), images:size(2), images:size(3), images:size(4)
    local imgsDisp = torch.zeros(nExamples*nStates, 1, h, w)
    local counter = 1
    for i=1,nExamples do
        for j=1,nStates do
            imgsDisp[counter] = images[i][j]
            counter = counter + 1
        end
    end

    local out = image.toDisplayTensor{input=imgsDisp, nrow=STATES_PER_EXAMPLE, padding=1}

    title = title or string.format("Images")
    if width then
        display.image(out, {win=windowId, width=width, title=title})
    else
        display.image(out, {win=windowId, title=title})
    end
    --print("network.displayBatch end")
end

-- Plot measured losses per N batches
function network.plotAverageLoss(lossData, clampTo)
    clampTo = clampTo or 10
    local losses = {}
    for i=1,#lossData do
        local entry = lossData[i]
        table.insert(losses, {entry[1], math.min(entry[2], clampTo), math.min(entry[3], clampTo)})
    end
    display.plot(losses, {win=4, labels={'batch group', 'training', 'validation'}, title='Average loss per batch'})
end


-- Prepares a network for saving to file by shrinking/removing unnecessary data.
-- Works in-place, i.e. does not return anything.
-- from https://github.com/torch/DEPRECEATED-torch7-distro/issues/47
-- Resize the output, gradInput, etc temporary tensors to zero (so that the on disk size is smaller)
function network.prepareNetworkForSave(node)
    -- from https://github.com/torch/DEPRECEATED-torch7-distro/issues/47
    function zeroDataSize(data)
        if type(data) == 'table' then
            for i = 1, #data do
                data[i] = zeroDataSize(data[i])
            end
        elseif type(data) == 'userdata' then
            data = torch.Tensor():typeAs(data)
        end
        return data
    end

    if node.output ~= nil then
        node.output = zeroDataSize(node.output)
    end
    if node.gradInput ~= nil then
        node.gradInput = zeroDataSize(node.gradInput)
    end
    if node.finput ~= nil then
        node.finput = zeroDataSize(node.finput)
    end
    -- Recurse on nodes with 'modules'
    if (node.modules ~= nil) then
        if (type(node.modules) == 'table') then
            for i = 1, #node.modules do
                local child = node.modules[i]
                network.prepareNetworkForSave(child)
            end
        end
    end
    collectgarbage()
end

-- Create a new spatial transformer network.
-- NOTE: This is adapted to this specific project. Rotation is likely not working anymore.
-- From: https://github.com/Moodstocks/gtsrb.torch/blob/master/networks.lua
-- @param allow_rotation Whether to allow the spatial transformer to rotate the image.
-- @param allow_scaling Whether to allow the spatial transformer to scale (zoom) the image.
-- @param allow_translation Whether to allow the spatial transformer to translate (shift) the image.
-- @param input_size Height/width of input images.
-- @param input_channels Number of channels of the image.
-- @param cuda Whether to activate cuda mode.
function network.createSpatialTransformer(allow_rotation, allow_scaling, allow_translation, input_size, input_channels, cuda)
    if cuda == nil then
        cuda = true
    end

    -- Get number of params and initial state
    local init_bias = {}
    local nbr_params = 0
    if allow_rotation then
        nbr_params = nbr_params + 1
        init_bias[nbr_params] = 0
    end
    if allow_scaling then
        nbr_params = nbr_params + 1
        init_bias[nbr_params] = 0.5
    end
    if allow_translation then
        nbr_params = nbr_params + 2
        init_bias[nbr_params-1] = 0
        init_bias[nbr_params] = 0
    end
    if nbr_params == 0 then
        -- fully parametrized case
        nbr_params = 6
        init_bias = {1,0,0,
                     0,1,0}
    end

    -- Create localization network
    local net = nn.Sequential()
    --net:add(nn.PrintSize("localizer"))
    net:add(nn.SpatialConvolution(input_channels, 32, 5, 5, 2, 2, (5-1)/2)) --> 16x16
    net:add(nn.SpatialBatchNormalization(32))
    net:add(nn.LeakyReLU(0.2, true))
    net:add(nn.SpatialDropout(0.1))

    net:add(nn.SpatialConvolution(32, 64, 3, 3, 2, 2, (3-1)/2)) --> 8x8
    net:add(nn.SpatialBatchNormalization(64))
    net:add(nn.LeakyReLU(0.2, true))
    net:add(nn.SpatialDropout(0.1))

    net:add(nn.SpatialConvolution(64, 64, 3, 3, 2, 2, (3-1)/2)) --> 4x4
    net:add(nn.SpatialBatchNormalization(64))
    net:add(nn.LeakyReLU(0.2, true))
    net:add(nn.Dropout(0.5))

    local newHeight = input_size/2/2/2
    net:add(nn.Reshape(64 * newHeight * newHeight)) -- must be reshape, nn.View converts (1, 16*H*W) to (16*H*W)

    net:add(nn.Linear(64 * newHeight * newHeight, 256):noBias())
    net:add(nn.BatchNormalization(256))
    net:add(nn.LeakyReLU(0.2, true))
    net:add(nn.Dropout(0.5))

    local classifier = nn.Linear(256, nbr_params)
    net:add(classifier)
    net:add(nn.Tanh())
    net:add(nn.L2Penalty(1e-3, false)) -- let the ST only change the area of focus if it really pays off

    -- We keep the classifier's output close to zero most of the time,
    -- and then add init_bias to its output. init_bias is configured so that
    -- the area of focus is in the center of the image and has ~50% of the size
    -- of the image.
    local constant_tnsr = torch.Tensor(init_bias)
    net:add(nn.AddConstantTensor(constant_tnsr))
    classifier:noBias()
    classifier.weight:zero()
    classifier.dontInitialize = true

    local localization_network = net

    -- Create the actual module structure
    -- branch1 is basically an identity matrix
    -- branch2 estimates the necessary rotation/scaling/translation (above localization network)
    -- They both feed into the BilinearSampler, which transforms the image
    local ct = nn.ConcatTable()
    local branch1 = nn.Sequential()
    branch1:add(nn.Transpose({3,4},{2,4}))
    -- see (1) below
    if cuda then
        branch1:add(nn.Copy('torch.CudaTensor', 'torch.FloatTensor', true, true))
    end
    local branch2 = nn.Sequential()
    branch2:add(localization_network)
    branch2:add(nn.AffineTransformMatrixGenerator(allow_rotation, allow_scaling, allow_translation))
    branch2:add(nn.AffineGridGeneratorBHWD(input_size, input_size))
    -- see (1) below
    if cuda then
        branch2:add(nn.Copy('torch.CudaTensor', 'torch.FloatTensor', true, true))
    end
    ct:add(branch1)
    ct:add(branch2)

    local st = nn.Sequential()
    st:add(ct)
    local sampler = nn.BilinearSamplerBHWD()
    -- (1)
    -- The sampler lead to non-reproducible results on GPU
    -- We want to always keep it on CPU
    -- This does no lead to slowdown of the training
    if cuda then
        sampler:type('torch.FloatTensor')
        -- make sure it will not go back to the GPU when we call
        -- ":cuda()" on the network later
        sampler.type = function(type) return self end
        --st:add(nn.PrintSize())
        st:add(sampler)
        st:add(nn.Copy('torch.FloatTensor','torch.CudaTensor', true, true))
    else
        st:add(sampler)
    end
    st:add(nn.Transpose({2,4},{3,4}))

    return st
end

-- Create a new spatial transformer network.
-- NOTE: This is adapted to this specific project. Rotation is likely not working anymore.
-- From: https://github.com/Moodstocks/gtsrb.torch/blob/master/networks.lua
-- @param allow_rotation Whether to allow the spatial transformer to rotate the image.
-- @param allow_scaling Whether to allow the spatial transformer to scale (zoom) the image.
-- @param allow_translation Whether to allow the spatial transformer to translate (shift) the image.
-- @param input_size Height/width of input images.
-- @param input_channels Number of channels of the image.
-- @param cuda Whether to activate cuda mode.
function network.createSpatialTransformer2(allow_rotation, allow_scaling, allow_translation, input_size, input_channels, cuda)
    if cuda == nil then
        cuda = true
    end

    -- Get number of params and initial state
    local init_bias = {}
    local nbr_params = 0
    if allow_rotation then
        nbr_params = nbr_params + 1
        init_bias[nbr_params] = 0
    end
    if allow_scaling then
        nbr_params = nbr_params + 1
        init_bias[nbr_params] = 0.5
    end
    if allow_translation then
        nbr_params = nbr_params + 2
        init_bias[nbr_params-1] = 0
        init_bias[nbr_params] = 0
    end
    if nbr_params == 0 then
        -- fully parametrized case
        nbr_params = 6
        init_bias = {1,0,0,
                     0,1,0}
    end

    -- Create localization network
    local net = nn.Sequential()
    --net:add(nn.PrintSize("localizer"))
    net:add(nn.SpatialConvolution(input_channels, 32, 5, 5, 2, 2, (5-1)/2)) --> 16x16
    net:add(nn.SpatialBatchNormalization(32))
    net:add(nn.LeakyReLU(0.2, true))
    net:add(nn.SpatialDropout(0.1))

    net:add(nn.SpatialConvolution(32, 64, 3, 3, 2, 2, (3-1)/2)) --> 8x8
    net:add(nn.SpatialBatchNormalization(64))
    net:add(nn.LeakyReLU(0.2, true))
    net:add(nn.SpatialDropout(0.1))

    net:add(nn.SpatialConvolution(64, 64, 3, 3, 2, 2, (3-1)/2)) --> 4x4
    net:add(nn.SpatialBatchNormalization(64))
    net:add(nn.LeakyReLU(0.2, true))
    net:add(nn.Dropout(0.2))

    local newHeight = input_size/2/2/2
    net:add(nn.Reshape(64 * newHeight * newHeight)) -- must be reshape, nn.View converts (1, 16*H*W) to (16*H*W)

    net:add(nn.Linear(64 * newHeight * newHeight, 256):noBias())
    net:add(nn.BatchNormalization(256))
    net:add(nn.LeakyReLU(0.2, true))
    net:add(nn.Dropout(0.25))

    local classifier = nn.Linear(256, nbr_params)
    net:add(classifier)
    net:add(nn.Tanh())
    net:add(nn.L2Penalty(1e-10, false)) -- let the ST only change the area of focus if it really pays off

    -- We keep the classifier's output close to zero most of the time,
    -- and then add init_bias to its output. init_bias is configured so that
    -- the area of focus is in the center of the image and has ~50% of the size
    -- of the image.
    local constant_tnsr = torch.Tensor(init_bias)
    net:add(nn.AddConstantTensor(constant_tnsr))
    classifier:noBias()
    classifier.weight:zero()
    classifier.dontInitialize = true

    local localization_network = net

    -- Create the actual module structure
    -- branch1 is basically an identity matrix
    -- branch2 estimates the necessary rotation/scaling/translation (above localization network)
    -- They both feed into the BilinearSampler, which transforms the image
    local ct = nn.ConcatTable()
    local branch1 = nn.Sequential()
    branch1:add(nn.Transpose({3,4},{2,4}))
    -- see (1) below
    if cuda then
        branch1:add(nn.Copy('torch.CudaTensor', 'torch.FloatTensor', true, true))
    end
    local branch2 = nn.Sequential()
    branch2:add(localization_network)
    branch2:add(nn.AffineTransformMatrixGenerator(allow_rotation, allow_scaling, allow_translation))
    branch2:add(nn.AffineGridGeneratorBHWD(input_size, input_size))
    -- see (1) below
    if cuda then
        branch2:add(nn.Copy('torch.CudaTensor', 'torch.FloatTensor', true, true))
    end
    ct:add(branch1)
    ct:add(branch2)

    local st = nn.Sequential()
    st:add(ct)
    local sampler = nn.BilinearSamplerBHWD()
    -- (1)
    -- The sampler lead to non-reproducible results on GPU
    -- We want to always keep it on CPU
    -- This does no lead to slowdown of the training
    if cuda then
        sampler:type('torch.FloatTensor')
        -- make sure it will not go back to the GPU when we call
        -- ":cuda()" on the network later
        sampler.type = function(type) return self end
        --st:add(nn.PrintSize())
        st:add(sampler)
        st:add(nn.Copy('torch.FloatTensor','torch.CudaTensor', true, true))
    else
        st:add(sampler)
    end
    st:add(nn.Transpose({2,4},{3,4}))

    return st
end

return network
