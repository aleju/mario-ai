require 'torch'
require 'paths'
require 'nn'
require 'layers.Residual'
require 'layers.PrintSize'
require 'layers.BilinearSamplerBHWD2'
require 'stn'

local network = {}

function network.load()
    local fp = "learned/network.th7"
    if paths.filep(fp) then
        local savedData = torch.load(fp)
        return savedData
    else
        print("[INFO] Could not load previously saved network, file does not exist.")
        return nil
    end
end

function network.save()
    network.prepareNetworkForSave(Q)
    local fp = "learned/network.th7"
    torch.save(fp, Q)
end

--[[
function network.createQ()
    local net = nn.Sequential()
    if GPU then
        net:add(nn.Copy('torch.FloatTensor', 'torch.CudaTensor', true, true))
    end
    net:add(nn.SpatialConvolution(IMG_DIMENSIONS_Q[1], 8, 5, 5, 2, 2, (5-1)/2, (5-1)/2))
    net:add(nn.ReLU())
    net:add(nn.SpatialConvolution(8, 128, 7, 7, 4, 4, (7-1)/2, (7-1)/2))
    net:add(nn.ReLU())
    net:add(nn.SpatialConvolution(128, 64, 3, 3, 2, 2, (3-1)/2, (3-1)/2))
    net:add(nn.ReLU())
    net:add(nn.SpatialConvolution(64, 32, 3, 3, 2, 2, (3-1)/2, (3-1)/2))
    net:add(nn.ReLU())
    net:add(nn.Dropout(0.5))
    local outSize = 32 * (IMG_DIMENSIONS_Q[2]/2/2/2/2/2) * 2*(IMG_DIMENSIONS_Q[3]/2/2/2/2/2)
    net:add(nn.Reshape(outSize))
    --net:add(nn.Dropout(0.25))
    net:add(nn.Linear(outSize, 256))
    --net:add(nn.BatchNormalization(128))
    net:add(nn.ReLU())
    --net:add(nn.Linear(128, 128))
    --net:add(nn.BatchNormalization(128))
    --net:add(nn.LeakyReLU(0.2))
    net:add(nn.Linear(256, #actions.ACTIONS_NETWORK))

    if GPU then
        net:add(nn.Copy('torch.CudaTensor', 'torch.FloatTensor', true, true))
        net:cuda()
    end

    return net
end
--]]

--[[
function network.createQ()
    local c, h, w = unpack(IMG_DIMENSIONS_Q)
    local net = nn.Sequential()
    if GPU then
        net:add(nn.Copy('torch.FloatTensor', 'torch.CudaTensor', true, true))
    end
    local concat = nn.Concat(2)
    local conv1 = nn.Sequential():add(nn.SpatialConvolution(c, 32, 1, 7, 1, 1, (1-1)/2, (7-1)/2))
    local conv2 = nn.Sequential():add(nn.SpatialConvolution(c, 64, 1, 5, 1, 1, (1-1)/2, (5-1)/2))
    local conv3 = nn.Sequential():add(nn.SpatialConvolution(c, 16, 5, 1, 1, 1, (5-1)/2, (1-1)/2))
    local conv4 = nn.Sequential():add(nn.SpatialConvolution(c, 16, 7, 1, 1, 1, (7-1)/2, (1-1)/2))
    local conv5 = nn.Sequential():add(nn.SpatialConvolution(c, 16, 3, 3, 1, 1, (3-1)/2, (3-1)/2))
    concat:add(conv1):add(conv2):add(conv3):add(conv4):add(conv5)
    net:add(concat)
    net:add(nn.SpatialBatchNormalization(32+64+16+16+16))
    net:add(nn.ReLU())
    net:add(nn.SpatialMaxPooling(2, 2))
    --net:add(nn.L1Penalty(1e-5))

    net:add(nn.SpatialConvolution(32+64+16+16+16, 32, 1, 1, 1, 1, (1-1)/2, (1-1)/2))
    net:add(nn.SpatialBatchNormalization(32))
    net:add(nn.ReLU())
    net:add(nn.SpatialMaxPooling(2, 2))

    net:add(nn.SpatialConvolution(32, 32, 3, 3, 2, 2, (3-1)/2, (3-1)/2))
    net:add(nn.SpatialBatchNormalization(32))
    net:add(nn.ReLU())
    net:add(nn.SpatialMaxPooling(2, 2))
    --net:add(nn.Dropout(0.5))

    local outSize = 32 * (IMG_DIMENSIONS_Q[2]/2/2/2/2) * 2*(IMG_DIMENSIONS_Q[3]/2/2/2/2)
    net:add(nn.Reshape(outSize))
    net:add(nn.Linear(outSize, 128))
    net:add(nn.BatchNormalization(128))
    net:add(nn.Tanh())
    --net:add(nn.Dropout(0.25))
    net:add(nn.Linear(128, #actions.ACTIONS_NETWORK))

    if GPU then
        net:add(nn.Copy('torch.CudaTensor', 'torch.FloatTensor', true, true))
        net:cuda()
    end

    return net
end
--]]

--[[
function network.createQ()
    local c, h, w = unpack(IMG_DIMENSIONS_Q)
    local net = nn.Sequential()
    if GPU then
        net:add(nn.Copy('torch.FloatTensor', 'torch.CudaTensor', true, true))
    end

    net:add(nn.SpatialConvolution(STATES_PER_EXAMPLE, 16, 5, 5, 2, 2, (5-1)/2, (5-1)/2))
    --net:add(nn.SpatialBatchNormalization(16))
    net:add(nn.LeakyReLU(0.2))
    net:add(nn.SpatialConvolution(16, 32, 3, 3, 1, 1, (3-1)/2, (3-1)/2))
    --net:add(nn.SpatialBatchNormalization(32))
    net:add(nn.LeakyReLU(0.2))
    net:add(nn.SpatialMaxPooling(2, 2))
    net:add(nn.SpatialConvolution(32, 64, 3, 3, 1, 1, (3-1)/2, (3-1)/2))
    --net:add(nn.SpatialBatchNormalization(64))
    net:add(nn.LeakyReLU(0.2))
    net:add(nn.SpatialMaxPooling(2, 2))
    net:add(nn.SpatialConvolution(64, 64, 3, 3, 1, 1, (3-1)/2, (3-1)/2))
    --net:add(nn.SpatialBatchNormalization(64))
    net:add(nn.LeakyReLU(0.2))
    net:add(nn.SpatialMaxPooling(2, 2))

    local outSize = 64 * (IMG_DIMENSIONS_Q[2]/2/2/2/2) * (IMG_DIMENSIONS_Q[3]/2/2/2/2)
    net:add(nn.Reshape(outSize))
    net:add(nn.Linear(outSize, 256))
    --net:add(nn.BatchNormalization(256))
    net:add(nn.LeakyReLU(0.2))
    net:add(nn.Linear(256, 128))
    --net:add(nn.BatchNormalization(128))
    net:add(nn.Tanh())
    --net:add(nn.Dropout(0.25))
    net:add(nn.Linear(128, #actions.ACTIONS_NETWORK))

    if GPU then
        net:add(nn.Copy('torch.CudaTensor', 'torch.FloatTensor', true, true))
        net:cuda()
    end

    return net
end
--]]

--[[
function network.createQ()
    local c, h, w = unpack(IMG_DIMENSIONS_Q)
    local net = nn.Sequential()
    if GPU then
        net:add(nn.Copy('torch.FloatTensor', 'torch.CudaTensor', true, true))
    end

    net:add(nn.SpatialConvolution(STATES_PER_EXAMPLE, 64, 5, 5, 2, 2, (5-1)/2, (5-1)/2))
    net:add(nn.LeakyReLU(0.2))
    net:add(nn.SpatialConvolution(64, 128, 3, 3, 2, 2, (3-1)/2, (3-1)/2))
    net:add(nn.LeakyReLU(0.2))
    net:add(nn.SpatialConvolution(128, 256, 3, 3, 2, 2, (3-1)/2, (3-1)/2))
    net:add(nn.LeakyReLU(0.2))
    local outSize = 256 * (h/4/2) * (w/4/2)
    net:add(nn.Reshape(outSize))
    net:add(nn.Linear(outSize, 512))
    net:add(nn.LeakyReLU(0.2))
    net:add(nn.Linear(512, #actions.ACTIONS_NETWORK))

    if GPU then
        net:add(nn.Copy('torch.CudaTensor', 'torch.FloatTensor', true, true))
        net:cuda()
    end

    return net
end
--]]

--[[
function network.createQ()
    function createResidual(nbInputPlanes, ks)
        local activation = function () return nn.LeakyReLU(0.2, true) end
        ks = ks or 3

        local seq = nn.Sequential()
        local inner = nn.Sequential()
        inner:add(cudnn.SpatialConvolution(nbInputPlanes, nbInputPlanes/2, 1, 1, 1, 1, (1-1)/2, (1-1)/2))
        --inner:add(nn.SpatialBatchNormalization(nbInputPlanes/2))
        inner:add(activation())
        inner:add(cudnn.SpatialConvolution(nbInputPlanes/2, nbInputPlanes/2, ks, ks, 1, 1, (ks-1)/2, (ks-1)/2))
        --inner:add(nn.SpatialBatchNormalization(nbInputPlanes/2))
        inner:add(activation())
        inner:add(cudnn.SpatialConvolution(nbInputPlanes/2, nbInputPlanes, 1, 1, 1, 1, (1-1)/2, (1-1)/2))
        --inner:add(nn.SpatialBatchNormalization(nbInputPlanes))

        --seq:add(nn.StochasticDepth(0.2, inner))
        --seq:add(nn.StochasticDepth(pl, inner))
        seq:add(nn.Residual(inner))
        seq:add(activation())

        return seq
    end

    local c, h, w = unpack(IMG_DIMENSIONS_Q)
    local net = nn.Sequential()
    if GPU then
        net:add(nn.Copy('torch.FloatTensor', 'torch.CudaTensor', true, true))
    end

    net:add(nn.SpatialConvolution(STATES_PER_EXAMPLE, 64, 3, 3, 1, 1, (3-1)/2, (3-1)/2))
    net:add(nn.LeakyReLU(0.2))
    net:add(createResidual(64, 3))
    net:add(createResidual(64, 3))
    net:add(nn.SpatialConvolution(64, 128, 3, 3, 2, 2, (3-1)/2, (3-1)/2))
    net:add(nn.LeakyReLU(0.2))
    net:add(createResidual(128, 3))
    net:add(nn.SpatialConvolution(128, 128, 3, 3, 2, 2, (3-1)/2, (3-1)/2))
    net:add(nn.LeakyReLU(0.2))
    net:add(createResidual(128, 3))
    local outSize = 128 * (h/2/2) * (w/2/2)
    net:add(nn.Reshape(outSize))
    net:add(nn.Linear(outSize, 512))
    net:add(nn.LeakyReLU(0.2))
    net:add(nn.Linear(512, #actions.ACTIONS_NETWORK))

    if GPU then
        net:add(nn.Copy('torch.CudaTensor', 'torch.FloatTensor', true, true))
        net:cuda()
    end

    return net
end
--]]

function network.createQ1()
    function createResidual(nbInputPlanes, ks)
        local activation = function () return nn.LeakyReLU(0.2, true) end
        ks = ks or 3

        local seq = nn.Sequential()
        local inner = nn.Sequential()
        inner:add(cudnn.SpatialConvolution(nbInputPlanes, nbInputPlanes/4, 1, 1, 1, 1, (1-1)/2, (1-1)/2))
        inner:add(nn.SpatialBatchNormalization(nbInputPlanes/4))
        inner:add(activation())
        inner:add(cudnn.SpatialConvolution(nbInputPlanes/4, nbInputPlanes/4, ks, ks, 1, 1, (ks-1)/2, (ks-1)/2))
        inner:add(nn.SpatialBatchNormalization(nbInputPlanes/4))
        inner:add(activation())
        inner:add(cudnn.SpatialConvolution(nbInputPlanes/4, nbInputPlanes, 1, 1, 1, 1, (1-1)/2, (1-1)/2))
        inner:add(nn.SpatialBatchNormalization(nbInputPlanes))

        --seq:add(nn.StochasticDepth(0.2, inner))
        --seq:add(nn.StochasticDepth(pl, inner))
        seq:add(nn.Residual(inner))
        seq:add(activation())

        return seq
    end

    local cH, hH, wH = unpack(IMG_DIMENSIONS_Q_HISTORY)
    local cL, hL, wL = unpack(IMG_DIMENSIONS_Q_LAST)
    local net = nn.Sequential()

    local actionHistory = nn.Sequential()
    if GPU then actionHistory:add(nn.Copy('torch.FloatTensor', 'torch.CudaTensor', true, true)) end
    actionHistory:add(nn.Reshape(STATES_PER_EXAMPLE * #actions.ACTIONS_NETWORK))
    actionHistory:add(nn.Linear(STATES_PER_EXAMPLE * #actions.ACTIONS_NETWORK, 64))
    --actionHistory:add(nn.L1Penalty(1e-6))
    actionHistory:add(nn.BatchNormalization(64))
    actionHistory:add(nn.LeakyReLU(0.2))
    local actionHistorySize = 64

    local imageHistory = nn.Sequential()
    if GPU then imageHistory:add(nn.Copy('torch.FloatTensor', 'torch.CudaTensor', true, true)) end
    imageHistory:add(cudnn.SpatialConvolution(STATES_PER_EXAMPLE, 128, 5, 5, 2, 2, (5-1)/2, (5-1)/2))
    --imageHistory:add(nn.L1Penalty(1e-7))
    imageHistory:add(nn.SpatialBatchNormalization(128))
    imageHistory:add(nn.LeakyReLU(0.2))
    imageHistory:add(cudnn.SpatialConvolution(128, 32, 3, 3, 2, 2, (3-1)/2, (3-1)/2))
    --imageHistory:add(nn.L1Penalty(1e-6))
    imageHistory:add(nn.SpatialBatchNormalization(32))
    imageHistory:add(nn.LeakyReLU(0.2))
    imageHistory:add(cudnn.SpatialConvolution(32, 32, 3, 3, 1, 1, (3-1)/2, (3-1)/2))
    --imageHistory:add(nn.L1Penalty(1e-6))
    imageHistory:add(nn.SpatialBatchNormalization(32))
    imageHistory:add(nn.LeakyReLU(0.2))
    imageHistory:add(nn.Dropout(0.25))
    local imageHistorySize = 32 * hH/4 * wH/4
    imageHistory:add(nn.Reshape(imageHistorySize))

    local lastImage = nn.Sequential()
    if GPU then lastImage:add(nn.Copy('torch.FloatTensor', 'torch.CudaTensor', true, true)) end
    --[[
    lastImage:add(cudnn.SpatialConvolution(cL, 32, 3, 3, 1, 1, (3-1)/2, (3-1)/2))
    --lastImage:add(nn.L1Penalty(1e-8))
    lastImage:add(nn.SpatialBatchNormalization(32))
    lastImage:add(nn.LeakyReLU(0.2))
    lastImage:add(cudnn.SpatialConvolution(32, 32, 3, 3, 2, 2, (3-1)/2, (3-1)/2))
    --lastImage:add(nn.L1Penalty(1e-7))
    lastImage:add(nn.SpatialBatchNormalization(32))
    lastImage:add(nn.LeakyReLU(0.2))
    lastImage:add(cudnn.SpatialConvolution(32, 32, 3, 3, 2, 2, (3-1)/2, (3-1)/2))
    --lastImage:add(nn.L1Penalty(1e-6))
    lastImage:add(nn.SpatialBatchNormalization(32))
    lastImage:add(nn.LeakyReLU(0.2))
    lastImage:add(cudnn.SpatialConvolution(32, 32, 3, 3, 2, 2, (3-1)/2, (3-1)/2))
    --lastImage:add(nn.L1Penalty(1e-6))
    lastImage:add(nn.SpatialBatchNormalization(32))
    lastImage:add(nn.LeakyReLU(0.2))
    --lastImage:add(nn.Dropout(0.25))
    local lastImageSize = 32 * hL/2/2/2 * wL/2/2/2
    lastImage:add(nn.Reshape(lastImageSize))
    --]]
    lastImage:add(cudnn.SpatialConvolution(cL, 256, 7, 7, 4, 4, (7-1)/2, (7-1)/2))
    --lastImage:add(nn.L1Penalty(1e-8))
    lastImage:add(nn.SpatialBatchNormalization(256))
    lastImage:add(nn.LeakyReLU(0.2))
    lastImage:add(cudnn.SpatialConvolution(256, 64, 1, 1, 1, 1, (1-1)/2, (1-1)/2))
    --lastImage:add(nn.L1Penalty(1e-7))
    lastImage:add(nn.SpatialBatchNormalization(64))
    lastImage:add(nn.LeakyReLU(0.2))
    lastImage:add(createResidual(64, 3))
    lastImage:add(createResidual(64, 3))
    lastImage:add(createResidual(64, 3))
    lastImage:add(createResidual(64, 3))
    lastImage:add(nn.Dropout(0.25))
    local lastImageSize = 64 * hL/2/2 * wL/2/2
    lastImage:add(nn.Reshape(lastImageSize))

    local parallel = nn.ParallelTable():add(actionHistory):add(imageHistory):add(lastImage)
    net:add(parallel)
    net:add(nn.JoinTable(1, 1))

    net:add(nn.Linear(actionHistorySize + imageHistorySize + lastImageSize, 512))
    net:add(nn.L1Penalty(1e-6))
    --net:add(nn.BatchNormalization(512))
    net:add(nn.Tanh())
    net:add(nn.Linear(512, #actions.ACTIONS_NETWORK))

    if GPU then
        net:add(nn.Copy('torch.CudaTensor', 'torch.FloatTensor', true, true))
        net:cuda()
    end

    local function weights_init(m)
        local name = torch.type(m)

        if name:find('Convolution') then
            m.weight:normal(0.0, 0.02)
            --m.bias:fill(0)
            m.bias:normal(0.0, 0.02)
        elseif name:find('Linear') then
            m.weight:normal(0.0, 0.02)
            --m.bias:fill(0)
            m.bias:normal(0.0, 0.02)
        elseif name:find('BatchNormalization') then
            if m.weight then m.weight:normal(0.2, 0.03) end
            if m.bias then m.bias:fill(0) end
        end
    end
    net:apply(weights_init)

    return net
end

function network.createQ2()
    function createResidual(nbInputPlanes, ks)
        local activation = function () return nn.LeakyReLU(0.2, true) end
        ks = ks or 3

        local seq = nn.Sequential()
        local inner = nn.Sequential()
        inner:add(cudnn.SpatialConvolution(nbInputPlanes, nbInputPlanes/4, 1, 1, 1, 1, (1-1)/2, (1-1)/2))
        inner:add(nn.SpatialBatchNormalization(nbInputPlanes/4))
        inner:add(activation())
        inner:add(cudnn.SpatialConvolution(nbInputPlanes/4, nbInputPlanes/4, ks, ks, 1, 1, (ks-1)/2, (ks-1)/2))
        inner:add(nn.SpatialBatchNormalization(nbInputPlanes/4))
        inner:add(activation())
        inner:add(cudnn.SpatialConvolution(nbInputPlanes/4, nbInputPlanes, 1, 1, 1, 1, (1-1)/2, (1-1)/2))
        inner:add(nn.SpatialBatchNormalization(nbInputPlanes))

        --seq:add(nn.StochasticDepth(0.2, inner))
        --seq:add(nn.StochasticDepth(pl, inner))
        seq:add(nn.Residual(inner))
        seq:add(activation())

        return seq
    end

    local cH, hH, wH = unpack(IMG_DIMENSIONS_Q_HISTORY)
    local cL, hL, wL = unpack(IMG_DIMENSIONS_Q_LAST)
    local net = nn.Sequential()

    local actionHistory = nn.Sequential()
    if GPU then actionHistory:add(nn.Copy('torch.FloatTensor', 'torch.CudaTensor', true, true)) end
    actionHistory:add(nn.Reshape(STATES_PER_EXAMPLE * #actions.ACTIONS_NETWORK))
    actionHistory:add(nn.Linear(STATES_PER_EXAMPLE * #actions.ACTIONS_NETWORK, 32))
    --actionHistory:add(nn.L1Penalty(1e-6))
    actionHistory:add(nn.BatchNormalization(32))
    actionHistory:add(nn.LeakyReLU(0.2))
    local actionHistorySize = 32

    local imageHistory = nn.Sequential()
    if GPU then imageHistory:add(nn.Copy('torch.FloatTensor', 'torch.CudaTensor', true, true)) end
    imageHistory:add(cudnn.SpatialConvolution(STATES_PER_EXAMPLE, 64, 5, 5, 1, 1, (5-1)/2, (5-1)/2))
    --imageHistory:add(nn.L1Penalty(1e-7))
    imageHistory:add(nn.SpatialBatchNormalization(64))
    imageHistory:add(nn.LeakyReLU(0.2))
    imageHistory:add(cudnn.SpatialConvolution(64, 32, 3, 3, 2, 2, (3-1)/2, (3-1)/2))
    --imageHistory:add(nn.L1Penalty(1e-6))
    imageHistory:add(nn.SpatialBatchNormalization(32))
    imageHistory:add(nn.LeakyReLU(0.2))
    imageHistory:add(cudnn.SpatialConvolution(32, 16, 3, 3, 2, 2, (3-1)/2, (3-1)/2))
    --imageHistory:add(nn.L1Penalty(1e-6))
    imageHistory:add(nn.SpatialBatchNormalization(16))
    imageHistory:add(nn.LeakyReLU(0.2))
    --imageHistory:add(nn.Dropout(0.25))
    local imageHistorySize = 16 * hH/4 * wH/4
    imageHistory:add(nn.Reshape(imageHistorySize))

    local lastImage = nn.Sequential()
    if GPU then lastImage:add(nn.Copy('torch.FloatTensor', 'torch.CudaTensor', true, true)) end
    --[[
    lastImage:add(cudnn.SpatialConvolution(cL, 32, 3, 3, 1, 1, (3-1)/2, (3-1)/2))
    --lastImage:add(nn.L1Penalty(1e-8))
    lastImage:add(nn.SpatialBatchNormalization(32))
    lastImage:add(nn.LeakyReLU(0.2))
    lastImage:add(cudnn.SpatialConvolution(32, 32, 3, 3, 2, 2, (3-1)/2, (3-1)/2))
    --lastImage:add(nn.L1Penalty(1e-7))
    lastImage:add(nn.SpatialBatchNormalization(32))
    lastImage:add(nn.LeakyReLU(0.2))
    lastImage:add(cudnn.SpatialConvolution(32, 32, 3, 3, 2, 2, (3-1)/2, (3-1)/2))
    --lastImage:add(nn.L1Penalty(1e-6))
    lastImage:add(nn.SpatialBatchNormalization(32))
    lastImage:add(nn.LeakyReLU(0.2))
    lastImage:add(cudnn.SpatialConvolution(32, 32, 3, 3, 2, 2, (3-1)/2, (3-1)/2))
    --lastImage:add(nn.L1Penalty(1e-6))
    lastImage:add(nn.SpatialBatchNormalization(32))
    lastImage:add(nn.LeakyReLU(0.2))
    --lastImage:add(nn.Dropout(0.25))
    local lastImageSize = 32 * hL/2/2/2 * wL/2/2/2
    lastImage:add(nn.Reshape(lastImageSize))
    --]]
    lastImage:add(cudnn.SpatialConvolution(cL, 64, 5, 5, 1, 1, (5-1)/2, (5-1)/2))
    --lastImage:add(nn.L1Penalty(1e-8))
    lastImage:add(nn.SpatialBatchNormalization(64))
    lastImage:add(nn.LeakyReLU(0.2))
    lastImage:add(cudnn.SpatialConvolution(64, 64, 3, 3, 2, 2, (3-1)/2, (3-1)/2))
    lastImage:add(nn.SpatialBatchNormalization(64))
    lastImage:add(nn.LeakyReLU(0.2))
    lastImage:add(cudnn.SpatialConvolution(64, 16, 3, 3, 2, 2, (3-1)/2, (3-1)/2))
    lastImage:add(nn.SpatialBatchNormalization(16))
    lastImage:add(nn.LeakyReLU(0.2))
    local lastImageSize = 16 * hL/2/2 * wL/2/2
    lastImage:add(nn.Reshape(lastImageSize))

    local parallel = nn.ParallelTable():add(actionHistory):add(imageHistory):add(lastImage)
    net:add(parallel)
    net:add(nn.JoinTable(1, 1))

    net:add(nn.Linear(actionHistorySize + imageHistorySize + lastImageSize, 256))
    net:add(nn.L1Penalty(1e-6))
    --net:add(nn.BatchNormalization(512))
    net:add(nn.LeakyReLU(0.2))
    net:add(nn.Linear(256, #actions.ACTIONS_NETWORK))

    if GPU then
        net:add(nn.Copy('torch.CudaTensor', 'torch.FloatTensor', true, true))
        net:cuda()
    end

    local function weights_init(m)
        local name = torch.type(m)

        if name:find('Convolution') then
            m.weight:normal(0.0, 0.02)
            --m.bias:fill(0)
            m.bias:normal(0.0, 0.02)
        elseif name:find('Linear') then
            m.weight:normal(0.0, 0.02)
            --m.bias:fill(0)
            m.bias:normal(0.0, 0.02)
        elseif name:find('BatchNormalization') then
            if m.weight then m.weight:normal(0.2, 0.03) end
            if m.bias then m.bias:fill(0) end
        end
    end
    net:apply(weights_init)

    return net
end

function network.createQ3()
    function createResidual(nbInputPlanes, ks)
        local activation = function () return nn.LeakyReLU(0.2, true) end
        ks = ks or 3

        local seq = nn.Sequential()
        local inner = nn.Sequential()
        inner:add(cudnn.SpatialConvolution(nbInputPlanes, nbInputPlanes/2, 1, 1, 1, 1, (1-1)/2, (1-1)/2))
        inner:add(nn.SpatialBatchNormalization(nbInputPlanes/2))
        inner:add(activation())
        inner:add(cudnn.SpatialConvolution(nbInputPlanes/2, nbInputPlanes/2, ks, ks, 1, 1, (ks-1)/2, (ks-1)/2))
        inner:add(nn.SpatialBatchNormalization(nbInputPlanes/2))
        inner:add(activation())
        inner:add(cudnn.SpatialConvolution(nbInputPlanes/2, nbInputPlanes, 1, 1, 1, 1, (1-1)/2, (1-1)/2))
        inner:add(nn.SpatialBatchNormalization(nbInputPlanes))

        seq:add(nn.Residual(inner))
        seq:add(activation())

        return seq
    end

    function createDimChanger(nbInputPlanes, nbOutputPlanes)
        return nn.Sequential()
                :add(cudnn.SpatialConvolution(nbInputPlanes, nbOutputPlanes, 1, 1, 1, 1, (1-1)/2, (1-1)/2))
                :add(nn.SpatialBatchNormalization(nbOutputPlanes))
                :add(nn.LeakyReLU(0.2, true))
    end

    local cH, hH, wH = unpack(IMG_DIMENSIONS_Q_HISTORY)
    local cL, hL, wL = unpack(IMG_DIMENSIONS_Q_LAST)
    local net = nn.Sequential()

    local actionHistory = nn.Sequential()
    if GPU then actionHistory:add(nn.Copy('torch.FloatTensor', 'torch.CudaTensor', true, true)) end
    actionHistory:add(nn.Reshape(STATES_PER_EXAMPLE * #actions.ACTIONS_NETWORK))
    actionHistory:add(nn.Linear(STATES_PER_EXAMPLE * #actions.ACTIONS_NETWORK, 32))
    actionHistory:add(nn.BatchNormalization(32))
    actionHistory:add(nn.LeakyReLU(0.2))
    local actionHistorySize = 32

    local imageHistory = nn.Sequential()
    if GPU then imageHistory:add(nn.Copy('torch.FloatTensor', 'torch.CudaTensor', true, true)) end
    imageHistory:add(cudnn.SpatialConvolution(STATES_PER_EXAMPLE, 64, 5, 5, 1, 1, (5-1)/2, (5-1)/2))
    imageHistory:add(nn.SpatialBatchNormalization(64))
    imageHistory:add(nn.LeakyReLU(0.2, true))
    imageHistory:add(createResidual(64, 3))
    imageHistory:add(nn.SpatialMaxPooling(2, 2)) -- 16x16
    imageHistory:add(createResidual(64, 3))
    imageHistory:add(nn.SpatialMaxPooling(2, 2)) -- 8x8
    imageHistory:add(createResidual(64, 3))
    imageHistory:add(nn.SpatialMaxPooling(2, 2)) -- 4x4
    local imageHistorySize = 64 * hH/2/2/2 * wH/2/2/2
    imageHistory:add(nn.Reshape(imageHistorySize))

    local lastImage = nn.Sequential()
    if GPU then lastImage:add(nn.Copy('torch.FloatTensor', 'torch.CudaTensor', true, true)) end
    lastImage:add(cudnn.SpatialConvolution(cL, 64, 5, 5, 1, 1, (5-1)/2, (5-1)/2))
    lastImage:add(nn.SpatialBatchNormalization(64))
    lastImage:add(nn.LeakyReLU(0.2, true))
    lastImage:add(createResidual(64, 3))
    lastImage:add(nn.SpatialMaxPooling(2, 2)) -- 32x32
    lastImage:add(createResidual(64, 3))
    lastImage:add(createDimChanger(64, 128))
    lastImage:add(nn.SpatialMaxPooling(2, 2)) -- 16x16
    lastImage:add(createResidual(128, 3))
    lastImage:add(createDimChanger(128, 256))
    lastImage:add(nn.SpatialMaxPooling(2, 2)) -- 8x8
    lastImage:add(createResidual(256, 3))
    lastImage:add(nn.SpatialMaxPooling(2, 2)) -- 4x4
    local lastImageSize = 256 * hL/2/2/2/2 * wL/2/2/2/2
    lastImage:add(nn.Reshape(lastImageSize))

    local parallel = nn.ParallelTable():add(actionHistory):add(imageHistory):add(lastImage)
    net:add(parallel)
    net:add(nn.JoinTable(1, 1))

    net:add(nn.Linear(actionHistorySize + imageHistorySize + lastImageSize, 128))
    net:add(nn.L1Penalty(1e-8))
    net:add(nn.LeakyReLU(0.2, true))
    net:add(nn.Linear(128, #actions.ACTIONS_NETWORK))

    if GPU then
        net:add(nn.Copy('torch.CudaTensor', 'torch.FloatTensor', true, true))
        net:cuda()
    end

    local function weights_init(m)
        local name = torch.type(m)

        if name:find('Convolution') then
            m.weight:normal(0.0, 0.02)
            --m.bias:fill(0)
            m.bias:normal(0.0, 0.02)
        elseif name:find('Linear') then
            m.weight:normal(0.0, 0.02)
            --m.bias:fill(0)
            m.bias:normal(0.0, 0.02)
        elseif name:find('BatchNormalization') then
            if m.weight then m.weight:normal(0.2, 0.03) end
            if m.bias then m.bias:fill(0) end
        end
    end
    net:apply(weights_init)

    return net
end

function network.createQ4()
    function createResidual(nbInputPlanes, ks)
        local activation = function () return nn.ELU(1.0, true) end
        ks = ks or 3

        local seq = nn.Sequential()
        local inner = nn.Sequential()
        inner:add(nn.SpatialBatchNormalization(nbInputPlanes))
        inner:add(activation())
        inner:add(cudnn.SpatialConvolution(nbInputPlanes, nbInputPlanes/2, ks, ks, 1, 1, (ks-1)/2, (ks-1)/2))
        inner:add(nn.SpatialBatchNormalization(nbInputPlanes/2))
        inner:add(activation())
        inner:add(cudnn.SpatialConvolution(nbInputPlanes/2, nbInputPlanes, ks, ks, 1, 1, (ks-1)/2, (ks-1)/2))
        seq:add(nn.Residual(inner))

        return seq
    end

    function createDimChanger(nbInputPlanes, nbOutputPlanes)
        return nn.Sequential()
                :add(cudnn.SpatialConvolution(nbInputPlanes, nbOutputPlanes, 1, 1, 1, 1, (1-1)/2, (1-1)/2))
                :add(nn.SpatialBatchNormalization(nbOutputPlanes))
                :add(nn.ELU(1.0, true))
    end

    function createPooling(nbInputPlanes, ks)
        local activation = function () return nn.ELU(1.0, true) end
        ks = ks or 3

        local seq = nn.Sequential()
        local inner = nn.Sequential()
        inner:add(nn.SpatialBatchNormalization(nbInputPlanes))
        inner:add(activation())
        inner:add(cudnn.SpatialConvolution(nbInputPlanes, nbInputPlanes, ks, ks, 2, 2, (ks-1)/2, (ks-1)/2))
        local skip = nn.SpatialAveragePooling(1, 1, 2, 2)
        seq:add(nn.Residual(inner, skip))

        return seq
    end

    local cH, hH, wH = unpack(IMG_DIMENSIONS_Q_HISTORY)
    local cL, hL, wL = unpack(IMG_DIMENSIONS_Q_LAST)
    local net = nn.Sequential()

    local actionHistory = nn.Sequential()
    if GPU then actionHistory:add(nn.Copy('torch.FloatTensor', 'torch.CudaTensor', true, true)) end
    actionHistory:add(nn.Reshape(STATES_PER_EXAMPLE * #actions.ACTIONS_NETWORK))
    actionHistory:add(nn.Linear(STATES_PER_EXAMPLE * #actions.ACTIONS_NETWORK, 32))
    actionHistory:add(nn.L1Penalty(1e-10))
    actionHistory:add(nn.ELU(1.0, true))
    local actionHistorySize = 32

    local imageHistory = nn.Sequential()
    if GPU then imageHistory:add(nn.Copy('torch.FloatTensor', 'torch.CudaTensor', true, true)) end
    imageHistory:add(cudnn.SpatialConvolution(STATES_PER_EXAMPLE, 64, 5, 5, 1, 1, (5-1)/2, (5-1)/2))
    imageHistory:add(nn.SpatialBatchNormalization(64))
    imageHistory:add(nn.ELU(1.0, true))
    imageHistory:add(createResidual(64, 3))
    imageHistory:add(createPooling(64)) -- 16x16
    imageHistory:add(createResidual(64, 3))
    imageHistory:add(createPooling(64)) -- 8x8
    imageHistory:add(createResidual(64, 3))
    imageHistory:add(createPooling(64)) -- 4x4
    imageHistory:add(nn.SpatialDropout(0.1))
    local imageHistorySize = 64 * hH/2/2/2 * wH/2/2/2
    imageHistory:add(nn.Reshape(imageHistorySize))

    local lastImage = nn.Sequential()
    if GPU then lastImage:add(nn.Copy('torch.FloatTensor', 'torch.CudaTensor', true, true)) end
    lastImage:add(cudnn.SpatialConvolution(cL, 64, 5, 5, 1, 1, (5-1)/2, (5-1)/2))
    lastImage:add(nn.SpatialBatchNormalization(64))
    lastImage:add(nn.ELU(1.0, true))
    lastImage:add(createResidual(64, 3))
    lastImage:add(createPooling(64)) -- 32x32
    lastImage:add(createResidual(64, 3))
    lastImage:add(createDimChanger(64, 128))
    lastImage:add(createPooling(128)) -- 16x16
    lastImage:add(createResidual(128, 3))
    lastImage:add(createPooling(128)) -- 8x8
    lastImage:add(createResidual(128, 3))
    lastImage:add(createPooling(128)) -- 4x4
    lastImage:add(nn.SpatialDropout(0.1))
    local lastImageSize = 128 * hL/2/2/2/2 * wL/2/2/2/2
    lastImage:add(nn.Reshape(lastImageSize))

    local parallel = nn.ParallelTable():add(actionHistory):add(imageHistory):add(lastImage)
    net:add(parallel)
    net:add(nn.JoinTable(1, 1))

    net:add(nn.Linear(actionHistorySize + imageHistorySize + lastImageSize, 256))
    net:add(nn.L1Penalty(1e-10))
    net:add(nn.ELU(1.0, true))
    net:add(nn.Dropout(0.1))
    net:add(nn.Linear(256, 256))
    net:add(nn.L1Penalty(1e-6))
    net:add(nn.ELU(1.0, true))
    net:add(nn.Dropout(0.1))
    net:add(nn.Linear(256, #actions.ACTIONS_NETWORK))

    if GPU then
        net:add(nn.Copy('torch.CudaTensor', 'torch.FloatTensor', true, true))
        net:cuda()
    end

    local function weights_init(m)
        local name = torch.type(m)

        if name:find('Convolution') then
            m.weight:normal(0.0, 0.03)
            --m.bias:fill(0)
            m.bias:normal(0.0, 0.05)
        elseif name:find('Linear') then
            m.weight:normal(0.0, 0.03)
            --m.bias:fill(0)
            m.bias:normal(0.0, 0.05)
        elseif name:find('BatchNormalization') then
            if m.weight then m.weight:normal(0.3, 0.03) end
            if m.bias then m.bias:fill(0) end
        end
    end
    net:apply(weights_init)

    return net
end

function network.createQ5()
    function conv(nbInputPlanes, nbOutputPlanes, ks, stride)
        return nn.Sequential()
                :add(cudnn.SpatialConvolution(nbInputPlanes, nbOutputPlanes, ks, ks, stride, stride, (ks-1)/2, (ks-1)/2))
                :add(nn.SpatialBatchNormalization(nbOutputPlanes))
                :add(nn.ELU(1.0, true))
    end

    local cH, hH, wH = unpack(IMG_DIMENSIONS_Q_HISTORY)
    local cL, hL, wL = unpack(IMG_DIMENSIONS_Q_LAST)
    local net = nn.Sequential()

    local actionHistory = nn.Sequential()
    if GPU then actionHistory:add(nn.Copy('torch.FloatTensor', 'torch.CudaTensor', true, true)) end
    actionHistory:add(nn.Reshape(STATES_PER_EXAMPLE * #actions.ACTIONS_NETWORK))
    actionHistory:add(nn.Linear(STATES_PER_EXAMPLE * #actions.ACTIONS_NETWORK, 32))
    actionHistory:add(nn.BatchNormalization(32))
    actionHistory:add(nn.ELU(1.0, true))
    local actionHistorySize = 32

    local imageHistory = nn.Sequential()
    if GPU then imageHistory:add(nn.Copy('torch.FloatTensor', 'torch.CudaTensor', true, true)) end
    imageHistory:add(conv(STATES_PER_EXAMPLE, 64, 5, 1))
    imageHistory:add(conv(64, 64, 3, 2))
    --imageHistory:add(nn.SpatialMaxPooling(2, 2)) -- 16x16
    imageHistory:add(conv(64, 64, 3, 2))
    --imageHistory:add(nn.SpatialMaxPooling(2, 2)) -- 8x8
    imageHistory:add(conv(64, 64, 3, 2))
    --imageHistory:add(nn.SpatialMaxPooling(2, 2)) -- 4x4
    local imageHistorySize = 64 * hH/2/2/2 * wH/2/2/2
    imageHistory:add(nn.Reshape(imageHistorySize))

    local lastImage = nn.Sequential()
    if GPU then lastImage:add(nn.Copy('torch.FloatTensor', 'torch.CudaTensor', true, true)) end
    lastImage:add(conv(cL, 64, 5, 1))
    lastImage:add(conv(64, 64, 3, 2))
    --lastImage:add(nn.SpatialMaxPooling(2, 2)) -- 32x32
    lastImage:add(conv(64, 128, 3, 2))
    --lastImage:add(nn.SpatialMaxPooling(2, 2)) -- 16x16
    lastImage:add(conv(128, 256, 3, 2))
    --lastImage:add(nn.SpatialMaxPooling(2, 2)) -- 8x8
    lastImage:add(conv(256, 256, 3, 2))
    --lastImage:add(nn.SpatialMaxPooling(2, 2)) -- 4x4
    local lastImageSize = 256 * hL/2/2/2/2 * wL/2/2/2/2
    lastImage:add(nn.Reshape(lastImageSize))

    local parallel = nn.ParallelTable():add(actionHistory):add(imageHistory):add(lastImage)
    net:add(parallel)
    net:add(nn.JoinTable(1, 1))

    net:add(nn.Linear(actionHistorySize + imageHistorySize + lastImageSize, 128))
    net:add(nn.L1Penalty(1e-8))
    net:add(nn.ELU(1.0, true))
    net:add(nn.Linear(128, #actions.ACTIONS_NETWORK))

    if GPU then
        net:add(nn.Copy('torch.CudaTensor', 'torch.FloatTensor', true, true))
        net:cuda()
    end

    local function weights_init(m)
        local name = torch.type(m)

        if name:find('Convolution') then
            m.weight:normal(0.0, 0.05)
            --m.bias:fill(0)
            m.bias:normal(0.0, 0.05)
        elseif name:find('Linear') then
            m.weight:normal(0.0, 0.05)
            --m.bias:fill(0)
            m.bias:normal(0.0, 0.05)
        elseif name:find('BatchNormalization') then
            if m.weight then m.weight:normal(0.3, 0.03) end
            if m.bias then m.bias:fill(0) end
        end
    end
    net:apply(weights_init)

    return net
end

function network.createQ6()
    function conv(nbInputPlanes, nbOutputPlanes, ks, stride)
        return nn.Sequential()
                :add(cudnn.SpatialConvolution(nbInputPlanes, nbOutputPlanes, ks, ks, stride, stride, (ks-1)/2, (ks-1)/2))
                :add(nn.SpatialBatchNormalization(nbOutputPlanes))
                :add(nn.ELU(1.0, true))
    end

    local cH, hH, wH = unpack(IMG_DIMENSIONS_Q_HISTORY)
    local cL, hL, wL = unpack(IMG_DIMENSIONS_Q_LAST)
    local net = nn.Sequential()

    local actionHistory = nn.Sequential()
    if GPU then actionHistory:add(nn.Copy('torch.FloatTensor', 'torch.CudaTensor', true, true)) end
    actionHistory:add(nn.Reshape(STATES_PER_EXAMPLE * #actions.ACTIONS_NETWORK))
    actionHistory:add(nn.Linear(STATES_PER_EXAMPLE * #actions.ACTIONS_NETWORK, 32))
    actionHistory:add(nn.BatchNormalization(32))
    actionHistory:add(nn.ELU(1.0, true))
    local actionHistorySize = 32

    local imageHistory = nn.Sequential()
    if GPU then imageHistory:add(nn.Copy('torch.FloatTensor', 'torch.CudaTensor', true, true)) end
    imageHistory:add(conv(STATES_PER_EXAMPLE, 64, 3, 1))
    imageHistory:add(conv(64, 64, 5, 2))
    imageHistory:add(conv(64, 64, 5, 4))
    local imageHistorySize = 64 * hH/2/4 * wH/2/4
    imageHistory:add(nn.Reshape(imageHistorySize))

    local lastImage = nn.Sequential()
    if GPU then lastImage:add(nn.Copy('torch.FloatTensor', 'torch.CudaTensor', true, true)) end
    lastImage:add(conv(cL, 64, 3, 1))
    lastImage:add(conv(64, 64, 5, 2))
    lastImage:add(conv(64, 128, 5, 4))
    local lastImageSize = 128 * hL/2/4 * wL/2/4
    lastImage:add(nn.Reshape(lastImageSize))

    local parallel = nn.ParallelTable():add(actionHistory):add(imageHistory):add(lastImage)
    net:add(parallel)
    net:add(nn.JoinTable(1, 1))

    net:add(nn.Linear(actionHistorySize + imageHistorySize + lastImageSize, 128))
    net:add(nn.L1Penalty(1e-8))
    net:add(nn.ELU(1.0, true))
    net:add(nn.Linear(128, #actions.ACTIONS_NETWORK))

    if GPU then
        net:add(nn.Copy('torch.CudaTensor', 'torch.FloatTensor', true, true))
        net:cuda()
    end

    local function weights_init(m)
        local name = torch.type(m)

        if name:find('Convolution') then
            m.weight:normal(0.0, 0.05)
            --m.bias:fill(0)
            m.bias:normal(0.0, 0.05)
        elseif name:find('Linear') then
            m.weight:normal(0.0, 0.05)
            --m.bias:fill(0)
            m.bias:normal(0.0, 0.05)
        elseif name:find('BatchNormalization') then
            if m.weight then m.weight:normal(0.3, 0.03) end
            if m.bias then m.bias:fill(0) end
        end
    end
    net:apply(weights_init)

    return net
end

function network.createQ7()
    function conv(nbInputPlanes, nbOutputPlanes, ks, stride)
        return nn.Sequential()
                :add(cudnn.SpatialConvolution(nbInputPlanes, nbOutputPlanes, ks, ks, stride, stride, (ks-1)/2, (ks-1)/2))
                :add(nn.SpatialBatchNormalization(nbOutputPlanes))
                :add(nn.LeakyReLU(0.2, true))
    end

    local cH, hH, wH = unpack(IMG_DIMENSIONS_Q_HISTORY)
    local cL, hL, wL = unpack(IMG_DIMENSIONS_Q_LAST)
    local net = nn.Sequential()

    local actionHistory = nn.Sequential()
    if GPU then actionHistory:add(nn.Copy('torch.FloatTensor', 'torch.CudaTensor', true, true)) end
    actionHistory:add(nn.Reshape(STATES_PER_EXAMPLE * #actions.ACTIONS_NETWORK))
    actionHistory:add(nn.Linear(STATES_PER_EXAMPLE * #actions.ACTIONS_NETWORK, 32))
    actionHistory:add(nn.BatchNormalization(32))
    actionHistory:add(nn.LeakyReLU(0.2, true))
    local actionHistorySize = 32

    local imageHistory = nn.Sequential()
    if GPU then imageHistory:add(nn.Copy('torch.FloatTensor', 'torch.CudaTensor', true, true)) end
    imageHistory:add(conv(STATES_PER_EXAMPLE, 64, 3, 1))
    imageHistory:add(conv(64, 64, 5, 2))
    imageHistory:add(conv(64, 64, 5, 4))
    local imageHistorySize = 64 * hH/2/4 * wH/2/4
    imageHistory:add(nn.Reshape(imageHistorySize))

    local lastImage = nn.Sequential()
    if GPU then lastImage:add(nn.Copy('torch.FloatTensor', 'torch.CudaTensor', true, true)) end
    lastImage:add(conv(cL, 64, 3, 1))
    lastImage:add(conv(64, 64, 5, 2))
    lastImage:add(conv(64, 128, 5, 4))
    local lastImageSize = 128 * hL/2/4 * wL/2/4
    lastImage:add(nn.Reshape(lastImageSize))

    local parallel = nn.ParallelTable():add(actionHistory):add(imageHistory):add(lastImage)
    net:add(parallel)
    net:add(nn.JoinTable(1, 1))

    net:add(nn.Linear(actionHistorySize + imageHistorySize + lastImageSize, 512))
    net:add(nn.L1Penalty(1e-8))
    net:add(nn.LeakyReLU(0.2, true))
    net:add(nn.Linear(512, #actions.ACTIONS_NETWORK))

    if GPU then
        net:add(nn.Copy('torch.CudaTensor', 'torch.FloatTensor', true, true))
        net:cuda()
    end

    local function weights_init(m)
        local name = torch.type(m)

        if name:find('Convolution') then
            m.weight:normal(0.0, 0.05)
            --m.bias:fill(0)
            m.bias:normal(0.0, 0.05)
        elseif name:find('Linear') then
            m.weight:normal(0.0, 0.05)
            --m.bias:fill(0)
            m.bias:normal(0.0, 0.05)
        elseif name:find('BatchNormalization') then
            if m.weight then m.weight:normal(0.3, 0.03) end
            if m.bias then m.bias:fill(0) end
        end
    end
    net:apply(weights_init)

    return net
end

function network.createQ8()
    function conv(nbInputPlanes, nbOutputPlanes, ks, stride)
        return nn.Sequential()
                :add(cudnn.SpatialConvolution(nbInputPlanes, nbOutputPlanes, ks, ks, stride, stride, (ks-1)/2, (ks-1)/2))
                :add(nn.SpatialBatchNormalization(nbOutputPlanes))
                :add(nn.ELU(1.0, true))
    end

    local cH, hH, wH = unpack(IMG_DIMENSIONS_Q_HISTORY)
    local cL, hL, wL = unpack(IMG_DIMENSIONS_Q_LAST)
    local net = nn.Sequential()

    local actionHistory = nn.Sequential()
    if GPU then actionHistory:add(nn.Copy('torch.FloatTensor', 'torch.CudaTensor', true, true)) end
    actionHistory:add(nn.Reshape(STATES_PER_EXAMPLE * #actions.ACTIONS_NETWORK))
    actionHistory:add(nn.Linear(STATES_PER_EXAMPLE * #actions.ACTIONS_NETWORK, 32))
    actionHistory:add(nn.ELU(1.0, true))
    local actionHistorySize = 32

    local imageHistory = nn.Sequential()
    if GPU then imageHistory:add(nn.Copy('torch.FloatTensor', 'torch.CudaTensor', true, true)) end
    imageHistory:add(conv(STATES_PER_EXAMPLE, 64, 3, 1))
    imageHistory:add(conv(64, 64, 5, 2))
    imageHistory:add(conv(64, 64, 5, 4))
    local imageHistorySize = 64 * hH/2/4 * wH/2/4
    imageHistory:add(nn.Reshape(imageHistorySize))

    local lastImage = nn.Sequential()
    if GPU then lastImage:add(nn.Copy('torch.FloatTensor', 'torch.CudaTensor', true, true)) end
    lastImage:add(conv(cL, 256, 7, 2))
    lastImage:add(conv(256, 128, 3, 2))
    lastImage:add(conv(128, 64, 3, 2))
    local lastImageSize = 64 * hL/2/2/2 * wL/2/2/2
    lastImage:add(nn.Reshape(lastImageSize))

    local parallel = nn.ParallelTable():add(actionHistory):add(imageHistory):add(lastImage)
    net:add(parallel)
    net:add(nn.JoinTable(1, 1))

    net:add(nn.Linear(actionHistorySize + imageHistorySize + lastImageSize, 512))
    net:add(nn.L1Penalty(1e-8))
    net:add(nn.ELU(1.0, true))
    net:add(nn.Linear(512, #actions.ACTIONS_NETWORK))

    if GPU then
        net:add(nn.Copy('torch.CudaTensor', 'torch.FloatTensor', true, true))
        net:cuda()
    end

    local function weights_init(m)
        local name = torch.type(m)

        if name:find('Convolution') then
            m.weight:normal(0.0, 0.05)
            --m.bias:fill(0)
            m.bias:normal(0.0, 0.05)
        elseif name:find('Linear') then
            m.weight:normal(0.0, 0.05)
            --m.bias:fill(0)
            m.bias:normal(0.0, 0.05)
        elseif name:find('BatchNormalization') then
            if m.weight then m.weight:normal(0.3, 0.03) end
            if m.bias then m.bias:fill(0) end
        end
    end
    net:apply(weights_init)

    return net
end

function network.createQ9()
    function conv(nbInputPlanes, nbOutputPlanes, ks, stride)
        return nn.Sequential()
                :add(cudnn.SpatialConvolution(nbInputPlanes, nbOutputPlanes, ks, ks, stride, stride, (ks-1)/2, (ks-1)/2))
                :add(nn.SpatialBatchNormalization(nbOutputPlanes))
                :add(nn.LeakyReLU(0.2, true))
    end

    local cH, hH, wH = unpack(IMG_DIMENSIONS_Q_HISTORY)
    local cL, hL, wL = unpack(IMG_DIMENSIONS_Q_LAST)
    local net = nn.Sequential()

    local actionHistory = nn.Sequential()
    if GPU then actionHistory:add(nn.Copy('torch.FloatTensor', 'torch.CudaTensor', true, true)) end
    actionHistory:add(nn.Reshape(STATES_PER_EXAMPLE * #actions.ACTIONS_NETWORK))
    actionHistory:add(nn.Linear(STATES_PER_EXAMPLE * #actions.ACTIONS_NETWORK, 32))
    actionHistory:add(nn.LeakyReLU(0.2, true))
    local actionHistorySize = 32

    local imageHistory = nn.Sequential()
    if GPU then imageHistory:add(nn.Copy('torch.FloatTensor', 'torch.CudaTensor', true, true)) end
    imageHistory:add(conv(STATES_PER_EXAMPLE, 64, 3, 1))
    imageHistory:add(conv(64, 64, 5, 2))
    imageHistory:add(conv(64, 64, 5, 4))
    local imageHistorySize = 64 * hH/2/4 * wH/2/4
    imageHistory:add(nn.Reshape(imageHistorySize))

    local lastImage = nn.Sequential()
    if GPU then lastImage:add(nn.Copy('torch.FloatTensor', 'torch.CudaTensor', true, true)) end
    lastImage:add(conv(cL, 256, 7, 2))
    lastImage:add(conv(256, 128, 3, 2))
    lastImage:add(conv(128, 64, 3, 2))
    local lastImageSize = 64 * hL/2/2/2 * wL/2/2/2
    lastImage:add(nn.Reshape(lastImageSize))

    local parallel = nn.ParallelTable():add(actionHistory):add(imageHistory):add(lastImage)
    net:add(parallel)
    net:add(nn.JoinTable(1, 1))

    net:add(nn.Linear(actionHistorySize + imageHistorySize + lastImageSize, 512))
    net:add(nn.L1Penalty(1e-8))
    net:add(nn.LeakyReLU(0.2, true))
    net:add(nn.Linear(512, #actions.ACTIONS_NETWORK))

    if GPU then
        net:add(nn.Copy('torch.CudaTensor', 'torch.FloatTensor', true, true))
        net:cuda()
    end

    local function weights_init(m)
        local name = torch.type(m)

        if name:find('Convolution') then
            m.weight:normal(0.0, 0.05)
            --m.bias:fill(0)
            m.bias:normal(0.0, 0.05)
        elseif name:find('Linear') then
            m.weight:normal(0.0, 0.05)
            --m.bias:fill(0)
            m.bias:normal(0.0, 0.05)
        elseif name:find('BatchNormalization') then
            if m.weight then m.weight:normal(0.3, 0.03) end
            if m.bias then m.bias:fill(0) end
        end
    end
    net:apply(weights_init)

    return net
end

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

    local actionHistory = nn.Sequential()
    if GPU then actionHistory:add(nn.Copy('torch.FloatTensor', 'torch.CudaTensor', true, true)) end
    actionHistory:add(nn.Reshape(STATES_PER_EXAMPLE * #actions.ACTIONS_NETWORK))
    actionHistory:add(nn.Linear(STATES_PER_EXAMPLE * #actions.ACTIONS_NETWORK, 32))
    actionHistory:add(nn.LeakyReLU(0.2, true))
    local actionHistorySize = 32

    local imageHistory = nn.Sequential()
    if GPU then imageHistory:add(nn.Copy('torch.FloatTensor', 'torch.CudaTensor', true, true)) end
    imageHistory:add(conv(STATES_PER_EXAMPLE, 64, 3, 1))
    imageHistory:add(conv(64, 64, 5, 2))
    imageHistory:add(conv(64, 64, 5, 4))
    local imageHistorySize = 64 * hH/2/4 * wH/2/4
    imageHistory:add(nn.Reshape(imageHistorySize))

    local lastImage = nn.Sequential()
    if GPU then lastImage:add(nn.Copy('torch.FloatTensor', 'torch.CudaTensor', true, true)) end
    lastImage:add(conv(cL, 256, 5, 1))
    lastImage:add(nn.SpatialMaxPooling(2, 2))
    lastImage:add(conv(256, 64, 3, 1))
    local liParallel = nn.Concat(2)
    local localizedNet = nn.Sequential()
    localizedNet:add(network.createSpatialTransformer(false, true, true, hL/2, 64, GPU))
    localizedNet:add(conv(64, 64, 3, 2))
    localizedNet:add(conv(64, 32, 3, 1))
    local localizedNetSize = 32*hL/2/2*wL/2/2
    localizedNet:add(nn.Reshape(localizedNetSize))

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

    local parallel = nn.ParallelTable():add(actionHistory):add(imageHistory):add(lastImage)
    net:add(parallel)
    net:add(nn.JoinTable(1, 1))

    net:add(nn.Linear(actionHistorySize + imageHistorySize + lastImageSize, 512))
    net:add(nn.L1Penalty(1e-8))
    net:add(nn.LeakyReLU(0.2, true))
    net:add(nn.Linear(512, #actions.ACTIONS_NETWORK))

    if GPU then
        net:add(nn.Copy('torch.CudaTensor', 'torch.FloatTensor', true, true))
        net:cuda()
    end

    local function weights_init(m)
        local name = torch.type(m)

        if name:find('Convolution') then
            m.weight:normal(0.0, 0.05)
            --m.bias:fill(0)
            m.bias:normal(0.0, 0.05)
        elseif name:find('Linear') then
            m.weight:normal(0.0, 0.05)
            --m.bias:fill(0)
            m.bias:normal(0.0, 0.05)
        elseif name:find('BatchNormalization') then
            if m.weight then m.weight:normal(0.3, 0.03) end
            if m.bias then m.bias:fill(0) end
        end
    end
    net:apply(weights_init)

    return net
end

function network.createOrLoadQ()
    local loaded = network.load()
    if loaded == nil then
        return network.createQ10()
    else
        return loaded
    end
end

function network.forwardBackwardBatch(batchInput, batchTarget)
    local loss
    --print("fwbw start")
    Q:training()
    --network.displayBatch(batchInput, 1, "Training images for Q")
    local feval = function(x)
        --print("batchInput", batchInput:size(1), batchInput:size(2), batchInput:size(3), batchInput:size(4))
        --print("batchOutput", batchTarget:size(1), batchTarget:size(2))

        local input = batchInput --:clone()
        local target = batchTarget --:clone()

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
    --print("in fw/bw")
    --feval()
    --print("fwbw end")
    Q:evaluate()
    return loss
end

function network.batchToLoss(batchInput, batchTarget)
    Q:training()
    --Q:evaluate()
    local batchOutput = Q:forward(batchInput)
    local err = CRITERION:forward(batchOutput, batchTarget)
    err = network.l2(PARAMETERS, nil, err, Q_L2_NORM)
    return err
end

function network.approximateActionValue(stateChain, action)
    assert(action ~= nil)
    local values = network.approximateActionValues(stateChain)
    --return {arrows = values[action.arrows], buttons = values[action.buttons]}
    return (values[action.arrow] + values[action.button])/2
end

function network.approximateActionValues(stateChain)
    assert(#stateChain == STATES_PER_EXAMPLE)

    local out = network.approximateActionValuesBatch({stateChain})
    out = out[1]

    return out
end

function network.approximateActionValuesBatch(stateChains)
    Q:training()
    --Q:evaluate()
    local batchInput = network.stateChainsToBatchInput(stateChains)
    local result = Q:forward(batchInput)
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

function network.approximateBestAction(stateChain)
    local values = network.approximateActionValues(stateChain)
    --[[
    print("[network.approximateBestAction] #values:", #values)
    for key,value in pairs(values) do
        print(key, value)
    end
    print("----")
    --]]

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

    --local maxs, indices = torch.max(result[1])
    --print("maxs:", maxs)
    --print("indices:", indices)
    --print(string.format("Best Action: idx %d with %.4f", bestActionIdx, bestActionValue))
    --return {bestActionArrow, bestActionButton}, {bestArrowValue, bestButtonValue}
    return Action.new(bestArrowIdx, bestButtonIdx), (bestArrowValue+bestButtonValue)/2
end

function network.approximateBestActionsBatch(stateChains)
    local result = {}
    local valuesBatch = network.approximateActionValuesBatch(stateChains)
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

function network.stateChainsToBatch(stateChains)
    local batchInput = network.stateChainsToBatchInput(stateChains)
    local batchTarget = network.stateChainsToBatchTarget(stateChains)
    return batchInput, batchTarget
end

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

function network.stateChainsToBatchTarget(stateChains)
    local batchSize = #stateChains
    local batchTarget = torch.zeros(batchSize, #actions.ACTIONS_NETWORK)
    for i=1,#stateChains do
        local stateChain = stateChains[i]
        batchTarget[i] = network.stateChainToTarget(stateChain)
    end

    return batchTarget
end

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

function network.stateChainToTarget(stateChain)
    local lastState = stateChain[#stateChain]
    local action = lastState.action
    local vec = network.actionToNetworkVector(action)
    vec:mul(rewards.getSumForTraining(lastState.reward))
    return vec
end

function network.actionToNetworkVector(action)
    local vec = torch.zeros(#actions.ACTIONS_NETWORK)
    vec[network.getNetworkPositionOfActionIdx(action.arrow)] = 1
    vec[network.getNetworkPositionOfActionIdx(action.button)] = 1
    return vec
end

function network.networkVectorToActionValues(vec)
    local out = {}
    for i=1,vec:size(1) do
        out[actions.ACTIONS_NETWORK[i]] = vec[i]
    end
    return out
end

function network.getNetworkPositionOfActionIdx(actionIdx)
    assert(actionIdx ~= nil)
    for i=1,#actions.ACTIONS_NETWORK do
        if actions.ACTIONS_NETWORK[i] == actionIdx then
            return i
        end
    end
    error("action not found: " .. actionIdx)
end

function network.clamp(gradParameters, clampValue)
    if clampValue ~= 0 then
        gradParameters:clamp((-1)*clampValue, clampValue)
    end
end

function network.l1(parameters, gradParameters, lossValue, l1weight)
    if l1weight ~= 0 then
        lossValue = lossValue + l1weight * torch.norm(parameters, 1)
        if gradParameters ~= nil then
            gradParameters:add(torch.sign(parameters):mul(l1Weight))
        end
    end
    return lossValue
end

function network.l2(parameters, gradParameters, lossValue, l2weight)
    if l2weight ~= 0 then
        lossValue = lossValue + l2weight * torch.norm(parameters, 2)^2/2
        if gradParameters ~= nil then
            gradParameters:add(parameters:clone():mul(l2weight))
        end
    end
    return lossValue
end

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
-- From: https://github.com/Moodstocks/gtsrb.torch/blob/master/networks.lua
-- @param allow_rotation Whether to allow the spatial transformer to rotate the image.
-- @param allow_rotation Whether to allow the spatial transformer to scale (zoom) the image.
-- @param allow_rotation Whether to allow the spatial transformer to translate (shift) the image.
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
        init_bias[nbr_params] = 1
    end
    if allow_translation then
        nbr_params = nbr_params + 2
        init_bias[nbr_params-1] = 0
        init_bias[nbr_params] = 0
    end
    if nbr_params == 0 then
        -- fully parametrized case
        nbr_params = 6
        init_bias = {1,0,0,0,1,0}
    end

    -- Create localization network
    local net = nn.Sequential()
    --net:add(nn.PrintSize("localizer"))
    net:add(nn.SpatialConvolution(input_channels, 16, 3, 3, 2, 2, (3-1)/2))
    net:add(nn.SpatialBatchNormalization(16))
    net:add(nn.LeakyReLU(0.2, true))
    net:add(nn.SpatialConvolution(16, 16, 3, 3, 2, 2, (3-1)/2))
    net:add(nn.SpatialBatchNormalization(16))
    net:add(nn.LeakyReLU(0.2, true))

    local newHeight = input_size/2/2
    net:add(nn.Reshape(16 * newHeight * newHeight)) -- must be reshape, nn.View converts (1, 16*H*W) to (16*H*W)
    net:add(nn.Linear(16 * newHeight * newHeight, 64))
    net:add(nn.LeakyReLU(0.2, true))
    local classifier = nn.Linear(64, nbr_params)
    net:add(classifier)

    --net = require('weight-init')(net, 'heuristic')
    -- Initialize the localization network (see paper, A.3 section)
    classifier.weight:zero()
    classifier.bias = torch.Tensor(init_bias)

    local localization_network = net

    -- Create the actual module structure
    -- branch1 is basically an identity matrix
    -- branch2 estimates the necessary rotation/scaling/translation (above localization network)
    -- They both feed into the BilinearSampler, which transforms the image
    local ct = nn.ConcatTable()
    local branch1 = nn.Sequential()
    --branch1:add(nn.Identity())
    --branch1:add(nn.PrintSize())
    branch1:add(nn.Transpose({3,4},{2,4}))
    --branch1:add(nn.PrintSize())
    -- see (1) below
    if cuda then
        branch1:add(nn.Copy('torch.CudaTensor', 'torch.FloatTensor', true, true))
    end
    local branch2 = nn.Sequential()
    branch2:add(localization_network)
    --branch2:add(nn.PrintSize("after localizer"))
    branch2:add(nn.AffineTransformMatrixGenerator(allow_rotation, allow_scaling, allow_translation))
    --branch2:add(nn.PrintSize("after affine matrix gen"))
    branch2:add(nn.AffineGridGeneratorBHWD(input_size, input_size))
    --branch2:add(nn.PrintSize("after affine grid gen"))
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
