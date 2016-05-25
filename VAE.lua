-- Old VAE stuff, ended up not being used.
-- Most of the code is adapted from https://github.com/y0ast/VAE-Torch .

require 'torch'
require 'nn'
require 'nngraph'
require 'layers.GaussianCriterion'
require 'layers.KLDCriterion'
require 'layers.Sampler'

local VAE = {}
VAE.continuous = false

function VAE.createVAE()
    local input_size = IMG_DIMENSIONS_AE[1] * IMG_DIMENSIONS_AE[2] * IMG_DIMENSIONS_AE[3]
    local hidden_layer_size = 1024
    local latent_variable_size = 512

    local encoder = VAE.get_encoder(input_size, hidden_layer_size, latent_variable_size)
    local decoder = VAE.get_decoder(input_size, hidden_layer_size, latent_variable_size, VAE.continuous)

    local input = nn.Identity()()
    local mean, log_var = encoder(input):split(2)
    local z = nn.Sampler()({mean, log_var})

    local reconstruction = decoder(z)
    local model = nn.gModule({input},{reconstruction, mean, log_var})
    local criterion_reconstruction = nn.BCECriterion()
    criterion_reconstruction.sizeAverage = false

    local criterion_latent = nn.KLDCriterion()

    local parameters, gradients = model:getParameters()

    return model, criterion_latent, criterion_reconstruction, parameters, gradients
end


function VAE.get_encoder(input_size, hidden_layer_size, latent_variable_size)
     -- The Encoder
    local encoder = nn.Sequential()
    if GPU then
        encoder:add(nn.Copy('torch.FloatTensor', 'torch.CudaTensor', true, true))
    end
    encoder:add(nn.SpatialConvolution(IMG_DIMENSIONS_AE[1], 8, 5, 5, 2, 2, (5-1)/2, (5-1)/2))
    encoder:add(nn.SpatialBatchNormalization(8))
    encoder:add(nn.LeakyReLU(0.2, true))
    encoder:add(nn.SpatialConvolution(8, 16, 5, 5, 2, 2, (5-1)/2, (5-1)/2))
    encoder:add(nn.SpatialBatchNormalization(16))
    encoder:add(nn.LeakyReLU(0.2, true))
    encoder:add(nn.SpatialConvolution(16, 32, 5, 5, 2, 2, (5-1)/2, (5-1)/2))
    encoder:add(nn.SpatialBatchNormalization(32))
    encoder:add(nn.LeakyReLU(0.2, true))
    encoder:add(nn.SpatialConvolution(32, 64, 5, 5, 2, 2, (5-1)/2, (5-1)/2))
    encoder:add(nn.SpatialBatchNormalization(64))
    encoder:add(nn.LeakyReLU(0.2, true))
    --encoder:add(nn.Reshape(input_size))
    local outSize = 64 * IMG_DIMENSIONS_AE[2]/2/2/2/2 * IMG_DIMENSIONS_AE[3]/2/2/2/2
    encoder:add(nn.Reshape(outSize))
    --encoder:add(nn.Linear(input_size, hidden_layer_size))
    encoder:add(nn.Linear(outSize, hidden_layer_size))
    encoder:add(nn.BatchNormalization(hidden_layer_size))
    encoder:add(nn.LeakyReLU(0.2, true))

    --if GPU then
    --    encoder:add(nn.Copy('torch.CudaTensor', 'torch.FloatTensor', true, true))
    --end

    mean_logvar = nn.ConcatTable()
    if GPU then
        mean_logvar:add(nn.Sequential():add(nn.Linear(hidden_layer_size, latent_variable_size)):add(nn.Copy('torch.CudaTensor', 'torch.FloatTensor', true, true)))
        mean_logvar:add(nn.Sequential():add(nn.Linear(hidden_layer_size, latent_variable_size)):add(nn.Copy('torch.CudaTensor', 'torch.FloatTensor', true, true)))
    else
        mean_logvar:add(nn.Linear(hidden_layer_size, latent_variable_size))
        mean_logvar:add(nn.Linear(hidden_layer_size, latent_variable_size))
    end

    encoder:add(mean_logvar)

    if GPU then
        encoder:cuda()
    end

    return encoder
end

function VAE.get_decoder(input_size, hidden_layer_size, latent_variable_size, continuous)
    --local c, h, w = unpack(IMG_DIMENSIONS)

    -- The Decoder
    local decoder = nn.Sequential()
    if GPU then
        decoder:add(nn.Copy('torch.FloatTensor', 'torch.CudaTensor', true, true))
    end
    decoder:add(nn.Linear(latent_variable_size, hidden_layer_size))
    decoder:add(nn.BatchNormalization(hidden_layer_size))
    decoder:add(nn.LeakyReLU(0.2, true))

    if continuous then
        mean_logvar = nn.ConcatTable()
        mean_logvar:add(nn.Linear(hidden_layer_size, input_size))
        mean_logvar:add(nn.Linear(hidden_layer_size, input_size))
        decoder:add(mean_logvar)
    else
        decoder:add(nn.Linear(hidden_layer_size, input_size/2/2))
        decoder:add(nn.Sigmoid(true))
        decoder:add(nn.Reshape(IMG_DIMENSIONS_AE[1], IMG_DIMENSIONS_AE[2]/2, IMG_DIMENSIONS_AE[3]/2))
        decoder:add(nn.SpatialUpSamplingNearest(2))
        --[[
        local c, h, w = unpack(IMG_DIMENSIONS)
        decoder:add(nn.Linear(latent_variable_size, 16*h/2/2*w/2/2))
        decoder:add(nn.ReLU(true))
        decoder:add(nn.Reshape(16, h/2/2, w/2/2)) -- 16x32
        decoder:add(nn.SpatialUpSamplingNearest(2)) -- 32x64
        decoder:add(nn.SpatialConvolution(16, 32, 3, 3, 1, 1, (3-1)/2, (3-1)/2))
        decoder:add(nn.ReLU(true))
        decoder:add(nn.SpatialUpSamplingNearest(2)) -- 64x128
        decoder:add(nn.SpatialConvolution(32, 1, 3, 3, 1, 1, (3-1)/2, (3-1)/2))
        decoder:add(nn.Sigmoid(true))
        --]]
    end

    if GPU then
        decoder:add(nn.Copy('torch.CudaTensor', 'torch.FloatTensor', true, true))
        decoder:cuda()
    end

    return decoder
end

function VAE.train(inputs, model, criterionLatent, criterionReconstruction, parameters, gradParameters, optconfig, optstate)


    local opfunc = function(x)
        assert(inputs ~= nil)
        assert(model ~= nil)
        assert(criterionLatent ~= nil)
        assert(criterionReconstruction ~= nil)
        assert(parameters ~= nil)
        assert(gradParameters ~= nil)
        assert(optconfig ~= nil)
        assert(optstate ~= nil)

        if x ~= parameters then
            parameters:copy(x)
        end

        model:zeroGradParameters()
        local reconstruction, reconstruction_var, mean, log_var
        if VAE.continuous then
            reconstruction, reconstruction_var, mean, log_var = unpack(model:forward(inputs))
            reconstruction = {reconstruction, reconstruction_var}
        else
            reconstruction, mean, log_var = unpack(model:forward(inputs))
        end

        local err = criterionReconstruction:forward(reconstruction, inputs)
        local df_dw = criterionReconstruction:backward(reconstruction, inputs)

        local KLDerr = criterionLatent:forward(mean, log_var)
        local dKLD_dmu, dKLD_dlog_var = unpack(criterionLatent:backward(mean, log_var))

        if VAE.continuous then
            error_grads = {df_dw[1], df_dw[2], dKLD_dmu, dKLD_dlog_var}
        else
            error_grads = {df_dw, dKLD_dmu, dKLD_dlog_var}
        end

        model:backward(inputs, error_grads)

        local batchlowerbound = err + KLDerr

        print(string.format("[BATCH AE] lowerbound=%.8f", batchlowerbound))
        util.displayBatch(inputs, 10, "Training images for AE (input)")
        util.displayBatch(reconstruction, 11, "Training images for AE (output)")

        return batchlowerbound, gradParameters
    end

    local x, batchlowerbound = optim.adam(opfunc, parameters, optconfig, optstate)

    return batchlowerbound
end

return VAE
