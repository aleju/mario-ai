local util = {}

function util.isGamePaused()
    return gui.get_runmode() == "pause"
end

function util.setGameSpeedToVeryFast()
    settings.set_speed(400)
end

function util.setGameSpeedToFast()
    settings.set_speed(200)
end

function util.setGameSpeedToNormal()
    settings.set_speed(1)
end

function util.getRandomEntry(arr)
    return arr[math.random(#arr)]
end

function util.getCurrentScore()
    local score = lsne_memory.readsword("WRAM", 0x0f34) * 10
    --print("Score:", score)
    return score
end

--[[
function util.getLevel()
    local level = lsne_memory.readsword("WRAM", 0x00fe)
    --print("Level:", level)
    return level
end
--]]
function util.getLevel()
    local level = lsne_memory.readsword("WRAM", 0x13bf)
    --print("Level:", level)
    return level
end

function util.getPlayerX()
    local x = lsne_memory.readsword("WRAM", 0x0094)
    --print("X:", x)
    return x
end

-- 0 = level
-- 1 = black screen?
-- 2 = overworld
function util.getMarioGameStatus()
    local status = lsne_memory.readsword("WRAM", 0x0D9B)
    return status
end

--[[
function util.getLevelTimer()
    local value = lsne_memory.readsword("WRAM", 0x1493)
    return value
end
--]]

function util.isLevelBeaten()
    return util.getLevelBeatenStatus() == 1
end

function util.isGameOver()
    return util.getLevelBeatenStatus() == 128
end

-- 0 = not beaten
-- 1 = beaten ?
-- 128 = game over
function util.getLevelBeatenStatus()
    local value = lsne_memory.readsword("WRAM", 0x0DD5)
    return value
end

--[[
function util.isSlidingDownFlagpole()
    --Player "float" state
    -- 0x00 - Standing on solid/else
    -- 0x01 - Airborn by jumping
    -- 0x02 - Airborn by walking of a ledge
    -- 0x03 - Sliding down flagpole
    local value = lsne_memory.readsword("WRAM", 0x001D)
    return value == 0x03
end
--]]

function util.getCountLifes()
    local value = lsne_memory.readsword("WRAM", 0x0DBE)
    -- value in memory is lifes-1
    -- for some reason the value is sometimes way to high (260), but still
    -- seems to decrease correctly by 1 when a life is lost
    return value + 1
end

-- TODO 0x13D9 0x13E4

function util.getMarioImage()
    local value = lsne_memory.readsword("WRAM", 0x13E0)
    return value
end

function util.isLevelEnding()
    local value = lsne_memory.readsword("WRAM", 0x1493)
    return (value > 0 and value <= 255)
end

--[[
function util.getTime()
    local hundreds = lsne_memory.readsword("WRAM", 0x0F31)
    local tens = lsne_memory.readsword("WRAM", 0x0F32)
    local ones = lsne_memory.readsword("WRAM", 0x0F33)
    --return hundreds*100 + tens*10 + ones*1
    return hundreds + tens + ones
end
--]]

function util.loadRandomTrainingSaveState()
    local stateNames = {}
    for fname in paths.iterfiles("states/train/") do
        if string.match(fname, "^.*\.lsmv$") then
            table.insert(stateNames, fname)
        end
    end
    --[[
    local stateNames = {
        "lvl1-left-1.lsmv", "lvl1-left-2.lsmv",
        "lvl1-right-1.lsmv", "lvl1-right-2.lsmv"
    }
    --]]
    if #stateNames == 0 then
        error("No training states found in 'states/train/' directory.")
    end
    local stateName = stateNames[math.random(#stateNames)]
    print("Reloading state ", stateName)
    local state = movie.to_rewind("states/train/" .. stateName)
    movie.unsafe_rewind(state)
end

-- convert rgb to grayscale by averaging channel intensities
-- https://gist.github.com/jkrish/29ca7302e98554dd0fcb
function util.rgb2y(im, threeChannels)
    -- Image.rgb2y uses a different weight mixture
    local dim, w, h = im:size()[1], im:size()[2], im:size()[3]
    if dim ~= 3 then
        print('<error> expected 3 channels')
        return im
    end

    -- a cool application of tensor:select
    local r = im:select(1, 1)
    local g = im:select(1, 2)
    local b = im:select(1, 3)

    local z = torch.Tensor(1, w, h):zero()

    -- z = z + 0.21r
    z = z:add(0.21, r)
    z = z:add(0.72, g)
    z = z:add(0.07, b)

    if threeChannels == true then
        z = torch.repeatTensor(z, 3, 1, 1)
    end

    return z
end

function util.toImageDimensions(img, dimensions)
    local c, h, w = img:size(1), img:size(2), img:size(3)
    if dimensions[1] == 1 and c ~= dimensions[1] then
        img = util.rgb2y(img)
    end
    if h ~= dimensions[2] or w ~= dimensions[3] then
        img = image.scale(img, dimensions[2], dimensions[3])
    end
    return img
end

function util.loadJPGCompressed(fp, channels, height, width)
    -- from https://github.com/torch/image/blob/master/doc/saveload.md
    --[[
    local fin = torch.DiskFile(fp, 'r')
    fin:binary()
    fin:seekEnd()
    local file_size_bytes = fin:position() - 1
    fin:seek(1)
    local img_binary = torch.ByteTensor(file_size_bytes)
    fin:readByte(img_binary:storage())
    fin:close()
    -- Then when you're ready to decompress the ByteTensor:
    im = image.decompressJPG(img_binary, 3)
    --]]
    local im = image.load(fp, 3, "float")
    local c, h, w = im:size(1), im:size(2), im:size(3)
    im = im[{{1,c}, {30,h}, {1,w}}] -- cut off 30px from the top
    if c ~= channels then
        im = util.rgb2y(im)
    end
    im = image.scale(im, height, width)
    local img_binary = util.compressJPG(im)

    return img_binary
end

function util.compressJPG(im)
    return image.compressJPG(im, 100)
end

function util.decompressJPG(img_binary)
    return image.decompressJPG(img_binary)
end

function util.saveStats()
    local fp = "learned/stats.th7"
    torch.save(fp, STATS)
end

function util.loadStats()
    local fp = "learned/stats.th7"
    if paths.filep(fp) then
        STATS = torch.load(fp)
    end
end

function util.sleep(seconds)
    os.execute("sleep " .. tonumber(seconds))
end

return util
