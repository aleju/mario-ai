-- Various utility functions involving game speed, extracting information from
-- the emulator's memory, screen capturing, image processing, save state handling
-- and plotting.
local util = {}

-- Returns whether the game is paused.
-- TODO is this still used?
function util.isGamePaused()
    return gui.get_runmode() == "pause"
end

-- Sets the game speed to very fast (400).
function util.setGameSpeedToVeryFast()
    settings.set_speed(400)
end

-- Sets the game speed to fast (200).
-- TODO is this still used?
function util.setGameSpeedToFast()
    settings.set_speed(200)
end

-- Sets the game speed to normal (1).
-- TODO is this still used?
function util.setGameSpeedToNormal()
    settings.set_speed(1)
end

-- Returns a random entry from an array.
-- TODO is this still used?
function util.getRandomEntry(arr)
    return arr[math.random(#arr)]
end

-- Returns the current ingame score.
function util.getCurrentScore()
    local score = lsne_memory.readsword("WRAM", 0x0f34) * 10
    return score
end

-- Returns the current level?
-- TODO is this still used?
function util.getLevel()
    local level = lsne_memory.readsword("WRAM", 0x13bf)
    return level
end

-- Returns Mario's current x-coordinate.
function util.getPlayerX()
    local x = lsne_memory.readsword("WRAM", 0x0094)
    return x
end

-- Returns the current game status.
-- 0 = level
-- 1 = black screen?
-- 2 = overworld
function util.getMarioGameStatus()
    local status = lsne_memory.readsword("WRAM", 0x0D9B)
    return status
end

-- Returns whether the level is beaten.
-- TODO is this still used?
-- TODO does this work?
function util.isLevelBeaten()
    return util.getLevelBeatenStatus() == 1
end

-- Returns whether the game is over.
-- TODO is this still used?
function util.isGameOver()
    return util.getLevelBeatenStatus() == 128
end

-- Returns the level beaten status.
-- 0 = not beaten
-- 1 = beaten ?
-- 128 = game over
function util.getLevelBeatenStatus()
    local value = lsne_memory.readsword("WRAM", 0x0DD5)
    return value
end

-- Returns Mario's count of lifes.
-- Seems to not be fully reliable.
function util.getCountLifes()
    local value = lsne_memory.readsword("WRAM", 0x0DBE)
    -- value in memory is lifes-1
    -- for some reason the value is sometimes way to high (260), but still
    -- seems to decrease correctly by 1 when a life is lost
    return value + 1
end

-- Returns Mario's current sprite.
-- 62 = Mario death animation sprite.
function util.getMarioImage()
    local value = lsne_memory.readsword("WRAM", 0x13E0)
    return value
end

-- Returns whether the level is currently ending (flat pole animation).
function util.isLevelEnding()
    local value = lsne_memory.readsword("WRAM", 0x1493)
    return (value > 0 and value <= 255)
end

-- Picks a random saved state and loads it (testing states only).
function util.loadRandomTrainingSaveState()
    local stateNames = {}
    for fname in paths.iterfiles("states/train/") do
        if string.match(fname, "^.*\.lsmv$") then
            table.insert(stateNames, fname)
        end
    end

    if #stateNames == 0 then
        error("No training states found in 'states/train/' directory.")
    end
    local stateName = stateNames[math.random(#stateNames)]
    print("Reloading state ", stateName)
    local state = movie.to_rewind("states/train/" .. stateName)
    movie.unsafe_rewind(state)
end

-- Picks a random saved state and loads it (testing states only).
function util.loadRandomTestSaveState()
    local stateNames = {}
    for fname in paths.iterfiles("states/test/") do
        if string.match(fname, "^.*\.lsmv$") then
            table.insert(stateNames, fname)
        end
    end

    if #stateNames == 0 then
        error("No test states found in 'states/test/' directory.")
    end
    local stateName = stateNames[math.random(#stateNames)]
    print("Reloading state ", stateName)
    local state = movie.to_rewind("states/test/" .. stateName)
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

-- Resize an image to given dimensions, including RGB to grayscale conversion.
-- TODO currently does not handle grayscale2rgb.
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

-- Take a screenshot of the game and return it as a tensor.
-- TODO no longer used?
function util.getScreen()
    local fp = SCREENSHOT_FILEPATH
    gui.screenshot(fp)
    local screen = image.load(fp, 3, "float"):clone()
    screen = image.scale(screen, IMG_DIMENSIONS[2], IMG_DIMENSIONS[3]):clone()
    if IMG_DIMENSIONS[1] == 1 then
        screen = util.rgb2y(screen)
    end
    return screen
end

-- Take a screenshot of the game and return it jpg-compressed as a tensor.
function util.getScreenCompressed()
    local fp = SCREENSHOT_FILEPATH
    gui.screenshot(fp)
    return util.loadJPGCompressed(fp, IMG_DIMENSIONS[1], IMG_DIMENSIONS[2], IMG_DIMENSIONS[3])
end

-- Load a JPG image from a file, but keep it compressed.
function util.loadJPGCompressed(fp, channels, height, width)
    -- from https://github.com/torch/image/blob/master/doc/saveload.md
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

-- Compress an uncompressed image tensor to a jpg-compressed image tensor.
function util.compressJPG(im)
    return image.compressJPG(im, 100)
end

-- Decompress a jpg-compressed image tensor.
function util.decompressJPG(img_binary)
    return image.decompressJPG(img_binary)
end

-- Save the global STATS-table to a file.
function util.saveStats()
    local fp = "learned/stats.th7"
    torch.save(fp, STATS)
end

-- Load the global STATS-table from a file.
function util.loadStats()
    local fp = "learned/stats.th7"
    if paths.filep(fp) then
        STATS = torch.load(fp)
    end
end

-- Sleep for N seconds.
function util.sleep(seconds)
    os.execute("sleep " .. tonumber(seconds))
end

-- plot average recieved rewards (per N actions)
function util.plotAverageReward(rewardData, clampTo)
    clampTo = clampTo or 10
    local points = {}
    for i=1,#rewardData do
        local point = rewardData[i]
        local direct = math.max(math.min(point[2], clampTo), (-1) * clampTo)
        local observedGamma = math.max(math.min(point[3], clampTo), (-1) * clampTo)
        local expectedGamma = math.max(math.min(point[4], clampTo), (-1) * clampTo)
        table.insert(points, {point[1], direct, observedGamma, expectedGamma})
    end
    display.plot(points, {win=3, labels={'action counter', 'direct', 'observed gamma', 'expected gamma'}, title='Average rewards per N actions'})
end

return util
