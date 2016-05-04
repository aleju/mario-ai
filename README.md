# About

This project contains code to train a model that automatically learns to play Super Mario World.
The underlying technique is deep Q-learning, as described in the [Atari paper](http://arxiv.org/abs/1312.5602).

# Methodology

## Basics, replay memory

The training method is deep Q-learning with a replay memory, i.e. the model observes sequences of screens,
saves them in its "memory" and later on has to learn to accurately predict the expected action reward values
("action" means "press button X") based on the collected memories.
The replay memory has a size of 110k training entries (small, because of the bad memory handling of the emulator).
When it's full, new entries replace randomly chosen older ones.
Before each training epoch, all indirect rewards are recalculated so that the agent is not getting trained on its outdated estimates from many epochs ago.
Examples for training batches are chosen randomly (uniform distribution).

## Inputs, outputs, actions

Each example's input has the following structure:
* The last T actions, each as two-hot-vectors. (Two, because the model can choose two buttons.)
* The last T screens, each downscaled to size 32x32 (grayscale, slightly cropped).
* The last screen at size 64x64 (grayscale, slightly cropped).

T is currently set to 4 (note that this includes the last screen of the sequence). Screens are captured at every 5th frame.
Each example's output are the action reward values.
The model can choose two actions per state: One arrow button (up, down, right, left) and one of the other control buttons (A, B, X, Y).
This is different from the Atari-model, in which the agent could only pick one button at a time.
(Without this change, the agent could theoretically not make many jumps, which force you to keep the A button pressed and move to the right.)
As the reward function is constructed in such a way that it is almost never 0, exactly two of each example's the output values are expected to be non-zero.

## Reward function

The agent gets the following rewards:
* X-Difference reward: `+0.5` if the agent moved to the right, `+1.0` if it moved *fast* to the right (8 pixels or more compared to the last game state), `-1.0` if it moved to the left and `-1.5` if it moved *fast* to the left (-8 pixels or more).
* Level finished: `+2.0` while the level-finished-animation is playing.
* Death: `-3.0` while the death animation is playing.

The `gamma` (discount for expected/indirect rewards) is set to `0.9`.

## Error function

A selective MSE is used to train the agent. That is, for each example gradients are calculated just like they would be for a MSE.
However, the gradients of all action values are set to 0 if their target reward was 0.
That's because each example contains only the received reward for one pair of actions (arrows, other buttons) that the action has chosen in the past.
Other pairs of actions would have been possible, but the agent didn't choose them and so the reward for them is unclear.
Their reward values (per example) are set to 0, but not because they were truely 0, but instead because we don't know what reward the agent would have received if it had chosen them.
Backpropagating gradient for them (i.e. if the agent predicts a value unequal to 0) is therefore not very reasonable.

This implementation can afford to detect "unchosen" actions based on their reward being 0, because the received reward is here almost never 0.
Other implementations would need to take more care of this step.

## Policy

The policy is an epsilon-greedy one, which starts at epsilon=0.8 and anneals that down to 0.1 at the 400k-th chosen action.
Whenever according to the policy a random action should be chosen, the agent throws a coin (i.e. 50:50 chance) and either randomizes one of its two (arrows, other buttons) actions or it randomizes both of them.

## Model architecture


# Limitations

The agent is trained mainly on the first level (first to the right in the overworld at the start).
Other levels suffer significantly more from various difficulties with which the agent can hardly deal, which mainly are:
* Jumping puzzles. The agent will just jump to right and straight into its death.
* Huge cannons balls. To get past them you have to jump on them or duck under them (big mario) or walk under them (small mario). Jumping on top of them is even rather hard for a human novice player. Ducking or walking under them is very hard for the agent due to the epsilon-greedy policy, which will randomly make mario jump and then instantly die.
* High walls/tubes. The agent has to *keep* A pressed to get over them. Again, hard to learn and runs contrary to epsilon-greedy.
* Horizontal tubes. These are sometimes located at the end of areas and you are supposed to walk into them to get to the next area. The agent instead will usually jump on them (because it loves to jump) and then keep walking to the right, hitting the next wall.

These difficulties also make it hard to train the agent on the first level and then test it on another one.
There simply isn't any other level that is comparable to the first one.

# Usage

## Basic requirements

* Ubuntu.
* Lots of time. This is not an easy install.
* 16GB+ of RAM.
* Around 2GB of disk space for the replay memory checkpoints.
* An NVIDIA GPU with 4+ GB of memory.
* CUDA. Version 7 or newer should do.
* CUDNN. Version 4 or newer should do.


## Install procedure

* Make sure that you have lua 5.1 installed. I had problems with 5.2 in torch.
* Make sure that you have gcc 4.9 or higher installed. The emulator will compile happily with gcc <4.9 but then sometimes throw errors when you actually use it.
* Install torch.
  * Make sure that the following packages are installed: nn, paths, image, dpnn, display. dpnn and display are usually not part of torch.
* Compile the emulator:
  * Download the source code of lsnes rr2 beta23. (Not version rr1! Other emulators will likely not work with the code.)
  * Extract the emulator source code and open the created directory.
  * Open `source/src/libray/lua.cpp` and insert the following code under `namespace {`:
```#ifndef LUA_OK
#define LUA_OK 0
#endif

#ifdef LUA_ERRGCMM
	REGISTER_LONG_CONSTANT("LUA_ERRGCMM", LUA_ERRGCMM, CONST_PERSISTENT | CONST_CS);
#endif
```
This makes the emulator run in lua 5.1. Newer versions of lsnes rr2 beta23 might not need this.
  * Open `source/include/core/controller.hpp` and change the function `do_button_action` from private to public. Simply cut the line `void do_button_action(const std::string& name, short newstate, int mode);` in the `private:` block and paste it into the `public:` block.
  * Open `source/src/lua/input.cpp` and before `lua::functions LUA_input_fns(...` (at the end of the file) insert:
```
	int do_button_action(lua::state& L, lua::parameters& P)
	{
		auto& core = CORE();
                
                std::string name;
                short newstate;
                int mode;

		P(name, newstate, mode);
                core.buttons->do_button_action(name, newstate, mode);
                return 1;
	}
```
  This method was necessary to actually press buttons from lua. All the default lua functions for that would just never work, because `core.lua2->input_controllerdata` apparently never gets set (which btw will let these functions silently fail, i.e. without any error).
  * Again in `source/src/lua/input.cpp`, at the block `lua::functions LUA_input_fns(...`, add `do_button_action` to the lua commands that can be called from lua scripts loaded in the emulator. To do that, change the line `{"controller_info", controller_info},` to `{"controller_info", controller_info}, {"do_button_action", do_button_action},`.
  * Switch back to `source/`.
  * Compile the emulator with `make`.
    * You will likely encounter many problems during this step that will require lots of googling to solve.
    * If you encounter problems with portaudio, deactivate it in the file `options.build`.
    * If you encounter problems with something like libwxgtk, then install package `libwxgtk3.0-dev` and not version 2.8-dev, as their official page might tell you.
  * From `source/` execute `sudo cp lsnes /usr/bin/ && sudo chown root:root /usr/bin/lsnes`.
* Now create a ramdisk. That will be used to save screenshots from the game (in order to get the pixel values). Do the following:
  * `sudo mkdir /media/ramdisk`
  * `sudo chmod 777 /media/ramdisk`
  * `sudo mount -t tmpfs -o size=128M none /media/ramdisk && mkdir /media/ramdisk/mario-ai-screenshots`
  * Note: You can choose a different path. Then you will have to change `SCREENSHOT_FILEPATH` in train.lua.
  * Note: You don't *have* to use a ramdisk, but your hard drive will probably not like the constant wear from tons of screenshots being saved.


## Training

* Clone this repository via `git clone https://github.com/aleju/mario-ai.git`.
* `cd` into the created directory.
* Download a Super Mario World (USA) ROM.
* Start lsnes (from the repository directory) by using `lsnes` in a terminal window.
* In the emulator, go to `Configure -> Settings -> Advanced` and set the lua memory to 1024MB. (Only has to be done the first time.)
* Configure your controller buttons. Play until the overworld pops up. There, move to the right and start that level. Play that level a bit and save a handful or so of states via the emulator's `File -> Save -> State` to the subdirectory `states/train`. Name doesn't matter, but they have to end in `.lsmv`.
* Start the display server by opening a command window and using `th -ldisplay.start`. If that doesn't work you haven't installed display yet, use `luarocks install display`.
* Open the display server by opening `http://localhost:8000/` in your browser.
* Now start the training via `Tools -> Run Lua script...` and select `train.lua`.
  * Don't be surprised if the batch loss keeps increasing for quite some time. It should however not reach absurd numbers, like say 100k+.
* You can stop the training via `Tools -> Reset Lua VM`.
* If you want to restart the training from scratch (e.g. for a second run), you will have to delete the files in `learned/`. Note that you can sometimes keep the replay memory and train a new network with it.
