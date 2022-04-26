# TrackMania-agent
Training a TrackMania agent using the `tmrl` library.

![Test run](test_run.gif)

# Introduction
* It is absolutely necessary to understand the structure of `tmrl` before writing your own training agents/workers.

* [This](https://github.com/trackmania-rl/tmrl/blob/master/readme/get_started.md) goes into great detail on how to get started with `tmrl`.

* Summary:
    * `Worker` : Plays the game, collects episodes using the model defined for the actor, sends episodes to the `Server`.
    * `Trainer` : Receives episodes from the `Server`, trains on sampled episodes, sends weights back to the `Server` after every run.
    * `Server` : Collects episodes from the `Worker` and sends it to the `Trainer`. Also receives weights from the `Trainer` after each run and broadcasts them to all `Workers` so that they can update their models.

A video of the trained agent driving on a test track can be found [here](https://wpi0-my.sharepoint.com/personal/kchin_wpi_edu/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fkchin%5Fwpi%5Fedu%2FDocuments%2FDS595%2DRL%20Project%204%2Ffinal%5Fdemo%2Eavi&parent=%2Fpersonal%2Fkchin%5Fwpi%5Fedu%2FDocuments%2FDS595%2DRL%20Project%204&ga=1).

# Installation

* Please go through the [`tmrl` installation guide](https://github.com/trackmania-rl/tmrl/blob/master/readme/Install.md) which explains things clearly.

* Summary:
    * `Workers` can only run on Windows because `TrackMania` is available only for Windows.
    * `Trainers` and `Servers` can run on any system provided they have `tmrl` installed.
    * All 3 modules require `tmrl` to be installed to function.
    * Extra steps are required for the Windows machine that will run TrackMania with the `Worker`, information in the `tmrl` installation guide.
    * You can have multiple `Workers` (running on different Windows laptops), but only `1` `Server` and `1` `Trainer`.

* Install using:
    ```sh
    pip install tmrl 
    ```

* This also creates a folder named `TmrlData` in your home directory. (`C:\Users\your username\` on Windows.)

* Look at [this](https://github.com/trackmania-rl/tmrl/blob/master/readme/Install.md#optional-configuremanage-tmrl) to understand what the different folders in `TmrlData` are for.

# Usage
* Before running anything, check `config.json` in `TmrlData/config` to see if you'd like to change anything.

* **DO NOT** change the name from **SAC** even if you're training a different agent. There are constants in the file installed by `pip` (in `site-packages`) which are hardcoded, so training won't work if the name's changed. 

* We *trick* `tmrl` into thinking it's running SAC, but we're actually running our own algorithm.

* You need TrackMania running on your train track before you can start your workers. See [getting started](https://github.com/trackmania-rl/tmrl/blob/master/readme/get_started.md).

* You can copy your train tracks to `\Documents\Trackmania\Maps\My Maps` so you can open them in game in the map editor.


* Run `Server` (any system) :
    ```sh
    python -m tmrl --server
    ```

* Run `Worker` (Windows system running TrackMania):
    ```sh
    cd src
    python3 .\worker.py --worker
    ```

* You can run `Workers` on different Windows laptops

* This will run **our** `Worker`, not the one that's installed with `tmrl`.

* Run `Trainer` (Any system with a good GPU, works on Clusters as well!):
    ```sh
    cd src
    python3 train_agent.py
    ```

* After training, you can use the updated weights to test the model.
    ```sh
    cd src
    python3 .\worker.py --test
    ```

* If you want to run our trained model, copy `DDPG_trained.pth` into the `TmrlData/weights` folder on your Windows laptop and **RENAME** it to **SAC.pth**. (_Important_)

* Run `worker.py` with the `--test` flag and it will load our trained model.

# Custom Agents
* To write custom agents, you need to implement `3` classes :
    * `training_agent_cls` : See `train_agent.py`  (for `Trainer`)
    * `model_cls` : See `ddpg_agent.py`  (for `Trainer`)
    * `actor_module_cls` : See `worker.py`  (for `Trainer` as well as `Workers`)
