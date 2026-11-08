import os
import sys

# Fix protobuf version mismatch when importing stable_baselines3 / tensorboard
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import time
import numpy as np
from gymnasium import spaces
from playwright.sync_api import sync_playwright
from stable_baselines3 import PPO

# Compatibility shim for loading a Colab (Python 3.11+ / NumPy 2.x) model on Python 3.10 / NumPy 1.x
import numpy.core.numeric
import numpy.core.multiarray
sys.modules["numpy._core"] = numpy.core
sys.modules["numpy._core.numeric"] = numpy.core.numeric
sys.modules["numpy._core.multiarray"] = numpy.core.multiarray

custom_objects = {
    "learning_rate": 0.0,
    "lr_schedule": lambda _: 0.0,
    "clip_range": lambda _: 0.0,
    "observation_space": spaces.Box(low=-2.0, high=3.0, shape=(12,), dtype=np.float32),
    "action_space": spaces.Discrete(3),
}

# 1. Load your Colab-trained Superhuman Model!
model = PPO.load("dino_1M_superhuman.zip", device="cpu", custom_objects=custom_objects)
print("✅ Loaded dino_1M_superhuman.zip! Launching Chrome...")

with sync_playwright() as p:
    try:
        browser = p.chromium.launch(headless=False)
    except Exception:
        # Fallback to system Google Chrome if Playwright's Chromium hasn't been installed yet
        browser = p.chromium.launch(channel="chrome", headless=False)
    page = browser.new_page()

    # Open the authentic Chrome Dino runner.
    # Note: chrome://dino intentionally triggers net::ERR_INTERNET_DISCONNECTED to render the offline game!
    try:
        page.goto("chrome://dino")
    except Exception:
        pass

    time.sleep(1.5)

    # Check if the built-in Chrome Dino Runner loaded; only fallback to chromedino.com if missing
    has_runner = page.evaluate(
        "() => typeof Runner !== 'undefined' && Boolean(Runner.instance_ || (Runner.getInstance && Runner.getInstance()))"
    )
    if not has_runner:
        page.goto("https://chromedino.com/", wait_until="domcontentloaded")
        time.sleep(2.0)

    page.keyboard.press("Space")  # Start the game!
    print("🦖 AI is now playing live in Chrome! Press Ctrl+C in terminal to stop.")

    is_ducking = False
    is_space_down = False
    while True:
        # Extract live game state directly from Chrome's internal Runner JS object
        state = page.evaluate("""() => {
            if (typeof Runner === 'undefined') return null;
            const r = Runner.instance_ || (Runner.getInstance && Runner.getInstance());
            if (!r || !r.tRex || !r.horizon) return null;
            const tRex = r.tRex;
            const obs = r.horizon.obstacles.filter(o => o.xPos + o.width >= tRex.xPos + 10);
            const mapObs = (o) => o ? {
                x: o.xPos,
                y: o.yPos,
                w: o.width,
                type: (o.typeConfig && o.typeConfig.type) ? String(o.typeConfig.type) : ""
            } : null;
            return {
                crashed: Boolean(r.crashed),
                speed: r.currentSpeed,
                dinoX: tRex.xPos,
                dinoY: tRex.yPos,
                groundY: tRex.groundYPos,
                velY: tRex.jumpVelocity,
                jumping: Boolean(tRex.jumping),
                obs1: mapObs(obs[0]),
                obs2: mapObs(obs[1])
            };
        }""")

        if not state:
            time.sleep(0.005)
            continue

        if state["crashed"]:
            print("💥 Crashed! Restarting in 1 second...")
            if is_ducking:
                page.keyboard.up("ArrowDown")
                is_ducking = False
            if is_space_down:
                page.keyboard.up("Space")
                is_space_down = False
            time.sleep(1.0)
            page.keyboard.press("Space")
            continue

        # Map Chrome's real JS coordinates into SuperDinoEnv's 12-Feature State Vector
        dino_x = state["dinoX"]
        speed = max(6.0, float(state["speed"]))

        def parse_obs(o, default_x):
            if not o:
                return default_x, 24.0, "small_cactus"
            w = float(o["w"])
            raw_type = o["type"].upper()
            if "CACTUS" in raw_type and "SMALL" in raw_type:
                t = "small_cactus"
            elif "CACTUS" in raw_type and "LARGE" in raw_type:
                t = "large_cactus"
            else:
                # Chrome spawns Pterodactyls at 3 heights:
                # - yPos == 100 (>= 90): Ground-skimming bird -> fatal if ducked! Must JUMP over like a cactus.
                # - yPos == 75 (65..89): Mid-level bird -> must DUCK under.
                # - yPos == 50 (< 65): High bird -> safe to RUN under, fatal if jumped into!
                if o["y"] >= 90:
                    t = "small_cactus"
                elif o["y"] >= 65:
                    t = "bird_low"
                else:
                    t = "bird_high"
            return float(o["x"]), w, t

        x1, w1, t1 = parse_obs(state["obs1"], 700.0)
        x2, w2, t2 = parse_obs(state["obs2"], 1100.0)

        # Adaptive lead offset:
        # At low speed (6-8), model waits until dx=30 so we add ~36-42px lead.
        # At high speed (10-13), model already triggers jump earlier (dx=60), so we reduce lead_px
        # to prevent jumping too early and landing on the back edge of wide 3-cactus clusters!
        lead_px = max(18.0, 42.0 - (speed - 6.0) * 3.0)
        dx1 = (x1 - lead_px) - (dino_x + 36.0)
        dx2 = (x2 - lead_px) - (dino_x + 36.0)
        tti1 = (dx1 / speed) / 25.0
        tti2 = (dx2 / speed) / 25.0

        dino_elevation = float(state["groundY"] - state["dinoY"])

        obs_vec = np.array([
            dx1 / 400.0,
            tti1,
            w1 / 80.0,
            1.0 if "cactus" in t1 else 0.0,
            1.0 if t1 == "bird_low" else 0.0,
            1.0 if t1 == "bird_high" else 0.0,
            dx2 / 400.0,
            tti2,
            1.0 if t2 == "bird_low" else 0.0,
            dino_elevation / 100.0,
            float(state["velY"]) / 16.0,
            (speed - 7.0) / 8.0
        ], dtype=np.float32)

        # Query your trained PPO Neural Network!
        action, _ = model.predict(obs_vec, deterministic=True)
        action = int(action)

        # Release Space once the Dino reaches jump apex (velY >= -2.0) or lands,
        # so it doesn't float too long in the air at high speed
        if (not state["jumping"] or float(state["velY"]) >= -2.0) and is_space_down:
            page.keyboard.up("Space")
            is_space_down = False

        # Send real keyboard inputs to Chrome
        cleared_obs1 = (x1 + w1) < (dino_x + 28.0)
        if action == 1:
            if is_ducking:
                page.keyboard.up("ArrowDown")
                is_ducking = False
            if not state["jumping"]:
                page.keyboard.down("Space")
                is_space_down = True
        elif action == 2 and (
            (not state["jumping"] and t1 == "bird_low")
            or (state["jumping"] and cleared_obs1)
        ):
            if not is_ducking:
                page.keyboard.down("ArrowDown")
                is_ducking = True
        else:
            if is_ducking:
                page.keyboard.up("ArrowDown")
                is_ducking = False

        time.sleep(0.004)  # High-frequency polling to match Chrome's 60 FPS loop
