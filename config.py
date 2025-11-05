# config.py

# --- Paths ---
TEXTS_FILE = "texts.json"
SCENE_GRAPHS_FILE = "scene_graphs.json"
OUTPUT_PATH = "predicted_scene_graphs.json"
EVALUATION_PATH = "spice_report.json"

# --- API Configurations (Example) ---
OPENAI_API_KEY = ""
MODEL_NAME = "gpt-5-mini"

# --- Evaluation Settings ---
EVALUATION_METRIC = "SPICE"
EVALUATION_BATCH_SIZE = 32

# --- Logging ---
LOG_FILE = "./logs/project.log"
LOG_LEVEL = "INFO"

# --- Proxy Settings ---
USE_PROXY = False
PROXY_URL = "http://127.0.0.1:25378/echo-pac?t=746605765"
