class EnvParams:
    SPECIES_AGENTS_RANGE = (3, 3)
    SPECIES_RANGE = (3, 5)
    TASKS_RANGE = (15, 50)
    MAX_TIME = 200
    TRAIT_DIM = 5
    DECISION_DIM = 30


class TrainParams:
    USE_GPU = False
    USE_GPU_GLOBAL = True
    NUM_GPU = 1
    NUM_META_AGENT = 4
    LR = 1e-5
    GAMMA = 1
    DECAY_STEP = 2e3
    RESET_OPT = False
    EVALUATE = True
    EVALUATION_SAMPLES = 256
    RESET_RAY = False
    INCREASE_DIFFICULTY = 20000
    SUMMARY_WINDOW = 8
    DEMON_RATE = 0.5
    IL_DECAY = -1e-5  # -1e-6 700k decay 0.5, -1e-5 70k decay 0.5, -1e-4 7k decay 0.5
    BATCH_SIZE = 2048
    AGENT_INPUT_DIM = 6 + EnvParams.TRAIT_DIM
    TASK_INPUT_DIM = 5 + 2 * EnvParams.TRAIT_DIM
    EMBEDDING_DIM = 128
    SAMPLE_SIZE = 200
    PADDING_SIZE = 50
    POMO_SIZE = 10
    FORCE_MAX_OPEN_TASK = False


class SaverParams:
    FOLDER_NAME = 'save_1'
    MODEL_PATH = f'model/{FOLDER_NAME}'
    TRAIN_PATH = f'train/{FOLDER_NAME}'
    GIFS_PATH = f'gifs/{FOLDER_NAME}'
    LOAD_MODEL = False
    LOAD_FROM = 'current'  # 'best'
    SAVE = True
    SAVE_IMG = True
    SAVE_IMG_GAP = 1000


class EvidenceParams:
    """Configuration for optional evidence-rich training reports."""

    ENABLE_EVIDENCE_LOGGING = True
    EVIDENCE_LOG_INTERVAL_STEPS = 10000
    EVIDENCE_OUTPUT_DIR = './evidence_logs'
    MAX_CASES_PER_REPORT = 5
    MAX_CANDIDATES_PER_DECISION = 5
    TOP_K_CANDIDATES = 5
    LOW_CAPABILITY_THRESHOLD = 0.3
    BETTER_ALTERNATIVE_GAP = 0.3
    DEADLOCK_LOOKBACK_DECISIONS = 10


class DeepSeekBiasParams:
    """Window-level DeepSeek bias configuration; API keys are never stored here."""

    ENABLE_DEEPSEEK_BIAS = True
    DEEPSEEK_BIAS_UPDATE_INTERVAL_STEPS = 30000
    DEEPSEEK_BASE_URL = 'https://api.deepseek.com'
    DEEPSEEK_MODEL = 'deepseek-v4-pro'
    DEEPSEEK_TEMPERATURE = 0.0
    DEEPSEEK_MAX_TOKENS = 4096
    DEEPSEEK_TIMEOUT = 120
    DEEPSEEK_USE_JSON_RESPONSE = True
    DEEPSEEK_THINKING_TYPE = 'disabled'
    DEEPSEEK_RESPONSE_OUTPUT_DIR = './evidence_logs/deepseek_responses'
    BIAS_CONFIG_OUTPUT_DIR = './evidence_logs/bias_configs'
    LLM_BIAS_EMA_ALPHA = 0.3
    WEIGHT_RANGE = (-2.0, 2.0)
    LAMBDA_RANGE = (0.0, 1.0)
    CLIP_BOUND_RANGE = (-10.0, 10.0)

