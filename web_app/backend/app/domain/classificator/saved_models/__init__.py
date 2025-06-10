import os

MODELS_DIR = os.path.dirname(os.path.abspath(__file__))

class SavedModelsPaths:
    class CamNetRegressor:
        REG_G_0661 =  os.path.join(MODELS_DIR, "reg_g_0661.pth")
    class FragmentClassifier:
        MODEL99 = os.path.join(MODELS_DIR, "model_99.pth")
    class CamNet:
        class Both:
            CAMNET_0845 = os.path.join(MODELS_DIR, "camnet_0845.pth")