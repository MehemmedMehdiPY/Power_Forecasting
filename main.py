from constants import (
    BATCH_SIZE
    )
from utils.preprocessing import DataHandler
# from models import Forecaster


DataHandler("../data", seq_length=24, batch_size=BATCH_SIZE)
