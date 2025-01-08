from constants import (
    BATCH_SIZE
    )
from utils.preprocessing import DataHandler
from models import Forecaster

loader = DataHandler("../data", seq_length=24, batch_size=BATCH_SIZE)

for _ in range(10):
    x, y = loader()
    print(x.shape, y.shape)
    break
