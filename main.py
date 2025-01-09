from constants import (
    SEQ_LENGTH,
    BATCH_SIZE,
    INPUT_SIZE,
    HIDDEN_SIZE,
    LEARNING_RATE,
    EPOCHS,
    FILEPATH,
    DEVICE
    )
from utils.preprocessing import DataHandler
from utils.train import Trainer
from models import Forecaster
from torch.optim import Adam
from torch.nn import MSELoss

train_handler = DataHandler("../data", mode="train", seq_length=SEQ_LENGTH, batch_size=BATCH_SIZE)
val_handler = DataHandler("../data", mode="val", seq_length=SEQ_LENGTH, batch_size=BATCH_SIZE)

model = Forecaster(input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE).to(DEVICE)

optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
loss_fn = MSELoss()

trainer = Trainer(model=model, train_handler=train_handler, val_handler=val_handler,
                  optimizer=optimizer, loss_fn=loss_fn, epochs=EPOCHS, 
                  filepath=FILEPATH, device=DEVICE)

trainer.run()