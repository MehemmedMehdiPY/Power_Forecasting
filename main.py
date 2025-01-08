from constants import (
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

loader = DataHandler("../data", seq_length=24, batch_size=BATCH_SIZE)

model = Forecaster(input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE)

optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
loss_fn = MSELoss()

trainer = Trainer(model=model, train_loader=loader, val_loader=loader, 
                  optimizer=optimizer, loss_fn=loss_fn, epochs=EPOCHS, 
                  filepath=FILEPATH, device=DEVICE)

trainer.run()