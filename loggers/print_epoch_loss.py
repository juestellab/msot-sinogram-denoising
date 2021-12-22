import time

from utils import stopwatch

def print_epoch_loss(phase, epoch_id, loss, time_epoch):

  UP_CLEAR_AND_RETURN = '\033[F\033[K' # up one line, clear until end of line, return

  # --- Only for Tst
  if(phase == 'Tst'):
    RETURN = '\n'
  else:
    RETURN = ''

  # --- Common to Trn/Val/Tst
  print('{}[Epoch: {}, \t {}] Loss: {:.6f}\t (Stopwatch: {}){}'.format(
  UP_CLEAR_AND_RETURN,
  epoch_id,
  phase,
  loss,
  stopwatch.stopwatch(time.time(), time_epoch),
  RETURN))
