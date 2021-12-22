import time
import numpy as np
from utils import stopwatch

def print_batch_loss(epoch_id, phase, epoch_loss_accumulator, time_epoch, batch_id, nb_batches):

  UP_AND_CLEAR = '\033[F\033[K' # up one line, clear until end of line
  if(batch_id == 0):
    UP_AND_CLEAR = '' # The first log of an epoch does not overwrite the console line above

  print('{}[Epoch: {},\t {}] Batch: {}/{} ({}%)\t Loss: {:.6f}\t (Stopwatch: {})'.format(
    UP_AND_CLEAR,
    epoch_id,
    phase,
    batch_id,
    nb_batches,
    np.rint(100 * batch_id / nb_batches).astype(np.int),
    epoch_loss_accumulator.get_and_reset_sub_loss(),
    stopwatch.stopwatch(time.time(), time_epoch)))
