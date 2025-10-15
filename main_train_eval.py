

"""Training and evaluation"""
import jax

import run_lib
from absl import app
from absl import flags
from ml_collections.config_flags import config_flags
FLAGS = flags.FLAGS
import warnings
warnings.filterwarnings("ignore")

config_flags.DEFINE_config_file(
  "config", None, "Training configuration.", lock_config=True)
flags.DEFINE_enum("mode", None, ["train", "inference"], "Running mode: train or eval")

flags.mark_flags_as_required(["config", "mode"])



def main(argv):
  if FLAGS.mode == "train":

    run_lib.train_flow(FLAGS.config)
  elif FLAGS.mode == "inference":

    run_lib.inference_mcmc(FLAGS.config)

  else:
    raise ValueError(f"Mode {FLAGS.mode} not recognized.")


if __name__ == "__main__":
  
  app.run(main)
