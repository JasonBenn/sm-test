import argparse
import os
from types import SimpleNamespace

import torch

# loads data from input channels
# configures training with hyperparameters (passed as args)

if __name__ =='__main__':
    parser = argparse.ArgumentParser()

    # hyperparameters sent by the client are passed as command-line arguments to the script.
    parser.add_argument('--attribute_name', type=str)

    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--learning-rate', type=float, default=0.05)
    # parser.add_argument('--use-cuda', type=bool, default=False)
    # print(os.environ.get("SM_NUM_GPUS"))

    # Data, model, and output directories
    parser.add_argument('--output-data-dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--train', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    parser.add_argument('--test', type=str, default=os.environ['SM_CHANNEL_TEST'])

    args, _ = parser.parse_known_args()

    # ... load from args.train and args.test, train a model, write model to args.model_dir.
    model = SimpleNamespace(state_dict={"testing": 'TESTING TESTING TESTING'})

    # ... train `model`, then save it to `model_dir`
    model_name = f'{args.attribute_name}_bert.pth'  # TODO: Follow naming convention from Sourceress
    with open(os.path.join(args.model_dir, model_name), 'wb') as f:
        torch.save(model.state_dict(), f)
