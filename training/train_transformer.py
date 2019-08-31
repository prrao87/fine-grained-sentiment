#!/usr/bin/env python3
"""
All code in this file is as per the NAACL transfer learning tutorial:
https://github.com/huggingface/naacl_transfer_learning_tutorial
"""
import argparse
import multiprocessing
import os
import torch
from pytorch_transformers import BertTokenizer, cached_path
from pytorch_transformers.optimization import AdamW
from ignite.engine import Engine, Events
from ignite.metrics import RunningAverage, Accuracy
from ignite.handlers import ModelCheckpoint
from ignite.contrib.handlers import PiecewiseLinear, ProgressBar

from transformer_utils.loader import TextProcessor, read_sst5
from transformer_utils.model import TransformerWithClfHeadAndAdapters

PRETRAINED_MODEL_URL = "https://s3.amazonaws.com/models.huggingface.co/naacl-2019-tutorial/"
TEXT_COL, LABEL_COL = 'text', 'truth'  # Column names in pd.DataFrame for sst dataset
n_cpu = multiprocessing.cpu_count()


def load_pretrained_model(args):
    "download pre-trained model and config"
    state_dict = torch.load(cached_path(os.path.join(args.model_checkpoint, "model_checkpoint.pth")),
                            map_location='cpu')
    config = torch.load(cached_path(os.path.join(args.model_checkpoint, "model_training_args.bin")))
    # Initialize model: Transformer base + classifier head
    model = TransformerWithClfHeadAndAdapters(config=config, fine_tuning_config=args).to(args.device)
    incompatible_keys = model.load_state_dict(state_dict, strict=False)
    print(f"Parameters discarded from the pretrained model: {incompatible_keys.unexpected_keys}")
    print(f"Parameters added in the model: {incompatible_keys.missing_keys}")

    if args.adapters_dim > 0:
        # Display adaptation parameters
        for name, param in model.named_parameters():
            if 'embeddings' not in name and 'classification' not in name and 'adapters_1' not in name and 'adapters_2' not in name:
                param.detach_()
                param.requires_grad = False
            else:
                param.requires_grad = True
        full_parameters = sum(p.numel() for p in model.parameters())
        trained_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

        print(f"\nWe will train {trained_parameters:,} parameters out of {full_parameters:,}"
              f" (i.e. {100 * trained_parameters/full_parameters:.1f}%) of the full parameters")

    return model, state_dict, config


def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_checkpoint", type=str, default=PRETRAINED_MODEL_URL, help="Path to the pretrained model checkpoint")
    parser.add_argument("--dataset_path", type=str, default='../data/sst', help="Directory to dataset.")
    parser.add_argument("--dataset_cache", type=str, default='./dataset_cache', help="Path to dataset cache")
    parser.add_argument("--logdir", type=str, default='./transformer_results', help="Path to logs")
    parser.add_argument("--num_classes", type=int, default=5, help="Number of classes for the target classification task")
    parser.add_argument("--adapters_dim", type=int, default=-1, help="If >0 add adapters to the model with adapters_dim dimension")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout for transformer module")
    parser.add_argument("--clf_loss_coef", type=float, default=1, help="If >0 add a classification loss")
    parser.add_argument("--train_batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--valid_batch_size", type=int, default=32, help="Batch size for validation")
    parser.add_argument("--valid_pct", type=float, default=0.1, help="Percentage of training data to use for validation")
    parser.add_argument("--lr", type=float, default=6.5e-5, help="Learning rate")
    parser.add_argument("--n_warmup", type=int, default=10, help="Number of warmup iterations")
    parser.add_argument("--max_norm", type=float, default=1.0, help="Clipping gradient norm")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay")
    parser.add_argument("--n_epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--gradient_acc_steps", type=int, default=2, help="Number of update steps to accumulate before a backward pass.")
    parser.add_argument("--init_range", type=float, default=0.02, help="Normal initialization standard deviation")

    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda or cpu)")
    args = parser.parse_args()

    # Define pretrained model and optimizer
    model, state_dict, config = load_pretrained_model(args)
    optimizer = AdamW(model.parameters(), lr=args.lr, correct_bias=False)
    num_model_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model has {num_model_params:,} parameters")
    # Define datasets
    datasets = read_sst5(args.dataset_path)

    # Define labels
    labels = list(set(datasets["train"][LABEL_COL].tolist()))
    assert len(labels) == args.num_classes  # Specified number of classes should be equal to that in the given dataset!
    label2int = {label: i for i, label in enumerate(labels)}
    int2label = {i: label for label, i in label2int.items()}

    # Get BertTokenizer for this pretrained model
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=False)
    clf_token = tokenizer.vocab['[CLS]']  # classifier token
    pad_token = tokenizer.vocab['[PAD]']  # pad token
    processor = TextProcessor(tokenizer, label2int, clf_token, pad_token, max_length=config.num_max_positions)

    train_dl = processor.create_dataloader(datasets["train"],
                                           shuffle=True,
                                           batch_size=args.train_batch_size,
                                           valid_pct=None)

    valid_dl = processor.create_dataloader(datasets["dev"],
                                           batch_size=args.train_batch_size,
                                           valid_pct=None)

    test_dl = processor.create_dataloader(datasets["test"],
                                          batch_size=args.valid_batch_size,
                                          valid_pct=None)

    # Training function and trainer
    def update(engine, batch):
        "update function for training"
        model.train()
        inputs, labels = (t.to(args.device) for t in batch)
        inputs = inputs.transpose(0, 1).contiguous()  # to shape [seq length, batch]
        _, loss = model(inputs,
                        clf_tokens_mask=(inputs == clf_token),
                        clf_labels=labels)
        loss = loss / args.gradient_acc_steps
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_norm)
        if engine.state.iteration % args.gradient_acc_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
        return loss.item()
    trainer = Engine(update)

    # Evaluation function and evaluator (evaluator output is the input of the metrics)
    def inference(engine, batch):
        model.eval()
        with torch.no_grad():
            batch, labels = (t.to(args.device) for t in batch)
            inputs = batch.transpose(0, 1).contiguous()  # to shape [seq length, batch]
            clf_logits = model(inputs,
                               clf_tokens_mask=(inputs == clf_token),
                               padding_mask=(batch == pad_token))
        return clf_logits, labels
    evaluator = Engine(inference)

    # add metric to evaluator
    Accuracy().attach(evaluator, "accuracy")

    # add evaluator to trainer: eval on valid set after each epoch
    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(engine):
        evaluator.run(valid_dl)
        print(f"validation epoch: {engine.state.epoch} acc: {100*evaluator.state.metrics['accuracy']:.3f}%")

    # Learning rate schedule: linearly warm-up to lr and then to zero
    scheduler = PiecewiseLinear(optimizer, 'lr', [(0, 0.0), (args.n_warmup, args.lr),
                                (len(train_dl) * args.n_epochs, 0.0)])
    trainer.add_event_handler(Events.ITERATION_STARTED, scheduler)

    # Add progressbar with loss
    RunningAverage(output_transform=lambda x: x).attach(trainer, "loss")
    ProgressBar(persist=True).attach(trainer, metric_names=['loss'])

    # Save checkpoints and finetuning config
    checkpoint_handler = ModelCheckpoint(args.logdir, 'checkpoint',
                                         save_interval=1, require_empty=False)
    trainer.add_event_handler(Events.EPOCH_COMPLETED, checkpoint_handler, {'sst_model': model})

    # Save metadata
    torch.save({
        "config": config,
        "config_ft": args,
        "int2label": int2label
    }, os.path.join(args.logdir, "model_training_args.bin"))

    # Run trainer
    trainer.run(train_dl, max_epochs=args.n_epochs)
    # Evaluate
    evaluator.run(test_dl)
    print(f"test results - acc: {100*evaluator.state.metrics['accuracy']:.3f}")
    # Save fine-tuned model weights
    torch.save(model.state_dict(), os.path.join(args.logdir, "model_weights.pth"))


if __name__ == "__main__":
    train()
