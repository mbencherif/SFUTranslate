"""
The universal main script of the translation training process in SFUTranslate. If you have implemented new models
    please add them to the if-block in main() function so they can be loaded using configuration files.
"""
import os
import math
import torch
from torch import nn
from tqdm import tqdm
from configuration import cfg, device
from readers.data_provider import DataProvider
from utils.optimizers import get_a_new_optimizer
from models.copy.model import CopyModel
from models.sts.model import STS
from models.transformer.model import Transformer
from models.aspects.model import AspectAugmentedTransformer, MultiHeadAspectAugmentedTransformer, SyntaxInfusedTransformer, BertFreezeTransformer
from models.transformer.optim import TransformerScheduler
from utils.init_nn import weight_init
from utils.evaluation import evaluate
from timeit import default_timer as timer
# To avoid the annoying UserWarnings of torchtext
# Remove this once the next version of torchtext is available
import warnings
warnings.filterwarnings("ignore", category=UserWarning)


def print_running_time(t):
    day = t // (24 * 3600)
    t = t % (24 * 3600)
    hour = t // 3600
    t %= 3600
    minutes = t // 60
    t %= 60
    seconds = t
    print("Total training execution time: {:.2f} days, {:.2f} hrs, {:.2f} mins, {:.2f} secs".format(day, hour, minutes, seconds))


def create_sts_model(SRC, TGT):
    model = STS(SRC, TGT).to(device)
    model.apply(weight_init)
    optimizer, scheduler = get_a_new_optimizer(cfg.optim, cfg.init_learning_rate, model.parameters())
    return model, optimizer, scheduler, bool(cfg.grad_clip), True


def create_copy_model(SRC, TGT):
    model = CopyModel(SRC, TGT).to(device)
    optimizer, scheduler = get_a_new_optimizer(cfg.optim, cfg.init_learning_rate, model.parameters())
    return model, optimizer, scheduler, bool(cfg.grad_clip), True


def create_transformer_model(model_type, SRC, TGT):
    model = model_type(SRC, TGT).to(device)
    model.init_model_params()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-7, betas=(0.9, 0.98), eps=1e-9)
    model_size = int(cfg.transformer_d_model)
    factor, warmup = int(cfg.transformer_opt_factor), int(cfg.transformer_opt_warmup * cfg.update_freq)
    scheduler = TransformerScheduler(model_size, factor, warmup, optimizer)
    return model, optimizer, scheduler, False, False


def average(model, models):
    """
        Average multiple checkpoints into one model
    """
    for ps in zip(*[m.parameters() for m in [model] + models]):
        temp = torch.zeros(ps[0].size()).to(device)
        for i in range(len(ps[1:])):
            temp += ps[i+1]
        ps[0].data.copy_(temp / len(ps[1:]))
        # ps[0].copy_(torch.sum(*ps[1:]) / len(ps[1:]))


def main(model_name):
    dp = DataProvider()
    if model_name == "sts":
        model, optimizer, scheduler, grad_clip, _ = create_sts_model(dp.SRC, dp.TGT)
    elif model_name == "transformer":
        model, optimizer, scheduler, grad_clip, _ = create_transformer_model(Transformer, dp.SRC, dp.TGT)
    elif model_name == "aspect_augmented_transformer":
        model, optimizer, scheduler, grad_clip, _ = create_transformer_model(AspectAugmentedTransformer, dp.SRC, dp.TGT)
    elif model_name == "multi_head_aspect_augmented_transformer":
        model, optimizer, scheduler, grad_clip, _ = create_transformer_model(MultiHeadAspectAugmentedTransformer, dp.SRC, dp.TGT)
    elif model_name == "syntax_infused_transformer":
        model, optimizer, scheduler, grad_clip, _ = create_transformer_model(SyntaxInfusedTransformer, dp.SRC, dp.TGT)
    elif model_name == "bert_freeze_input_transformer":
        model, optimizer, scheduler, grad_clip, _ = create_transformer_model(BertFreezeTransformer, dp.SRC, dp.TGT)
    elif model_name == "copy":
        model, optimizer, scheduler, grad_clip, _ = create_copy_model(dp.SRC, dp.TGT)
    else:
        raise ValueError("Model name {} is not defined.".format(model_name))
    if not os.path.exists("../.checkpoints/"):
        os.mkdir("../.checkpoints/")
    training_evaluation_results = []
    last_saved_model_id = 0
    max_saved_model_id = 0
    torch.save({'model': model}, "../.checkpoints/"+cfg.checkpoint_name + str(last_saved_model_id))
    model_last_saved_time = timer()
    if bool(cfg.debug_mode):
        evaluate(dp.val_iter, dp, model, dp.processed_data.addresses.val.src, dp.processed_data.addresses.val.tgt, "INIT")
    assert cfg.update_freq > 0, "update_freq must be a non-negative integer"
    n_actual_steps = cfg.n_steps * cfg.update_freq

    def get_data(steps):
        n_processed_steps = 0
        while n_processed_steps < steps:
            for example in dp.train_iter:
                yield example
                n_processed_steps += 1
                if n_processed_steps == steps:
                    break
    all_loss = 0.0
    batch_count = 0.0
    all_perp = 0.0
    all_tokens_count = 0.0
    ds = tqdm(get_data(n_actual_steps), total=n_actual_steps,  dynamic_ncols=True)
    optimizer.zero_grad()
    for step, instance in enumerate(ds):
        if instance.src[0].size(0) < 2:
            continue
        pred, _, lss, decoded_length, n_tokens = model(instance.src, instance.trg, test_mode=False, **instance.data_args)
        itm = lss.item()
        all_loss += itm
        all_tokens_count += n_tokens
        all_perp += math.exp(itm / max(n_tokens, 1.0))
        batch_count += 1.0
        lss /= (max(decoded_length, 1) * cfg.update_freq)
        lss.backward()
        if grad_clip:
            nn.utils.clip_grad_norm_(model.parameters(), float(cfg.max_grad_norm))
        if step % cfg.update_freq == 0:
            """Implementation of gradient accumulation as suggested in https://arxiv.org/pdf/1806.00187.pdf"""
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        current_perp = all_perp / batch_count
        if current_perp < 1500:
            ds.set_description("Step: {}, Average Loss: {:.2f}, Average Perplexity: {:.2f}".format(
                step+1, all_loss / all_tokens_count, current_perp))
        else:
            ds.set_description("Step: {}, Average Loss: {:.2f}".format(step+1, all_loss / all_tokens_count))
        if timer() - model_last_saved_time > cfg.save_every_n_minutes * 60:
            last_saved_model_id = (last_saved_model_id + 1) % cfg.n_checkpoints_to_keep
            max_saved_model_id = max(max_saved_model_id, last_saved_model_id)
            torch.save({'model': model}, "../.checkpoints/"+cfg.checkpoint_name + str(last_saved_model_id))
            model_last_saved_time = timer()
    if max_saved_model_id > 0:
        models = [torch.load("../.checkpoints/"+cfg.checkpoint_name + str(m_id), map_location=lambda storage, loc: storage)['model'].to(device)
                  for m_id in range(max_saved_model_id + 1)]
        print("Averaging the last {} checkpoints to create the final model".format(len(models)))
        average(model, models)
        val_l, val_bleu = evaluate(dp.val_iter, dp, model, dp.processed_data.addresses.val.src, dp.processed_data.addresses.val.tgt, "LAST")
        training_evaluation_results.append(val_bleu)
        torch.save({'model': model, 'field_src': dp.SRC, 'field_tgt': dp.TGT, 'training_evaluation_results': training_evaluation_results},
                   "../.checkpoints/"+cfg.checkpoint_name)


if __name__ == "__main__":
    start = timer()
    main(cfg.model_name)
    end = timer()
    print_running_time(end - start)

