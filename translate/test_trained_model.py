import torch
from configuration import cfg, device
from readers.data_provider import DataProvider
from utils.evaluation import evaluate


def test_trained_model():
    print("Loading the best trained model")
    saved_obj = torch.load(cfg.checkpoint_name, map_location=lambda storage, loc: storage)
    model = saved_obj['model'].to(device)
    model.beam_search_decoding = True
    model.beam_size = int(cfg.beam_size)
    model.beam_search_length_norm_factor = float(cfg.beam_search_length_norm_factor)
    model.beam_search_coverage_penalty_factor = float(cfg.beam_search_coverage_penalty_factor)
    SRC = saved_obj['field_src']
    TGT = saved_obj['field_tgt']
    dp = DataProvider(SRC, TGT, load_train_data=False)
    evaluate(dp.val_iter, dp, model, dp.src_val_file_address, dp.tgt_val_file_address,
             "VALID.{}".format(dp.val_iter.dataset.name), save_decoded_sentences=True)
    for test_iter, s, t in zip(dp.test_iters, dp.src_test_file_addresses, dp.tgt_test_file_addresses):
        evaluate(test_iter, dp, model, s, t, "TEST.{}".format(test_iter.dataset.name), save_decoded_sentences=True)


if __name__ == "__main__":
    test_trained_model()
