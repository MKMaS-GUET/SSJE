import torch
import os
import csv
from trainer.entities import TokenSpan
CSV_DELIMETER = ';'
def swap(v1, v2):
    return v2, v1

def extend_tensor(tensor, extended_shape, fill=0):
    tensor_shape = tensor.shape

    extended_tensor = torch.zeros(extended_shape, dtype=tensor.dtype).to(tensor.device)
    extended_tensor = extended_tensor.fill_(fill)

    if len(tensor_shape) == 1:
        extended_tensor[:tensor_shape[0]] = tensor
    elif len(tensor_shape) == 2:
        extended_tensor[:tensor_shape[0], :tensor_shape[1]] = tensor
    elif len(tensor_shape) == 3:
        extended_tensor[:tensor_shape[0], :tensor_shape[1], :tensor_shape[2]] = tensor
    elif len(tensor_shape) == 4:
        extended_tensor[:tensor_shape[0], :tensor_shape[1], :tensor_shape[2], :tensor_shape[3]] = tensor

    return extended_tensor

def padded_stack(tensors, padding=0):
    dim_count = len(tensors[0].shape)

    max_shape = [max([t.shape[d] for t in tensors]) for d in range(dim_count)]
    padded_tensors = []

    for t in tensors:
        e = extend_tensor(t, max_shape, fill=padding)
        padded_tensors.append(e)

    stacked = torch.stack(padded_tensors)
    return stacked

def batch_index(tensor, index, pad=False):
    if tensor.shape[0] != index.shape[0]:
        raise Exception()

    if not pad:
        return torch.stack([tensor[i][index[i]] for i in range(index.shape[0])])
    else:
        return padded_stack([tensor[i][index[i]] for i in range(index.shape[0])])

def check_version(config, model_class, model_path):
    if os.path.exists(model_path):
        model_path = model_path if model_path.endswith('.bin') else os.path.join(model_path, 'pytorch_model.bin')
        state_dict = torch.load(model_path, map_location=torch.device('cpu'))
        config_dict = config.to_dict()

        # version check
        loaded_version = config_dict.get('SSJE_version', '1.0')
        if 'senti_classifier.weight' in state_dict and loaded_version != model_class.VERSION:
            msg = ("Current SSJE version (%s) does not match the version of the loaded model (%s). "
                   % (model_class.VERSION, loaded_version))
            msg += "Use the code matching your version or train a new model."
            raise Exception(msg)

def create_directories_dir(d):
    if d and not os.path.exists(d):
        os.makedirs(d)
    return d
def create_csv(file_path, *column_names):
    if not os.path.exists(file_path):
        with open(file_path, 'w', newline='') as csv_file:
            writer = csv.writer(csv_file, delimiter=CSV_DELIMETER, quotechar='|', quoting=csv.QUOTE_MINIMAL)

            if column_names:
                writer.writerow(column_names)
def append_csv(file_path, *row):
    if not os.path.exists(file_path):
        raise Exception("The given file doesn't exist")

    with open(file_path, 'a', newline='') as csv_file:
        writer = csv.writer(csv_file, delimiter=CSV_DELIMETER, quotechar='|', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(row)

def to_device(batch, device):
    converted_batch = dict()
    for key in batch.keys():
        converted_batch[key] = batch[key].to(device)
    return converted_batch

def get_span_tokens(tokens, span):
    inside = False
    span_tokens = []

    for t in tokens:
        if t.span[0] == span[0]:
            inside = True

        if inside:
            span_tokens.append(t)

        if inside and t.span[1] == span[1]:
            return TokenSpan(span_tokens)

    return None