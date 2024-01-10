from model_code import TransformerCheck
from data_handler import get_synthetic_data_loader
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from configuration import Configuration, SWEEP_CONFIGURATION
from tqdm import tqdm
import numpy as np
from sklearn.neighbors import KDTree
import wandb
import os
from utils import get_cosine_schedule_with_warmup_lr_lambda
from functools import partial
import numpy as np
from shutil import rmtree

def load_model_and_training_artefacts(configuration, grad_steps, type='training'):
    model = TransformerCheck(mode=configuration.MODE, num_classes=configuration.NUM_CLASSES,
                            padding_value=configuration.PADDING_VALUE,
                            **configuration.TRANSFORMER_KWARGS)
    
    if type == 'inference': return model
    
    optimizer = Adam(params=model.parameters(), lr=configuration.LEARNING_RATE, weight_decay=configuration.WEIGHT_DECAY)
    
    # scheduler = CosineAnnealingLR(optimizer, grad_steps, eta_min=configuration.MIN_LR)
    lr_lambda = partial(get_cosine_schedule_with_warmup_lr_lambda, num_warmup_steps=configuration.WARMUP_STEPS // configuration.GRAD_ACC, num_training_steps=grad_steps, min_lr=configuration.MIN_LR)
    scheduler = LambdaLR(optimizer, lr_lambda)
    return model, optimizer, scheduler

def get_data_loader(configuration, type='train'):
    assert type in ['train', 'valid', 'test'], f"`type` can either be `train`, `valid` or `test` | Specified type `{type}` invalid"
    input_fname = f'{type}_input.csv'
    output_fname = f'{type}_output.csv'
    if type == 'train':
        data_loader = get_synthetic_data_loader(data_path=configuration.DATA_PATH,
                                    mode=configuration.MODE,
                                    input_fname=input_fname,
                                    output_fname=output_fname,
                                    padding_value=configuration.PADDING_VALUE,
                                    return_loss=configuration.RETURN_LOSS,
                                    batch_size=configuration.TRAIN_BATCH_SIZE,
                                    shuffle=True,
                                    num_workers=configuration.NUM_WORKERS,
                                    num_classes=configuration.NUM_CLASSES)
    elif type in ['valid', 'test']:
        data_loader = get_synthetic_data_loader(data_path=configuration.DATA_PATH,
                                    mode=configuration.MODE,
                                    input_fname=input_fname,
                                    output_fname=output_fname,
                                    padding_value=configuration.PADDING_VALUE,
                                    return_loss=configuration.RETURN_LOSS,
                                    batch_size=configuration.TEST_BATCH_SIZE,
                                    shuffle=False,
                                    num_workers=configuration.NUM_WORKERS,
                                    num_classes=configuration.NUM_CLASSES)
        
    return data_loader

def compute_metrics(labels, preds, avg_loss, set='eval'):
    if len(labels.size()) == 2: # (bsz, seq_len_out) --> classification
        labels = labels.reshape(-1) # (bsz * seq_len_out)
        preds_class = torch.argmax(preds, dim=-1) # (bsz, seq_len_out)
        preds_class = preds_class.reshape(-1) # (bsz * seq_len_out)
        
        accuracy = torch.sum((labels == preds_class).to(torch.int)) / labels.size(0)
        failure_rate = 1 - accuracy
        
        preds = preds.reshape(-1, preds.size(-1)) # (bsz * seq_len_out, emb_dim)
        correct_instances_at_k = {'1': 0, '2': 0, '3': 0, '4': 0}
        total_instances = preds.size(0)
        for inst_id in range(preds.size(0)):
            true_class = labels[inst_id].item()
            pred_instance = preds[inst_id]
            for key, val in correct_instances_at_k.items():
                if true_class in torch.topk(pred_instance, k=eval(key), dim=-1).indices: correct_instances_at_k[key] = val + 1
                
        failure_rate_at_k = {key: 1 - (correct_instances_at_k[key] / total_instances) for key in correct_instances_at_k}
        
    if len(labels.size()) == 3: # (bsz, seq_len_out, emb_dim) --> regression
        labels = labels.reshape(-1, labels.size(-1)).cpu().numpy() # (bsz * seq_len_out, emb_dim)
        preds = preds.reshape(-1, preds.size(-1)).cpu().numpy() # (bsz * seq_len_out, emb_dim)
        
        total_instances = labels.shape[0]
        correct_instances = 0
        correct_instances_at_k = {'1': 0, '2': 0, '3': 0, '4': 0}
        
        tree = KDTree(labels, leaf_size=1)
        for inst_id in range(preds.shape[0]):
            pred_instance = preds[inst_id]
            true_instance = labels[inst_id]
            pred_class = tree.query([pred_instance], k=1)[1].squeeze().item()
            true_class = tree.query([true_instance], k=1)[1].squeeze().item()
            
            pred_class_at_k = {key: tree.query([pred_instance], k=eval(key))[1].squeeze() for key in correct_instances_at_k}
            
            for key, val in correct_instances_at_k.items(): 
                if true_class in pred_class_at_k[key]: correct_instances_at_k[key] = val + 1
            
            if pred_class == true_class: correct_instances += 1

        failure_rate = 1 - (correct_instances / total_instances)
        assert correct_instances == correct_instances_at_k['1'], f"`failure_rate` and `failure_rate_at_1` not same!!"
        failure_rate_at_k = {key: 1 - (correct_instances_at_k[key] / total_instances) for key in correct_instances_at_k}
        
    log_item = {f'{set}/loss': avg_loss, f'{set}/failure_rate': failure_rate}
    log_item.update({f'{set}/failure_rate_at_{key}': val for key, val in failure_rate_at_k.items()})
    wandb.log(log_item)
    
    return failure_rate

def run_batch(model, batch, eval=False):
    if eval:
        with torch.no_grad():
            outputs = model(**batch)
    else:
        outputs = model(**batch)
    
    return outputs

def run_on_test_set(model, test_data_loader, configuration, run_dir):
    p_bar_test = tqdm(total=len(test_data_loader), desc='Evaluating on test dataset', leave=True, position=0)
    iterator = iter(test_data_loader)
    avg_loss = 0
    
    labels = None
    preds = None
    true_vectors = None
    gen_vectors = None
    enc_layers_sa_weights = None
    dec_layers_sa_weights = None
    dec_layers_xa_weights = None
    
    for step in range(len(test_data_loader)):
        batch = next(iterator)
        decoder_outputs = batch.pop("decoder_outputs")
        for k, v in batch.items():
            if type(v) == torch.Tensor: batch[k] = v.to(configuration.DEVICE)
            
        outputs = run_batch(model, batch, eval=True)
        assert 'loss' in outputs, f"`loss` key not found in `outputs` from the model, `loss` expected for logging"
        
        loss_item = outputs['loss'].detach().item()
        avg_loss = (avg_loss * step + loss_item) / (step + 1)
        
        p_bar_test.set_postfix(loss=avg_loss)
        p_bar_test.update(1)
        
        if labels is None: labels = batch['labels']
        else: labels = torch.cat((labels, batch['labels']), dim=0)
        
        if preds is None: preds = outputs['output']
        else: preds = torch.cat((preds, outputs['output']), dim=0) # (bsz * steps, seq_len_out, num_classes) if `classification` else (bsz * steps, seq_len_out, emb_dim)
        
        if true_vectors is None: true_vectors = decoder_outputs
        else: true_vectors = torch.cat((true_vectors, decoder_outputs), dim=0) # (bsz * steps, seq_len_out, emb_dim)
        
        if gen_vectors is None: gen_vectors = decoder_outputs
        else: gen_vectors = torch.cat((gen_vectors, decoder_outputs), dim=0) # (bsz * steps, seq_len_out, emb_dim)
        
        if configuration.LOG_ATTN_WEIGHTS:
            # Logging Attention Weights
            attention_weights = outputs['attention-weights']
            batch_enc_layers_sa_weights = attention_weights['enc-sa-weights']
            batch_dec_layers_sa_weights = attention_weights['dec-sa-weights']
            batch_dec_layers_xa_weights = attention_weights['dec-xa-weights']
            
            if enc_layers_sa_weights is None: enc_layers_sa_weights = batch_enc_layers_sa_weights
            else:
                for key, attn_weights in batch_enc_layers_sa_weights.items():
                    enc_layers_sa_weights[key] = torch.cat((enc_layers_sa_weights[key], attn_weights), dim=0) # concat along batch dimension
            if dec_layers_sa_weights is None: dec_layers_sa_weights = batch_dec_layers_sa_weights
            else:
                for key, attn_weights in batch_dec_layers_sa_weights.items():
                    dec_layers_sa_weights[key] = torch.cat((dec_layers_sa_weights[key], attn_weights), dim=0) # concat along batch dimension
            if dec_layers_xa_weights is None: dec_layers_xa_weights = batch_dec_layers_xa_weights
            else:
                for key, attn_weights in batch_dec_layers_xa_weights.items():
                    dec_layers_xa_weights[key] = torch.cat((dec_layers_xa_weights[key], attn_weights), dim=0) # concat along batch dimension
        
    if not os.path.exists(f"./{run_dir}/eval-test-set"): os.makedirs(f"./{run_dir}/eval-test-set")
    with open(f"./{run_dir}/eval-test-set/labels.npy", "wb") as file:
        np.save(file, labels.cpu().numpy())
    with open(f"./{run_dir}/eval-test-set/pred-labels.npy", "wb") as file:
        np.save(file, preds.cpu().numpy())
    with open(f"./{run_dir}/eval-test-set/pred-vectors.npy", "wb") as file:
        np.save(file, gen_vectors.cpu().numpy())
    with open(f"./{run_dir}/eval-test-set/true-vectors.npy", "wb") as file:
        np.save(file, true_vectors.cpu().numpy())
        
    if configuration.LOG_ATTN_WEIGHTS:
        for layer, attn_weights in enc_layers_sa_weights.items():
            if not os.path.exists(f"./{run_dir}/eval-test-set/{layer}"): os.makedirs(f"./{run_dir}/eval-test-set/{layer}")
            with open(f"./{run_dir}/eval-test-set/{layer}/enc-sa-weights.npy", "wb") as file:
                np.save(file, attn_weights)
                
        for layer, attn_weights in dec_layers_sa_weights.items():
            if not os.path.exists(f"./{run_dir}/eval-test-set/{layer}"): os.makedirs(f"./{run_dir}/eval-test-set/{layer}")
            with open(f"./{run_dir}/eval-test-set/{layer}/dec-sa-weights.npy", "wb") as file:
                np.save(file, attn_weights)
                
        for layer, attn_weights in dec_layers_xa_weights.items():
            if not os.path.exists(f"./{run_dir}/eval-test-set/{layer}"): os.makedirs(f"./{run_dir}/eval-test-set/{layer}")
            with open(f"./{run_dir}/eval-test-set/{layer}/dec-xa-weights.npy", "wb") as file:
                np.save(file, attn_weights)
        
    compute_metrics(labels, preds, avg_loss, set='test')

def run_eval(model, val_data_loader, configuration, train_step, run_dir, best_failure_rate, best_eval_train_step):
    p_bar_test = tqdm(total=len(val_data_loader), desc='Evaluating on validation dataset', leave=True, position=0)
    iterator = iter(val_data_loader)
    avg_loss = 0
    
    labels = None
    preds = None
    true_vectors = None
    gen_vectors = None
    
    for step in range(len(val_data_loader)):
        batch = next(iterator)
        decoder_outputs = batch.pop("decoder_outputs")
        for k, v in batch.items():
            if type(v) == torch.Tensor: batch[k] = v.to(configuration.DEVICE)
            
        outputs = run_batch(model, batch, eval=True)
        assert 'loss' in outputs, f"`loss` key not found in `outputs` from the model, `loss` expected for logging"
        
        loss_item = outputs['loss'].detach().item()
        avg_loss = (avg_loss * step + loss_item) / (step + 1)
        
        p_bar_test.set_postfix(loss=avg_loss)
        p_bar_test.update(1)
        
        if labels is None: labels = batch['labels']
        else: labels = torch.cat((labels, batch['labels']), dim=0)
        
        if preds is None: preds = outputs['output']
        else: preds = torch.cat((preds, outputs['output']), dim=0) # (bsz * steps, seq_len_out, num_classes) if `classification` else (bsz * steps, seq_len_out, emb_dim)
        
        if true_vectors is None: true_vectors = decoder_outputs
        else: true_vectors = torch.cat((true_vectors, decoder_outputs), dim=0) # (bsz * steps, seq_len_out, emb_dim)
        
        if gen_vectors is None: gen_vectors = outputs['vector-output']
        else: gen_vectors = torch.cat((gen_vectors, outputs['vector-output']), dim=0) # (bsz * steps, seq_len_out, emb_dim)
   
    if not os.path.exists(f"./{run_dir}/eval-{train_step}"): os.makedirs(f"./{run_dir}/eval-{train_step}")
    with open(f"./{run_dir}/eval-{train_step}/labels.npy", "wb") as file:
        np.save(file, labels.cpu().numpy())
    with open(f"./{run_dir}/eval-{train_step}/pred-labels.npy", "wb") as file:
        np.save(file, preds.cpu().numpy())
    with open(f"./{run_dir}/eval-{train_step}/pred-vectors.npy", "wb") as file:
        np.save(file, gen_vectors.cpu().numpy())
    with open(f"./{run_dir}/eval-{train_step}/true-vectors.npy", "wb") as file:
        np.save(file, true_vectors.cpu().numpy())
        
    failure_rate = compute_metrics(labels, preds, avg_loss)
    if best_failure_rate is None or failure_rate < best_failure_rate:
        print(f'{failure_rate} better than previous {best_failure_rate} | Saving state dict')
        best_model_sd = model.state_dict()
        best_failure_rate = failure_rate
        best_eval_train_step = train_step
    else: best_model_sd = None
    return best_model_sd, best_failure_rate, best_eval_train_step

def run_train(model, optimizer, scheduler, train_data_loader, val_data_loader, configuration, max_steps, run_dir):
    optimizer.zero_grad(set_to_none=True)
    p_bar_train = tqdm(total=max_steps, desc='Training on dataset', leave=True, position=0)
    running_loss = [0] * configuration.LOG_STEPS
    loss_write_ptr = 0
    best_model_sd = None
    best_failure_rate = 1.1 # Max failure rate is 1.0
    best_eval_train_step = -1
    
    for step in range(max_steps):
        batch = next(iter(train_data_loader))
        batch.pop("decoder_outputs")
        for k, v in batch.items():
            if type(v) == torch.Tensor: batch[k] = v.to(configuration.DEVICE)
        
        outputs = model(**batch) # run_batch(model, batch) # Forward Prop
        assert 'loss' in outputs, f"`loss` key not found in `outputs` from the model, `loss` is needed for training"
        loss = outputs['loss']
        loss.backward() # Backward Prop
        torch.nn.utils.clip_grad_norm_(model.parameters(), configuration.MAX_GRAD_NORM)
        loss_item = loss.detach().clone().item()
        running_loss[loss_write_ptr] = loss_item
        loss_write_ptr += 1
        
        if ((step + 1) % configuration.GRAD_ACC) == 0 or ((step + 1) == max_steps):
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            
        if (step + 1) % configuration.EVAL_STEPS == 0:
            model.eval()
            model_sd, best_failure_rate, best_eval_train_step = run_eval(model, val_data_loader, configuration, step, run_dir, best_failure_rate, best_eval_train_step)
            if model_sd is not None: best_model_sd = model_sd
            model.train()
            optimizer.zero_grad(set_to_none=True)
            break
            
        if (step + 1) % configuration.LOG_STEPS == 0:
            wandb.log({'train/loss': np.average(running_loss), 'train/learning_rate': scheduler.get_last_lr()[0]})
            running_loss = [0] * configuration.LOG_STEPS
            loss_write_ptr = 0
           
        p_bar_train.set_postfix(loss=loss_item)
        p_bar_train.update(1)
        
    wandb.log({'eval/best_failure_rate': best_failure_rate})
    return best_model_sd, best_eval_train_step, best_failure_rate
        
def run_sweep(config=None, sweep_config=None):
    with wandb.init(config=config) as run:
        config = wandb.config
        configuration = Configuration() 
        serialized_config_id = configuration.set_configuration_hparams(config)
        
        run_dir = "./run-files/" + sweep_config['name'] + "/" + run.name
        if not os.path.exists(run_dir): os.makedirs(run_dir)
        with open(f'./{run_dir}/configuration.txt', 'w') as file:
            file.write(configuration.serialize())
        
        artifact = wandb.Artifact(name=f"sweep-files-{serialized_config_id}", type="configuration")
        
        train_data_loader = get_data_loader(configuration, type='train')
        val_data_loader = get_data_loader(configuration, type='valid')
        test_data_loader = get_data_loader(configuration, type='test')
        train_steps = len(train_data_loader) * configuration.EPOCH
        model, optimizer, scheduler = load_model_and_training_artefacts(configuration, train_steps // configuration.GRAD_ACC)
        model = model.to(configuration.DEVICE).train()
        
        wandb.watch(model, log='all', log_freq=configuration.LOG_STEPS)
        
        best_model_sd, best_train_step, best_failure_rate = run_train(model, optimizer, scheduler, train_data_loader, val_data_loader, configuration, train_steps, run_dir)
        inference_model = load_model_and_training_artefacts(configuration, None, type='inference')
        print(f'Testing, Loading model from step {best_train_step} | Best Failure Rate: {best_failure_rate}')
        inference_model.load_state_dict(best_model_sd)
        inference_model = inference_model.to(configuration.DEVICE).eval()
        run_on_test_set(inference_model, test_data_loader, configuration, run_dir)
        
        artifact.add_dir(local_path=run_dir, name="train-artifacts")
        run.log_artifact(artifact)
        
        rmtree(run_dir)
        
if __name__ == '__main__':
    wandb.login()
    os.environ["WANDB_CONSOLE"] = "wrap"
    sweep_id = wandb.sweep(SWEEP_CONFIGURATION, project='transformer-continuous-experiments-v2')
    
    wandb.agent(sweep_id, lambda: run_sweep(sweep_config=SWEEP_CONFIGURATION), count=10)