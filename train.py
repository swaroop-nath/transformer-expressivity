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

def load_model_and_training_artefacts(configuration, grad_steps):
    model = TransformerCheck(mode=configuration.MODE, num_classes=configuration.NUM_CLASSES,
                            padding_value=configuration.PADDING_VALUE,
                            **configuration.TRANSFORMER_KWARGS)
    
    optimizer = Adam(params=model.parameters(), lr=configuration.LEARNING_RATE, weight_decay=configuration.WEIGHT_DECAY)
    
    # scheduler = CosineAnnealingLR(optimizer, grad_steps, eta_min=configuration.MIN_LR)
    lr_lambda = partial(get_cosine_schedule_with_warmup_lr_lambda, num_warmup_steps=configuration.WARMUP_STEPS, num_training_steps=grad_steps, min_lr=configuration.MIN_LR)
    scheduler = LambdaLR(optimizer, lr_lambda)
    return model, optimizer, scheduler

def get_data_loader(configuration, type='train'):
    assert type in ['train', 'test'], f"`type` can either be `train` or `test` | Specified type `{type}` invalid"
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
    elif type == 'test':
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

def run_batch(model, batch, eval=False):
    if eval:
        with torch.no_grad():
            outputs = model(**batch)
    else:
        outputs = model(**batch)
    
    return outputs

def run_eval(model, test_data_loader, configuration, step, run_dir):
    model.eval()
    p_bar_test = tqdm(total=len(test_data_loader), desc='Evaluating on dataset', leave=True, position=0)
    iterator = iter(test_data_loader)
    avg_loss = 0
    
    labels = None
    preds = None
    
    for step in range(len(test_data_loader)):
        batch = next(iterator)
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
   
    os.makedirs(f"./{run_dir}/eval-{step}")
    with open(f"./{run_dir}/eval-{step}/labels.npy", "w") as file:
        np.save(file, labels.cpu().numpy())
    with open(f"./{run_dir}/eval-{step}/preds.npy", "w") as file:
        np.save(file, preds.cpu().numpy())
        
    if len(labels.size()) == 2: # (bsz, seq_len_out) --> classification
        labels = labels.reshape(-1)
        preds = torch.argmax(preds, dim=-1) # (bsz, seq_len_out)
        preds = preds.reshape(-1)
        
        accuracy = torch.sum((labels == preds).to(torch.int)) / labels.size(0)
        failure_rate = 1 - accuracy
        
    if len(labels.size()) == 3: # (bsz, seq_len_out, emb_dim) --> regression
        labels = labels.reshape(-1, labels.size(-1)).cpu().numpy()
        preds = preds.reshape(-1, preds.size(-1)).cpu().numpy()
        
        total_instances = labels.shape[0]
        correct_instances = 0
        
        tree = KDTree(labels, leaf_size=1)
        for inst_id in range(preds.shape[0]):
            pred_instance = preds[inst_id]
            true_instance = labels[inst_id]
            pred_class = tree.query([pred_instance], k=1)[1].squeeze().item()
            true_class = tree.query([true_instance], k=1)[1].squeeze().item()
            
            if pred_class == true_class: correct_instances += 1
        
        failure_rate = 1 - (correct_instances / total_instances)
        
    print(f"\nEvaluation Done\nEvaluation Loss: {avg_loss}\nFailure Rate: {failure_rate * 100}\n\n")
    wandb.log({'eval/loss': avg_loss, 'eval/failure_rate': failure_rate})

def run_train(model, optimizer, scheduler, train_data_loader, test_data_loader, configuration, max_steps):
    optimizer.zero_grad(set_to_none=True)
    p_bar_train = tqdm(total=max_steps, desc='Training on dataset', leave=True, position=0)
    running_loss = [0] * configuration.LOG_STEPS
    loss_write_ptr = 0
    
    for step in range(max_steps):
        batch = next(iter(train_data_loader))
        for k, v in batch.items():
            if type(v) == torch.Tensor: batch[k] = v.to(configuration.DEVICE)
        
        outputs = run_batch(model, batch)
        assert 'loss' in outputs, f"`loss` key not found in `outputs` from the model, `loss` is needed for training"
        loss = outputs['loss']
        loss.backward()
        loss_item = loss.detach().clone().item()
        running_loss[loss_write_ptr] = loss_item
        loss_write_ptr += 1
        
        if ((step + 1) % configuration.GRAD_ACC) == 0 or ((step + 1) == max_steps):
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            
        if (step + 1) % configuration.EVAL_STEPS == 0:
            run_eval(model, test_data_loader, configuration)
            
        if (step + 1) % configuration.LOG_STEPS == 0:
            print(f"\nTraining Step Counter: {step+1}\nLoss: {np.average(running_loss)}\n\n")
            wandb.log({'train/loss': np.average(running_loss), 'train/learning_rate': scheduler.get_last_lr()[0]})
            running_loss = [0] * configuration.LOG_STEPS
            loss_write_ptr = 0
            
        p_bar_train.set_postfix(loss=loss_item)
        p_bar_train.update(1)
        
def run_sweep(config=None):
    with wandb.init(config=config) as run:
        config = wandb.config
        configuration = Configuration() 
        serialized_config_id = configuration.set_configuration_hparams(config)
        
        run_dir = "./run-files/" + configuration.DATA_PATH.split('/')[-1] + "/" + configuration.MODE
        if not os.path.exists(run_dir): os.makedirs(run_dir)
        with open(f'./{run_dir}/configuration.txt', 'w') as file:
            file.write(configuration.serialize())
        
        artifact = wandb.Artifact(name=f"sweep-files-{serialized_config_id}", type="configuration")
        
        train_data_loader = get_data_loader(configuration, type='train')
        test_data_loader = get_data_loader(configuration, type='test')
        train_steps = len(train_data_loader) * configuration.EPOCH
        model, optimizer, scheduler = load_model_and_training_artefacts(configuration, train_steps // configuration.GRAD_ACC)
        model = model.to(configuration.DEVICE)
        
        wandb.watch(model)
        run_train(model, optimizer, scheduler, train_data_loader, test_data_loader, configuration, train_steps)
        artifact.add_dir(local_path=run_dir, name="train-artifacts")
        run.log_artifact(artifact)
        
        rmtree(run_dir)
        
if __name__ == '__main__':
    wandb.login()
    sweep_id = wandb.sweep(SWEEP_CONFIGURATION, project='transformer-continuous-check')
    
    wandb.agent(sweep_id, run_sweep, count=10)