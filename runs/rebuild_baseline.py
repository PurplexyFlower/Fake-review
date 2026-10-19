"""Fit a pinned transformer or lexical baseline on the frozen grouped split.

One process per model/seed. No historical row split, historical exclusions,
test-driven hyperparameter selection, or shared baseline ledger is used.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

from rebuild_common import (
    OUT, ROOT, assert_isolated, complete, config_for, finish, get_job,
    load_plan, metrics, partitions, save_json, write_predictions,
)


def environment():
    versions = {}
    for name in ('torch', 'transformers', 'datasets', 'scikit-learn', 'peft',
                 'bitsandbytes', 'unsloth', 'numpy', 'scipy', 'pyarrow'):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    result = subprocess.run(['nvidia-smi', '--query-gpu=name,uuid,driver_version,memory.total',
                             '--format=csv,noheader'], capture_output=True, text=True, check=False)
    return {'python': sys.version, 'platform': platform.platform(), 'packages': versions,
            'gpu': result.stdout.strip(), 'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
            'vast_instance_id': os.environ.get('CONTAINER_ID'),
            'captured_at_utc': __import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat()}


def job_partitions(job):
    training = partitions(ROOT / job['data'])
    target = partitions(ROOT / f"dataset/source_grouped/modern_reviews_{job['target']}.csv")
    assert_isolated(training['train'], training['validation'])
    assert_isolated(training['train'] + training['validation'], target['test'])
    if job.get('evaluation') == 'logo':
        if any(r['generator'] == job['target'] and r['label'] == 'CG'
               for r in training['train'] + training['validation']):
            raise RuntimeError('held-out generator entered training or validation')
    return training, target['test']


def start_directory(plan, job):
    directory = OUT / job['id']
    if complete(plan, job):
        return None
    if directory.exists():
        # Preserve partial evidence, rather than silently overwriting/retraining.
        raise RuntimeError(f'incomplete existing job requires explicit recovery: {directory}')
    directory.mkdir(parents=True)
    save_json(directory / 'config.json', config_for(plan, job), exclusive=True)
    save_json(directory / 'runtime_environment.json', environment(), exclusive=True)
    return directory


def run_transformer(plan, job):
    import pyarrow  # noqa: F401; precede the heavy stack
    import numpy as np
    import torch
    from datasets import Dataset
    from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
        DataCollatorWithPadding, EarlyStoppingCallback, Trainer, TrainingArguments, set_seed)

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError('one visible CUDA GPU required')
    training, testing = job_partitions(job)
    directory = start_directory(plan, job)
    if directory is None:
        print('ALREADY_VERIFIED', job['id'], flush=True)
        return
    set_seed(job['seed'])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    tokenizer = AutoTokenizer.from_pretrained(job['model'], revision=job['revision'])
    extra = {'attn_implementation': 'sdpa'} if job['model'].startswith('answerdotai/') else {}
    model = AutoModelForSequenceClassification.from_pretrained(
        job['model'], revision=job['revision'], num_labels=2,
        id2label={0: 'OR', 1: 'CG'}, label2id={'OR': 0, 'CG': 1}, **extra)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.pad_token_id

    def tokenize(rows):
        raw = Dataset.from_list(rows)
        encoded = raw.map(lambda batch: tokenizer(batch['text_'], truncation=True,
            max_length=job['max_seq']), batched=True)
        encoded = encoded.map(lambda batch: {'labels': [int(y == 'CG') for y in batch['label']]},
                              batched=True)
        columns = {'input_ids', 'attention_mask', 'token_type_ids', 'labels'}
        return encoded.remove_columns([c for c in encoded.column_names if c not in columns])

    train_data, val_data, test_data = (tokenize(rows) for rows in
        (training['train'], training['validation'], testing))

    def eval_metrics(prediction):
        logits, labels = prediction
        if isinstance(logits, tuple):
            logits = logits[0]
        scores = torch.softmax(torch.as_tensor(logits).float(), -1)[:, 1].numpy()
        return {'accuracy': float(np.mean((scores >= .5) == labels))}

    bf16 = torch.cuda.is_bf16_supported()
    trainer = Trainer(model=model, processing_class=tokenizer,
        train_dataset=train_data, eval_dataset=val_data,
        data_collator=DataCollatorWithPadding(tokenizer), compute_metrics=eval_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=job['patience'])],
        args=TrainingArguments(output_dir=str(directory / 'checkpoints'),
            per_device_train_batch_size=job['batch'], per_device_eval_batch_size=32,
            gradient_accumulation_steps=job['effective_batch'] // job['batch'],
            learning_rate=job['learning_rate'], num_train_epochs=job['epochs'],
            warmup_ratio=job['warmup_ratio'], weight_decay=job['weight_decay'],
            bf16=bf16, fp16=not bf16, logging_steps=50, eval_strategy='steps',
            eval_steps=job['eval_steps'], save_strategy='steps', save_steps=job['eval_steps'],
            load_best_model_at_end=True, metric_for_best_model='eval_accuracy',
            greater_is_better=True, save_total_limit=1, seed=job['seed'], report_to='none'))
    # Tokenization/model download time is separate from the training timer.
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    trainer.train()
    runtime = time.monotonic() - started
    resource = {'train_runtime_s': runtime, 'executed_steps': trainer.state.global_step,
        'best_step': int(Path(trainer.state.best_model_checkpoint).name.rsplit('-', 1)[-1]),
        'peak_allocated_gib': torch.cuda.max_memory_allocated() / 1024 ** 3,
        'peak_reserved_gib': torch.cuda.max_memory_reserved() / 1024 ** 3,
        'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
        'total_parameters': sum(p.numel() for p in model.parameters())}
    result = {'resources': resource}
    for name, encoded, rows in (('val', val_data, training['validation']), ('test', test_data, testing)):
        prediction = trainer.predict(encoded).predictions
        if isinstance(prediction, tuple):
            prediction = prediction[0]
        scores = torch.softmax(torch.as_tensor(prediction).float(), -1)[:, 1].numpy().tolist()
        write_predictions(directory / f'{name}_probs.csv', rows, scores)
        result[name] = metrics(rows, scores)
    trainer.save_model(str(directory / 'model'))
    tokenizer.save_pretrained(directory / 'model')
    save_json(directory / 'metrics.json', result, exclusive=True)
    save_json(directory / 'log_history.json', trainer.state.log_history, exclusive=True)
    trainer.state.save_to_json(str(directory / 'trainer_state.json'))
    # Checkpoints are our temporary intermediates; keep the selected model.
    checkpoint_root = (directory / 'checkpoints').resolve()
    if checkpoint_root.parent != directory.resolve():
        raise RuntimeError('unexpected checkpoint path')
    shutil.rmtree(checkpoint_root)
    load_plan()  # re-check every frozen data/code identity after training
    finish(plan, job)
    print('COMPLETE', job['id'], result['test'], flush=True)


def run_tfidf(plan, job):
    import pyarrow  # noqa: F401
    import joblib
    import numpy as np
    from scipy.sparse import hstack
    from scipy.special import expit
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import LinearSVC
    from threadpoolctl import threadpool_limits

    training, testing = job_partitions(job)
    directory = start_directory(plan, job)
    if directory is None:
        print('ALREADY_VERIFIED', job['id'], flush=True)
        return
    vectorizer = TfidfVectorizer(sublinear_tf=True, ngram_range=(1, 2), min_df=job['min_df'],
                               max_features=job['word_features'])
    characters = TfidfVectorizer(sublinear_tf=True, analyzer='char_wb', ngram_range=(3, 5),
                               min_df=job['min_df'], max_features=job['char_features'])
    labels = np.asarray([r['label'] == 'CG' for r in training['train']], dtype=int)
    texts = [r['text_'] for r in training['train']]
    started = time.monotonic()
    matrix = hstack([vectorizer.fit_transform(texts), characters.fit_transform(texts)]).tocsr()
    if job['estimator'] == 'logreg':
        estimator = LogisticRegression(C=job['C'], max_iter=job['max_iter'], random_state=job['seed'])
    else:
        estimator = LinearSVC(C=job['C'], max_iter=job['max_iter'], random_state=job['seed'])
    with threadpool_limits(limits=4):
        estimator.fit(matrix, labels)
    if estimator.n_iter_ is not None and np.max(estimator.n_iter_) >= estimator.max_iter:
        raise RuntimeError('lexical baseline did not converge; no completed result exported')
    result = {'resources': {'fit_and_vectorization_runtime_s': time.monotonic() - started,
              'features': matrix.shape[1]},
              'score_kind': 'probability' if job['estimator'] == 'logreg'
                            else 'sigmoid_of_uncalibrated_margin',
              'seed_note': 'lbfgs logistic fitting is deterministic; random_state need not change the solution'}
    for name, rows in (('val', training['validation']), ('test', testing)):
        texts = [r['text_'] for r in rows]
        features = hstack([vectorizer.transform(texts), characters.transform(texts)]).tocsr()
        scores = (estimator.predict_proba(features)[:, 1] if job['estimator'] == 'logreg'
                  else expit(estimator.decision_function(features))).tolist()
        write_predictions(directory / f'{name}_probs.csv', rows, scores)
        result[name] = metrics(rows, scores)
    joblib.dump({'word': vectorizer, 'char': characters, 'estimator': estimator}, directory / 'model.joblib')
    save_json(directory / 'metrics.json', result, exclusive=True)
    load_plan()
    finish(plan, job)
    print('COMPLETE', job['id'], result['test'], flush=True)


def evaluate_adapter(plan, job, run_dir):
    import pyarrow  # noqa: F401
    import torch
    from unsloth import FastModel
    from transformers import AutoModelForSequenceClassification

    training, testing = job_partitions(job)
    directory = start_directory(plan, job)
    if directory is None:
        return
    run_dir = Path(run_dir)
    original_config = __import__('json').loads((run_dir / 'config.json').read_text())
    save_json(directory / 'training_config.json', original_config, exclusive=True)
    original_metrics = __import__('json').loads((run_dir / 'metrics.json').read_text())
    save_json(directory / 'training_metrics.json', original_metrics, exclusive=True)
    for name in ('adapter', 'log_history.json', 'trainer_state.json'):
        source = run_dir / name
        if source.is_dir():
            shutil.copytree(source, directory / name)
        elif source.exists():
            shutil.copy2(source, directory / name)
    model, tokenizer = FastModel.from_pretrained(model_name=str(directory / 'adapter'),
        auto_model=AutoModelForSequenceClassification, max_seq_length=512, dtype=None,
        num_labels=2, load_in_4bit=True, full_finetuning=False)
    FastModel.for_inference(model)
    model.eval()
    device = next(model.parameters()).device
    from safetensors import safe_open
    import math
    with safe_open(directory / 'adapter/adapter_model.safetensors', framework='pt', device='cpu') as weights:
        saved_count = sum(math.prod(weights.get_slice(key).get_shape()) for key in weights.keys())
    result = {'resources': {key: value for key, value in original_metrics.items()
                           if key not in {'val', 'test'}} | {'saved_adapter_parameter_count': saved_count},
        'raw_run_directory': str(run_dir.relative_to(ROOT)),
        'evaluation_note': 'reloaded selected adapter; matched unseen-source DeepSeek test'}
    for name, rows in (('val', training['validation']), ('test', testing)):
        scores = []
        for start in range(0, len(rows), 32):
            batch = rows[start:start + 32]
            inputs = tokenizer([r['text_'] for r in batch], padding=True, truncation=True,
                max_length=512, return_tensors='pt').to(device)
            with torch.inference_mode():
                values = torch.softmax(model(**inputs).logits.float(), -1)[:, 1]
            scores.extend(values.cpu().tolist())
        write_predictions(directory / f'{name}_probs.csv', rows, scores)
        result[name] = metrics(rows, scores)
    save_json(directory / 'metrics.json', result, exclusive=True)
    load_plan()
    finish(plan, job)
    print('COMPLETE', job['id'], result['test'], flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--job-id', required=True)
    parser.add_argument('--adapter-run-dir')
    args = parser.parse_args()
    plan = load_plan()
    job = get_job(plan, args.job_id)
    if job['kind'] == 'transformer':
        run_transformer(plan, job)
    elif job['kind'] == 'tfidf':
        run_tfidf(plan, job)
    elif args.adapter_run_dir:
        evaluate_adapter(plan, job, ROOT / args.adapter_run_dir)
    else:
        raise RuntimeError('adapter evaluation requires the exact saved run directory')


if __name__ == '__main__':
    main()
