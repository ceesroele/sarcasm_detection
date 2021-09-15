"""
Train a model on the Reddit dataset by Khodak.
"""

import functools
import time
import logging
import pickle
import os

import pandas as pd
from sklearn.model_selection import train_test_split
from simpletransformers.classification import ClassificationModel, ClassificationArgs

from utils import (
    hour_min_sec,
    has_markdown,
    combine_with_context
)

TRAIN_FILE = 'data/train.csv'
TEST_FILE = 'data/test.csv'

EXAMPLES = 100
EPOCHS = 1
USE_CUDA = True
BATCH_SIZE = 16
MAX_COMMENT_LENGTH = 150

MODEL_ARGS = ClassificationArgs(
    eval_batch_size=BATCH_SIZE,
    train_batch_size=BATCH_SIZE,
    evaluate_during_training=True,
    evaluate_during_training_verbose=True,
    use_multiprocessing=False,
    use_multiprocessing_for_evaluation=False,
    overwrite_output_dir=True,
    save_eval_checkpoints=True,
    save_model_every_epoch=True,
    #save_steps=-1
)

# Set logging to DEBUG level
logging.basicConfig(filename='sarcasm_run.log', level=logging.DEBUG, format='%(asctime)s %(message)s')


def timer(func):
    """Timer decorator: prints elapsed time for function call and also writes it to log file"""
    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        tic = time.perf_counter()
        value = func(*args, **kwargs)
        toc = time.perf_counter()
        elapsed_time = toc - tic
        args_repr = ()
        for a in args:
            if isinstance(a, pd.DataFrame):
                args_repr += ('<DataFrame>',)
            else:
                args_repr += (a,)
        for k, v in kwargs.items():
            args_repr += (f'{k}={v}',)
        message = f"Called: {func.__name__}{args_repr}\t->\tElapsed time: {hour_min_sec(elapsed_time, hms=True)} seconds"
        print(f'{message}')
        logging.info(f'{message}')
        return value
    return wrapper_timer


def read_df_from_csv(filename):
    """Read CSV file into dataframe.
    Force string type on `comment`, `subreddit`, and `parent_comment` fields and
    convert any NaN for string values to an empty string."""
    return pd.read_csv(
        filename,
        dtype={
            'comment': pd.StringDtype(),
            'subreddit': pd.StringDtype(),
            'parent_comment': pd.StringDtype()
        },
        keep_default_na=False,  # Convert any NaN to empty string (for string dtype)
        verbose=True
    )


def result_to_metrics(result):
    """Specific for the result dictionary of simpletransformers binary classification,
    which is a dictionary including keys: `tp`, `fp`, `tn`, and `fn`.

    TP = True Positive
    FP = False Positive
    TN = True Negative
    FN = False Negative

    accuracy = (TP + TN) / (TP + FP + TN + FN)
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    :returns accuracy, precision, recall
    """
    accuracy = (result['tp'] + result['tn']) / (result['tp'] + result['fp'] + result['tn'] + result['fn'])
    positives = result['tp'] + result['fp']
    if positives > 0:
        precision = result['tp'] / (result['tp'] + result['fp'])
    else:
        # If there are no positives, we set the precision to 1
        precision = 1.0
    labeled_positives = result['tp'] + result['fn']
    if labeled_positives > 0:
        recall = result['tp'] / (result['tp'] + result['fn'])
    else:
        # If there are no labelled positives, we set the recall to 1
        recall = 1.0
    if precision + recall > 0:
        f1 = (2 * precision * recall) / (precision + recall)
    else:
        f1 = 0.0
    return accuracy, precision, recall, f1



def get_new_model(num_train_epochs=EPOCHS, use_cuda=USE_CUDA):
    logging.info(MODEL_ARGS)

    MODEL_ARGS.num_train_epochs = num_train_epochs

    # Create a ClassificationModel
    model = ClassificationModel(
        "roberta", "roberta-base", args=MODEL_ARGS, use_cuda=use_cuda
    )
    return model


def eval(model, eval_df):
    """Evaluate a model with a given evaluation dataset"""
    result, _, _ = model.eval_model(eval_df)

    print(result)

    accuracy, precision, recall, f1 = result_to_metrics(result)
    metrics_message = f'Accuracy = {accuracy:0.4f}; Precision = {precision:0.4f}; Recall = {recall:0.4f}; F1 = {f1:0.4f}'
    print(metrics_message)
    logging.info(metrics_message)
    return model


@timer
def train(train_df, dev_df, eval_df, epochs=EPOCHS, field='comment'):
    """Train the model and evaluate after training. Uses early stopping.
    :train_df Dataframe with training data
    :dev_df Dataframe with evaluation data for early stopping
    :eval_df Dataframe with evaluation data after training is completed."""
    # Optional model configuration
    model = get_new_model(num_train_epochs=epochs)

    # Convert the train dataframe to training format and train the model
    train_df = train_df[[field, 'label']]
    train_df.columns = ['text', 'labels']
    print('shape of train_df =',train_df.shape)

    # Convert the train dataframe to training format and train the model
    dev_df = dev_df[[field, 'label']]
    dev_df.columns = ['text', 'labels']
    print('shape of dev_df =', dev_df.shape)

    model.train_model(train_df, eval_df=dev_df)

    # Convert the train dataframe to training format and train the model
    eval_df = eval_df[[field, 'label']]
    eval_df.columns = ['text', 'labels']
    print('shape of eval_df = ', eval_df.shape)
    eval(model, eval_df)
    return model


def prepare_train_dataframes(df, count=100, field='comment'):
    """We split the dataset in train, dev, and eval parts.
    Next we clean all parts clean all parts, pickle them (to reproduce in any post-mortem)
    and truncate them to the wanted size."""
    train_df, _df = train_test_split(df, test_size=0.2)
    eval_df, dev_df = train_test_split(_df, test_size=0.3)

    if field == 'target':
        train_df = combine_with_context(train_df)
        dev_df = combine_with_context(dev_df)
        eval_df = combine_with_context(eval_df)
    count_eval = count
    # Minimum and maximum items for evaluation
    if count_eval > 20000:
        count_eval = 20000
    elif count_eval < 10000:
        count_eval = 10000
    else:
        count_eval = count
    count_dev = count_eval//2
    train_df = train_df.sample(n=count)
    dev_df = dev_df.sample(n=count_dev)
    eval_df = eval_df.sample(n=count_eval)
    # For the case of a post-mortem we save our samples
    with open('train_df.pkl', 'wb') as f:
        pickle.dump(train_df, f)
    with open('dev_df.pkl', 'wb') as f:
        pickle.dump(dev_df, f)
    with open('eval_df.pkl', 'wb') as f:
        pickle.dump(eval_df, f)
    return train_df, dev_df, eval_df



# FIXME: allow for including parent_comment if required
def prepare_test_dataframe(df, count=None):
    """Create dataframe suitable for testing"""
    if count is None:
        select_df = df
    else:
        select_df = df.sample(n=count)
    test_df = select_df[['id', 'comment']]
    test_df.columns = ['id', 'text']
    return test_df


# Time estimations

@timer
def estimate_training(count=100, epochs=1):
    """Estimate total training time based on time spent training a subset of the training set."""
    msg = f'Number of objects: {count}; number of epochs: {epochs}'
    print(msg)
    logging.info(msg)
    train_df = read_df_from_csv(TRAIN_FILE)
    number_of_records = train_df.shape[0]
    subtrain_df, subdev_df, subeval_df = prepare_train_dataframes(train_df, count=count)
    model = get_new_model(num_train_epochs=epochs)

    start = time.perf_counter()

    # Train and evaluate
    model.train_model(subtrain_df)
    eval(model, subeval_df)

    end = time.perf_counter()
    elapsed_time = end - start

    msg = f'Training of {count} items for {epochs} epochs took {elapsed_time:0.2f} seconds.\n' + \
          f'Total time expected for training {number_of_records} items: {hour_min_sec(elapsed_time*(number_of_records//count), hms=True)}.\n' + \
          f'Training 1000 items takes {hour_min_sec(elapsed_time*(1000//count), hms=True)}'
    print(msg)
    logging.info(msg)


@timer
def estimate_predictions(count=100):
    """Estimate total time based on time spent making predictions for subset of test set."""
    test_df = read_df_from_csv(TEST_FILE)
    number_of_records = test_df.shape[0]
    subtest_df = prepare_test_dataframe(test_df, count=count)
    model = load_best_model()

    start = time.perf_counter()
    create_result_csv(model, subtest_df, filename='dummy.csv')
    end = time.perf_counter()
    elapsed_time = end - start

    msg = f'Prediction of {count} items took {elapsed_time:0.2f} seconds.\n' + \
          f'Total time expected for {number_of_records} items: {hour_min_sec(elapsed_time*(number_of_records//count), hms=True)}.\n' + \
          f'Predicting 1000 items takes {hour_min_sec(elapsed_time*(1000//count), hms=True)}.'
    print(msg)
    logging.info(msg)


# Delivery
#  -- create CSV file with predictions
def create_result_csv(model, test_df, filename='sarcasm_predictions.csv'):
    """Create a CSV with columns `id` and `label` with predictions for all items in the test dataset"""
    predictions = make_predictions(test_df, model)
    df = pd.DataFrame({'id': test_df['id'].to_list(), 'label': predictions})
    df.to_csv(filename, index=False)


@timer
def make_predictions(test_df, model):
    """Make predictions and time them."""
    # predict
    predictions, raw_outputs = model.predict(test_df['text'].to_list())
    return predictions


def train_and_evaluate(count=EXAMPLES, epochs=EPOCHS, field='comment'):
    """Main function"""
    msg = f'Train and evaluate. {count} objects, {epochs} epochs.'
    print(msg)
    logging.info(msg)
    # Read training data
    dataset_df = read_df_from_csv(TRAIN_FILE)
    train_df, dev_df, eval_df = prepare_train_dataframes(dataset_df, count=count, field=field)
    model = train(train_df, dev_df, eval_df, epochs=epochs, field=field)
    return model


@timer
def check_markdown_impact(count=EXAMPLES, epochs=EPOCHS):
    # Read training data
    print("WITH MARKDOWN")
    logging.info("WITH MARKDOWN")
    dataset_df = read_df_from_csv(TRAIN_FILE)
    markdown_df = dataset_df[dataset_df['comment'].apply(has_markdown)]

    train_df, eval_df = prepare_train_dataframes(markdown_df, count=count)
    model = train(train_df, eval_df, epochs=epochs)

    print("WITHOUT MARKDOWN")
    logging.info("WITHOUT MARKDOWN")
    no_markdown_df = dataset_df[dataset_df['comment'].apply(lambda x: not has_markdown(x))]

    train_df, eval_df = prepare_train_dataframes(no_markdown_df, count=count)
    model = train(train_df, eval_df, epochs=epochs)


def load_best_model(dir, num_train_epochs=EPOCHS):
    """Load the best model coming out of training."""
    model_args = ClassificationArgs(
        num_train_epochs=EPOCHS,
        eval_batch_size=BATCH_SIZE,
        overwrite_output_dir=False)
    model = ClassificationModel(
        "roberta",
        dir,
        args=model_args,
        use_cuda=True
    )
    return model


def check_best_model(dir, sample=10000):
    model = load_best_model(dir)

    df = pd.read_csv(TRAIN_FILE)

    df = df.sample(n=sample)
    df = df[['comment', 'label']]
    df.columns = ['text', 'labels']

    eval(model, df)

    return model


if __name__ == '__main__':
    ### Train and evaluate
    train_and_evaluate(count=4400, epochs=20, field='comment')

    ### Utility: check the impact of markup in text on the result
    #check_markdown_impact(count=3000, epochs=7)

    ### Utility: make an estimation on how long it will take to make predictions
    #estimate_predictions()

    ### Utility: make an estimation on how long it will take to train a model
    #estimate_training(count=500, epochs=7)

    ### Utility/verification: check the manually selected 'best model' (see outcome_exploration.ipynb!)
    # by evaluating it on a part of the original dataset.
    # check_best_model('outputs/checkpoint-36000')

    ### Create a prediction of the outcome of the test data and write it to CSV
    # best_model = load_best_model('top_outputs/checkpoint-10000')
    #final_test_df = read_df_from_csv(TEST_FILE)
    #final_test_df = final_test_df[['id', 'comment']]
    #final_test_df.columns = ['id', 'text']
    #create_result_csv(best_model, final_test_df)
