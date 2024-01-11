# Modules specially for machine learning
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup

# Modules for data science in general
import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt

# basic python modules
import os
import json
import random
from tqdm import tqdm

rel_idx = {
    "SUBSIDIARY_OF": "0",
    "FOUNDED_BY": "1",
    "EMPLOYEE_OR_MEMBER_OF": "2",
    "CEO": "3",
    "DATE_FOUNDED": "4",
    "HEADQUARTERS": "5",
    "EDUCATED_AT": "6",
    "NATIONALITY": "7",
    "PLACE_OF_RESIDENCE": "8",
    "PLACE_OF_BIRTH": "9",
    "DATE_OF_DEATH": "10",
    "DATE_OF_BIRTH": "11",
    "SPOUSE": "12",
    "CHILD_OF": "13",
    "POLITICAL_AFFILIATION": "14",
    "PLACE_OF_DEATH": "15",
    "AUTHOR": "16"
    }

def vec_to_rel(ls):
    res = []
    rels = ["SUBSIDIARY_OF", "FOUNDED_BY", "EMPLOYEE_OR_MEMBER_OF", "CEO", "DATE_FOUNDED", "HEADQUARTERS", "EDUCATED_AT", "NATIONALITY", "PLACE_OF_RESIDENCE", "PLACE_OF_BIRTH", "DATE_OF_DEATH", "DATE_OF_BIRTH", "SPOUSE", "CHILD_OF", "POLITICAL_AFFILIATION", "PLACE_OF_DEATH", "AUTHOR"]
    for x,y in zip(ls,rels):
        if x == 1:
            res.append(y)
    return res


def calc_metrics2(text, kn_list, my_list, dict_tp, dict_tn, dict_fp, dict_fn, label_length):

    for i in range(label_length):
        
        if (kn_list[i] == 1) and (my_list[i] == 1):
            if not(text in dict_tp):
                dict_tp[text] = [0] * label_length
            dict_tp[text][i] = 1
            dict_tp["counter"] += 1
            
        elif (kn_list[i] == 0) and (my_list[i] == 1):
            if not(text in dict_fp):
                dict_fp[text] = [0] * label_length
            dict_fp[text][i] = 1
            dict_fp["counter"] += 1
            
        elif (kn_list[i] == 1) and (my_list[i] == 0):
            if not(text in dict_fn):
                dict_fn[text] = [0] * label_length
            dict_fn[text][i] = 1
            dict_fn["counter"] += 1
        
        elif (kn_list[i] == 0) and (my_list[i] == 0):
            if not(text in dict_tn):
                dict_tn[text] = [0] * label_length
            dict_tn[text][i] = 1
            dict_tn["counter"] += 1


def trainee(train_ds):
    train_df = train_ds
    
    batch_size = 8
    num_epochs = 10
    learning_rate = 2e-5
    max_sequence_length = 128  # Maximum sequence length for padding

    # We use a pretrained BERTTokenizer from huggingface
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Convert labels(hot encoded vectors) to tensors
    labels = torch.tensor(train_df['label'].tolist(), dtype=torch.float32)
    print(len(labels[0]))
    # Create a list with all the facts as training input
    texts = train_df['passageText'].tolist()

    # Tokenize, pad, and create input masks
    input_ids = []
    attention_masks = []

    for text in texts:
        # take a fact from list texts and convert it into tokens, which are then directly transformed into tensors
        encoding = tokenizer(text, truncation=True, padding='max_length', max_length=max_sequence_length, return_tensors='pt')
        # Take the inputId of the tensor and put the in separate list
        input_ids.append(encoding['input_ids'])
        # Take the attention_mask of the tensor and put the in separate list
        attention_masks.append(encoding['attention_mask'])

    # Convert lists of tensors into a batch
    input_ids = torch.cat(input_ids, dim=0)  # Use torch.cat to concatenate tensors
    attention_masks = torch.cat(attention_masks, dim=0)

    # Create DataLoader for training
    # Create the final dataset we will input into the model for training
    dataset = TensorDataset(input_ids, attention_masks, labels)

    # Creates the dataloader (Iterable which will subdived the dataset into mini-batches of size batch_size)
    train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Define the BERT model for classification
    # we are using pretrained model from huggingface
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(labels[0]))

    # We are using Binary Cross Entropy with a Sigmoid layer on top,
    # since we want the model to treat the categories independently for multi-label
    criterion = nn.BCEWithLogitsLoss()
    # We are using a variant of the traditional Adam optimizer to prevent overfitting
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    # We will keep the learning rate constant from the beginning
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=len(train_dataloader) * num_epochs)
    
    
    # Training loop with loading bar
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for batch in tqdm(train_dataloader, desc=f'Epoch {epoch + 1}'):
            input_ids, attention_masks, labels = batch
            # input_ids, attention_masks, labels = input_ids.cuda(), attention_masks.cuda(), labels.cuda()

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask=attention_masks, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()

            loss.backward()
            optimizer.step()
            scheduler.step()

        average_loss = total_loss / len(train_dataloader)
        print(f'Epoch {epoch + 1} - Average Loss: {average_loss:.4f}')
    
    return model




from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def evaluator(test_set, model, thresh, length):
    val_df = test_set
    
    # Load and preprocess the test dataset
    # Labels
    test_labels = torch.tensor(val_df['label'].tolist(), dtype=torch.float32)
    # Text
    test_texts = val_df['passageText'].tolist()
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    max_sequence_length = 128
    batch_size = 8
    num_epochs = 10
    learning_rate = 2e-5

    # print some information
    print("Number of test examples:", len(test_texts))
    print("Test text example:", test_texts[0])
    print("Test label example:", test_labels[0])

    test_input_ids = []
    test_attention_masks = []

    # iterates over all the text samples
    for text in test_texts:
        # encodes text -> we will have the "input_ids" and the "attention_mask" of each text
        encoding = tokenizer(text, truncation=True, padding='max_length', max_length=max_sequence_length, return_tensors='pt')
        # puts them into two seperate lists
        test_input_ids.append(encoding['input_ids'])
        test_attention_masks.append(encoding['attention_mask'])

    # Check if the test_input_ids and test_attention_masks lists are empty
    if not test_input_ids or not test_attention_masks:
        raise ValueError("Test data is empty or improperly formatted.")

    test_input_ids = torch.cat(test_input_ids, dim=0)
    test_attention_masks = torch.cat(test_attention_masks, dim=0)

    # Create DataLoader for testing
    test_dataset = TensorDataset(test_input_ids, test_attention_masks, test_labels)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Set the model to evaluation mode
    model.eval()
    #model.eval()

    # Variables to store evaluation metrics
    total_correct = 0
    total_samples = 0

    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    # ...

    # Variables to store evaluation metrics
    true_labels = []
    predicted_labels = []

    # Loop through the test DataLoader
    with torch.no_grad():
        for test_batch in tqdm(test_dataloader, desc='Testing'):
            test_input_ids, test_attention_masks, test_labels = test_batch

            # Forward pass
            outputs = model(test_input_ids, attention_mask=test_attention_masks)
            # outputs = model(test_input_ids, attention_mask=test_attention_masks)

            # Convert logits to predictions (you may need to adjust this depending on your threshold)
            predictions = (torch.sigmoid(outputs.logits) > thresh).to(torch.float32)

            true_labels.append(test_labels.cpu().numpy())
            predicted_labels.append(predictions.cpu().numpy())

    # Concatenate true labels and predicted labels
    true_labels = np.concatenate(true_labels, axis=0)
    predicted_labels = np.concatenate(predicted_labels, axis=0)

    # Calculate accuracy, precision, recall, and F1-score
    accuracy = accuracy_score(true_labels, predicted_labels)
    precision = precision_score(true_labels, predicted_labels, average='micro')
    recall = recall_score(true_labels, predicted_labels, average='micro')
    f1 = f1_score(true_labels, predicted_labels, average='micro')
    print("Threshold:      ", thresh)
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"Precision: {precision * 100:.2f}%")
    print(f"Recall: {recall * 100:.2f}%")
    print(f"F1-Score: {f1 * 100:.2f}%")
    
    return (thresh, f1, length)

def save_model(model, path1, batch_size, num_epochs, learning_rate,max_sequence_length):
    model_filename = path1 + ".pth"
    save_directory = '../../models'
    model_path = os.path.join(save_directory, model_filename)
    # Get the model configuration
    model_config = model.config

    # Create a dictionary to save model configuration along with the state_dict
    model_info = {
        "config": model_config,
        "state_dict": model.state_dict(),
        "other_info": {
            "batch_size": batch_size,
            "num_epochs": num_epochs,
            "learning_rate": learning_rate,
            "max_sequence_length": max_sequence_length,
        }
    }

    #model_filename = "bert_classification_model.pth"
    model_path = os.path.join(save_directory, model_filename)

    # Save the model configuration and its state dictionary
    torch.save(model_info, model_path)



def model_loader(filename):
    # Define the path to the saved model
    model_filename = filename
    save_directory = '../../models'

    model_path = os.path.join(save_directory, model_filename)
    saved_model_info = torch.load(model_path)

    # Retrieve model configuration and state dictionary from loaded information
    loaded_model_config = saved_model_info['config']
    loaded_state_dict = saved_model_info['state_dict']
    loaded_other_info = saved_model_info['other_info']

    # Assuming 'batch_size', 'num_epochs', 'learning_rate', 'max_sequence_length' are present in 'other_info'
    batch_size = loaded_other_info['batch_size']
    num_epochs = loaded_other_info['num_epochs']
    learning_rate = loaded_other_info['learning_rate']
    max_sequence_length = loaded_other_info['max_sequence_length']

    # Create a new instance of the model using the configuration
    loaded_model = BertForSequenceClassification(loaded_model_config)

    # Load the state dictionary into the model
    loaded_model.load_state_dict(loaded_state_dict)

    return loaded_model


def evaluator_3(res,thresh, train_df):
    relations = ["SUBSIDIARY_OF", "FOUNDED_BY", "EMPLOYEE_OR_MEMBER_OF", "CEO", "DATE_FOUNDED", "HEADQUARTERS", "EDUCATED_AT", "NATIONALITY", "PLACE_OF_RESIDENCE", "PLACE_OF_BIRTH", "DATE_OF_DEATH", "DATE_OF_BIRTH", "SPOUSE", "CHILD_OF", "POLITICAL_AFFILIATION", "PLACE_OF_DEATH", "AUTHOR"]
    label_len = 17

    dict_tp = {}
    dict_fp = {}
    dict_fn = {}
    dict_tn = {}
    dict_tp["counter"] = 0
    dict_fp["counter"] = 0
    dict_fn["counter"] = 0
    dict_tn["counter"] = 0

    num_tp_rel = {}
    num_fp_rel = {}
    num_fn_rel = {}
    num_tn_rel = {}

    accuracy_rel = {}
    prec_rel = {}
    recall_rel = {}
    f1_rel = {}
    
    
    
    for index, row in res.iterrows():
        tx = row["text"]
        knl = row["KN_label"]
        myl = row["My_label"]

        calc_metrics2(tx, knl, myl, dict_tp, dict_tn, dict_fp, dict_fn, label_len)

    fin_res = pd.DataFrame(columns=["text","KN_label","My_label"])
    fin_res["text"] = res["text"]
    fin_res["KN_label"] = res["KN_label"].apply(vec_to_rel)
    fin_res["My_label"] = res["My_label"].apply(vec_to_rel)

    # Will compute the amount of tp,fp and fn for all relations separetly in saves it to a dict
    for i in relations:
        num_tp_rel[i] = 0
        num_fp_rel[i] = 0
        num_fn_rel[i] = 0
        num_tn_rel[i] = 0
        for idx,row in fin_res.iterrows():
            if (i in row['KN_label']) and (i in row["My_label"]):
                num_tp_rel[i] += 1

            elif (i in row['KN_label']) and not(i in row["My_label"]):
                num_fn_rel[i] += 1

            elif not(i in row['KN_label']) and (i in row["My_label"]):
                num_fp_rel[i] += 1

            elif not(i in row['KN_label']) and not(i in row["My_label"]):
                num_tn_rel[i] += 1


    tp_df = pd.DataFrame(columns=["text", "label"])
    for key,value in dict_tp.items():
        if key != "counter":
            new_row = pd.DataFrame({"text": [key], "label": [value]})
            tp_df = pd.concat([tp_df,new_row], ignore_index=True)

    tp_df["label"] = tp_df["label"].apply(vec_to_rel)



    fp_df = pd.DataFrame(columns=["text", "label"])
    for key,value in dict_fp.items():
        if key != "counter":
            new_row = pd.DataFrame({"text": [key], "label": [value]})
            fp_df = pd.concat([fp_df,new_row], ignore_index=True)

    fp_df["label"] = fp_df["label"].apply(vec_to_rel)



    fn_df = pd.DataFrame(columns=["text", "label"])
    for key,value in dict_fn.items():
        if key != "counter":
            new_row = pd.DataFrame({"text": [key], "label": [value]})
            fn_df = pd.concat([fn_df,new_row], ignore_index=True)

    fn_df["label"] = fn_df["label"].apply(vec_to_rel)


    tn_df = pd.DataFrame(columns=["text", "label"])
    for key,value in dict_fn.items():
        if key != "counter":
            new_row = pd.DataFrame({"text": [key], "label": [value]})
            tn_df = pd.concat([tn_df,new_row], ignore_index=True)

    tn_df["label"] = tn_df["label"].apply(vec_to_rel)




    for rel in relations:
        accuracy_rel[rel] = (num_tp_rel[rel]+num_tn_rel[rel])/(num_tp_rel[rel]+num_tn_rel[rel]+num_fp_rel[rel]+num_fn_rel[rel])
        prec_rel[rel] = num_tp_rel[rel]/(num_tp_rel[rel] + num_fp_rel[rel])
        recall_rel[rel] = num_tp_rel[rel]/(num_tp_rel[rel] + num_fn_rel[rel])
        f1_rel[rel] = 2*((prec_rel[rel]*recall_rel[rel])/(prec_rel[rel]+recall_rel[rel]))

    metr_names = ["Accuracy", "Precision", "Recall", "F1-Score", "TP", "TN", "FP", "FN"]
    diclis = [accuracy_rel, prec_rel,recall_rel,f1_rel,num_tp_rel, num_tn_rel, num_fp_rel, num_fn_rel]
    
    rel_metr_df = pd.DataFrame(columns=['RELATION','#TP', '#TN','#FP', '#FN', '#TRAIN_EXPL','Accuracy', 'Precision', 'Recall', 'F1-Score'])

    for i in relations:
        #print(i)
        row = {}
        row["RELATION"] = i
        row["#TP"] = diclis[4][i]
        row["#TN"] = diclis[5][i]
        row["#FP"] = diclis[6][i]
        row["#FN"] = diclis[7][i]
        row["#TRAIN_EXPL"] = (train_df['label'].apply(lambda x: x[int(rel_idx[i])] == 1)).sum()
        row["Accuracy"] = diclis[0][i]
        row["Precision"] = diclis[1][i]
        row["Recall"] = diclis[2][i]
        row["F1-Score"] = diclis[3][i]

        rel_metr_df.loc[len(rel_metr_df)] = row


    acc_macro = rel_metr_df['Accuracy'].sum()/len(rel_metr_df)
    prec_macro = rel_metr_df['Precision'].sum()/len(rel_metr_df)
    rec_macro = rel_metr_df['Recall'].sum()/len(rel_metr_df)
    f1_macro = rel_metr_df['F1-Score'].sum()/len(rel_metr_df)

    acc_micro = (rel_metr_df['#TP'].sum() + rel_metr_df['#TN'].sum()) / (rel_metr_df['#TP'].sum() + rel_metr_df['#TN'].sum() + rel_metr_df['#FP'].sum() + rel_metr_df['#FN'].sum())
    prec_micro = rel_metr_df['#TP'].sum() / (rel_metr_df['#TP'].sum() + rel_metr_df['#FP'].sum())
    rec_micro = rel_metr_df['#TP'].sum() / (rel_metr_df['#TP'].sum() + rel_metr_df['#FN'].sum())
    f1_micro = rel_metr_df['#TP'].sum() / (rel_metr_df['#TP'].sum() + 0.5 * (rel_metr_df['#FP'].sum() + rel_metr_df['#FN'].sum()))

    all_row = {'RELATION': ['MACRO AVERAGE', 'MICRO AVERAGE'],
              '#TP': [rel_metr_df['#TP'].sum(), None],
              '#TN': [rel_metr_df['#TN'].sum(), None],
              '#FP': [rel_metr_df['#FP'].sum(), None],
              '#FN': [rel_metr_df['#FN'].sum(), None],
              '#TRAIN_EXPL': [rel_metr_df['#TRAIN_EXPL'].sum(), None],
              'Accuracy': [acc_macro, acc_micro],
              'Precision': [prec_macro, prec_micro],
              'Recall': [rec_macro, rec_micro],
              'F1-Score': [f1_macro, f1_micro]}

    df_micmac = pd.DataFrame(all_row)

    rel_metr_df = pd.concat([rel_metr_df, df_micmac], ignore_index=True)
    
    return [thresh, rel_metr_df]

print("Script was successfully imported")


