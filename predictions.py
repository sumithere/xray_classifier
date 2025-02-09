from dataset import *
import pandas as pd
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import sklearn.metrics as sklm
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import tensorflow as tf


def make_pred_multilabel(model, test_df, val_df, path_image):
    """
    Gives predictions for test fold and calculates AUCs using previously trained model
    Args:

        model: densenet-121 from torchvision previously fine tuned to training data
        test_df : dataframe csv file
        PATH_TO_IMAGES:
    Returns:
        pred_df: dataframe containing individual predictions and ground truth for each test image
        auc_df: dataframe containing aggregate AUCs by train/test tuples
    """

    BATCH_SIZE = 32
    workers = 12

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    dataset_test = NIH(test_df, path_image=path_image, transform=transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize]))
    test_loader = torch.utils.data.DataLoader(dataset_test, BATCH_SIZE, shuffle=True, num_workers=workers,
                                              pin_memory=True)

    dataset_val = NIH(val_df, path_image=path_image, transform=transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize]))
    val_loader = torch.utils.data.DataLoader(dataset_val, BATCH_SIZE, shuffle=True, num_workers=workers,
                                             pin_memory=True)

    size = len(test_df)
    print("Test _df size :", size)
    size = len(val_df)
    print("val_df size :", size)



    # criterion = nn.BCELoss().to(device)
    # model = model.to(device)
    # to find this thresold, first we get the precision and recall withoit this, from there we calculate f1 score, using f1score, we found this theresold which has best precsision and recall.  Then this threshold activation are used to calculate our binary output.


    PRED_LABEL = ['Atelectasis',
            'Cardiomegaly',
            'Effusion',
            'Infiltration',
            'Mass',
            'Nodule',
            'Pneumonia',
            'Pneumothorax',
            'Consolidation',
            'Edema',
            'Emphysema',
            'Fibrosis',
            'Pleural_Thickening',
            'Hernia']

    for mode in ["Threshold", "test"]:
        # create empty dfs
        pred_df = pd.DataFrame(columns=["path"])
        bi_pred_df = pd.DataFrame(columns=["path"])
        true_df = pd.DataFrame(columns=["path"])

        if mode == "Threshold":
            loader = val_loader
            Eval_df = pd.DataFrame(columns=["label", 'bestthr'])
            thrs = []

        if mode == "test":
            loader = test_loader
            TestEval_df = pd.DataFrame(columns=["label", 'auc', "auprc"])

            Eval = pd.read_csv("./results/Threshold.csv")
            thrs = [Eval["bestthr"][Eval[Eval["label"] == "Atelectasis"].index[0]],
                    Eval["bestthr"][Eval[Eval["label"] == "Cardiomegaly"].index[0]],
                    Eval["bestthr"][Eval[Eval["label"] == "Effusion"].index[0]],
                    Eval["bestthr"][Eval[Eval["label"] == "Infiltration"].index[0]],
                    Eval["bestthr"][Eval[Eval["label"] == "Mass"].index[0]],
                    Eval["bestthr"][Eval[Eval["label"] == "Nodule"].index[0]],
                    Eval["bestthr"][Eval[Eval["label"] == "Pneumonia"].index[0]],
                    Eval["bestthr"][Eval[Eval["label"] == "Pneumothorax"].index[0]],
                    Eval["bestthr"][Eval[Eval["label"] == "Consolidation"].index[0]],
                    Eval["bestthr"][Eval[Eval["label"] == "Edema"].index[0]],
                    Eval["bestthr"][Eval[Eval["label"] == "Emphysema"].index[0]],
                    Eval["bestthr"][Eval[Eval["label"] == "Fibrosis"].index[0]],
                    Eval["bestthr"][Eval[Eval["label"] == "Pleural_Thickening"].index[0]],
                    Eval["bestthr"][Eval[Eval["label"] == "Hernia"].index[0]]]

        for i, data in enumerate(loader):
            inputs, labels, item = data

            # inputs = inputs.to(device)
            # labels = labels.to(device)
            batch_size = inputs.shape[0]
            inputs = inputs.permute(0, 2, 3,1)
            inputs_numpy = np.array(inputs)  # Assuming images is on GPU, move to CPU first
            labels_numpy = np.array(labels)
            item = np.array(item)

            # true_labels = labels.cpu().data.numpy()

            # batch_size = true_labels.shape

            # model.eval()
            # with torch.no_grad():
            #     outputs = model(inputs)
            #     probs = outputs.cpu().data.numpy()
            probs = model.predict(inputs_numpy)
            print("predict 125")

            # If you need to convert the outputs to numpy array

            # get predictions and true values for each item in batch
            for j in range(0, batch_size):
                thisrow = {}
                bi_thisrow = {}
                truerow = {}
                

                truerow["path"] = item[j]
                thisrow["path"] = item[j]
                if mode == "test":
                    bi_thisrow["path"] = item[j]
                    # iterate over each entry in prediction vector; each corresponds to
                    # individual label
                for k in range(len(PRED_LABEL)):
                    thisrow["prob_" + PRED_LABEL[k]] = probs[j, k]
                    truerow[PRED_LABEL[k]] = labels_numpy[j, k]

                    if mode == "test":
                       bi_thisrow["bi_" + PRED_LABEL[k]] = probs[j, k] >= thrs[k]

                # pred_df = pred_df.append(thisrow, ignore_index=True)
                pred_df = pd.concat([pred_df, pd.DataFrame(thisrow, index=[0])], ignore_index=True)
                # true_df = true_df.append(truerow, ignore_index=True)
                true_df = pd.concat([true_df, pd.DataFrame(truerow, index=[0])], ignore_index=True)
                if mode == "test":
                    # bi_pred_df = bi_pred_df.append(bi_thisrow, ignore_index=True)
                    bi_pred_df = pd.concat([bi_pred_df, pd.DataFrame(bi_thisrow, index=[0])], ignore_index=True)
                    

            if (i % 200 == 0):
                print(str(i * BATCH_SIZE))

        # print(pred_df,"true dataframe")
        for column in true_df:
            if column not in PRED_LABEL:
                continue
            actual = true_df[column]
            print(column, "column")
            # print(actual , "actual")
            pred = pred_df["prob_" + column]
            # print(pred,"prediction")
            thisrow = {}
            thisrow['label'] = column
            
            if mode == "test":
                bi_pred = bi_pred_df["bi_" + column]            
                thisrow['auc'] = np.nan
                thisrow['auprc'] = np.nan
            else:
                thisrow['bestthr'] = np.nan

            try:
#                 n_booatraps = 1000
#                 rng_seed = int(size / 100)
#                 bootstrapped_scores = []

#                 rng = np.random.RandomState(rng_seed)
#                 for i in range(n_booatraps):
#                     indices = rng.random_integers(0, len(actual.as_matrix().astype(int)) - 1, len(pred.as_matrix()))
#                     if len(np.unique(actual.as_matrix().astype(int)[indices])) < 2:
#                         continue

#                     score = sklm.roc_auc_score(
#                         actual.as_matrix().astype(int)[indices], pred.as_matrix()[indices])
#                     bootstrapped_scores.append(score)

#                 thisrow['auc'] = np.mean(bootstrapped_scores)
                
                print("line 197")
                if mode == "test":
                    print(np.array(actual), np.array(pred), "line 196")
                    
                    
                    
                    thisrow['auc'] = sklm.roc_auc_score(
                        np.array(actual), np.array(pred))
                    print("line 200")
                    

                    thisrow['auprc'] = sklm.average_precision_score(
                        np.array(actual), np.array(pred))
                    print("line 205")
                    
                else:

                    p, r, t = sklm.precision_recall_curve(np.array(actual), np.array(pred))
                    # Choose the best threshold based on the highest F1 measure
                    print("line 211")
                    f1 = np.multiply(2, np.divide(np.multiply(p, r), np.add(r, p)))
                    print("line 213")
                    # bestthr = t[np.where(f1 == max(f1))] 
                    max_index = np.argmax(f1)
                    bestthr = t[max_index]
                    print("line 215")
                    thrs.append(bestthr)
                    print(bestthr,"line 217")
                    thisrow['bestthr'] = bestthr


            except BaseException as e:
                print("can't calculate auc for " + str(column))
                print(e , "occured")

            if mode == "Threshold":
                # Eval_df = Eval_df.append(thisrow, ignore_index=True)
                Eval_df = pd.concat([Eval_df, pd.DataFrame(thisrow, index=[0])], ignore_index=True)
                

            if mode == "test":
                # TestEval_df = TestEval_df.append(thisrow, ignore_index=True)
                TestEval_df = pd.concat([TestEval_df, pd.DataFrame(thisrow, index=[0])], ignore_index=True)
                

        pred_df.to_csv("results/preds.csv", index=False)
        true_df.to_csv("results/True.csv", index=False)


        if mode == "Threshold":
            Eval_df.to_csv("results/Threshold.csv", index=False)

        if mode == "test":
            TestEval_df.to_csv("results/TestEval.csv", index=False)
            bi_pred_df.to_csv("results/bipred.csv", index=False)

    
    print("AUC ave:", TestEval_df['auc'].sum() / 14.0)

    print("done")

    return pred_df, Eval_df, bi_pred_df, TestEval_df  # , bi_pred_df , Eval_bi_df

