

def print_matrix(true,predicted):
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    for i in range(0,len(true)):
        if true[i] == predicted[i]:
            if true[i]:
                tp = tp+1
            else:
                tn = tn+1
        else:
            if true[i]:
                fn = fn+1
            else:
                fp = fp+1

    accuracy = (tp + tn) / max(tp + tn + fp + fn,1)
    precision = tp / max(tp + fp,1)
    sensitivity = tp / max(tp + fn,1)
    specifity = tn / max(tn + fp,1)
    print("\t\tActual label")
    print("\t\tPositive\tNegative")
    print("Predicted Positive ",tp,"\t",fp)
    print("Predicted Negative ",fn,"\t",tn)
    print("==================")
    print("Accuracy = " , accuracy)
    print("Precision = ",precision)
    print("Sensitivity = ",sensitivity)
    print("Specifity = ",specifity)
    print("==================")
