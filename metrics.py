import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


class Metrics:

    @staticmethod
    def confusion_matrix(labels, predict, threshold) -> (int, int, int, int):
        prediction = np.greater(predict, threshold)
        tn, fp, fn, tp = confusion_matrix(labels, prediction, labels=[0, 1]).ravel()
        return tn, fp, fn, tp

    @staticmethod
    def accuracy(conf_matrix: (int, int, int, int)) -> float:
        accuracy = (conf_matrix[3] + conf_matrix[0]) / np.sum(conf_matrix)
        return accuracy

    @staticmethod
    def f_score(precision: float, recall: float) -> float:
        return (2 * precision * recall) / (precision + recall)

    @staticmethod
    def precision(conf_matrix: (int, int, int, int)) -> float:
        if (conf_matrix[3] + conf_matrix[1]) == 0:
            return 1e-10
        else:
            return conf_matrix[3] / (conf_matrix[3] + conf_matrix[1])

    @staticmethod
    def recall(conf_matrix: (int, int, int, int)) -> float:
        if (conf_matrix[3] + conf_matrix[2]) == 0:
            return 1e-10
        else:
            return conf_matrix[3] / (conf_matrix[3] + conf_matrix[2])

    @staticmethod
    def error_rate(conf_matrix: (int, int, int, int)) -> (float, float):
        far, frr = 1e-10, 1e-10
        if (conf_matrix[1] + conf_matrix[0]) != 0:
            far = conf_matrix[1] / (conf_matrix[1] + conf_matrix[0])
        if (conf_matrix[3] + conf_matrix[2]) != 0:
            frr = conf_matrix[2] / (conf_matrix[3] + conf_matrix[2])
        return far, frr

    @staticmethod
    def roc_values(labels, predict):
        fpr, tpr, threshold = roc_curve(labels, predict)
        auc_value = auc(fpr, tpr)
        dist = abs((1 - fpr) - tpr)
        eer = fpr[np.argmin(dist)]
        return fpr, tpr, auc_value, dist, eer

    @staticmethod
    def apcer(labels, predict) -> float:
        prediction = np.greater(predict, 0.5)
        tn, fp, fn, tp = confusion_matrix(labels, prediction, labels=[0, 1]).ravel()
        if (fp + tn) == 0:
            return 1e-10
        else:
            return fp / (fp + tn)

    @staticmethod
    def bpcer(labels, predict) -> float:
        prediction = np.greater(predict, 0.5)
        tn, fp, fn, tp = confusion_matrix(labels, prediction, labels=[0, 1]).ravel()
        if (fn + tp) == 0:
            return 1e-10
        else:
            return fn / (fn + tp)

    @staticmethod
    def acer(apcer: float, bpcer: float) -> float:
        return (apcer + bpcer) / 2.0

    @staticmethod
    def hter(far: float, frr: float) -> float:
        return (far + frr) / 2.0

    @staticmethod
    def plot_roc(fpr, tpr, auc_value, dist, eer):
        plt.plot(fpr, tpr, label='area under curve(auc): %0.2f' % auc_value)
        plt.plot([0, 1], [1, 0])
        plt.plot([eer, eer], [0, tpr[np.argmin(dist)]], label='@EER', linestyle='--')
        plt.xlabel('FPR')
        plt.ylabel('TPR')
        plt.title('ROC')
        plt.legend()
        plt.show()
        # plt.savefig("roc.png")

    @staticmethod
    def plot_pr(label, predict):
        precisions, recalls, thresholds = precision_recall_curve(label, predict)
        f_scores = [Metrics.f_score(i, j) for i, j in zip(precisions, recalls)]
        max_f_score_index = np.argmax(f_scores)
        recall = recalls[max_f_score_index]
        precision = precisions[max_f_score_index]
        threshold = thresholds[max_f_score_index]
        f_score = f_scores[max_f_score_index]
        if max_f_score_index:
            plt.axvline(threshold, c='r', ls=':')
            plt.scatter(threshold, precision)
            plt.text(threshold + 0.01, precision - 0.17,
                     f'f_score: {f_score:.3f}\npresision: {precision:.3f}\nrecall: {recall:.3f}\nthreshold: {threshold:.3f}')
        plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
        plt.plot(thresholds, recalls[:-1], "g--", label="Recall")
        plt.xlabel("Threshold")
        plt.legend(loc='best')
        plt.show()
        # plt.grid(b=True, which="both", axis="both", color='gray', linestyle='-', linewidth=1)
        #plt.savefig("pr_curve.png")

    @staticmethod
    def plot_histogram(label, predict, num):
        live_data_predict = [predict[i]*100 for i in range(0, len(label)) if label[i] == 0]
        spoof_data_predict = [predict[i]*100 for i in range(0, len(label)) if label[i] == 1]

        plt.hist(np.asarray(live_data_predict), density=False, bins=100, color='blue')  # density=False would make counts
        plt.hist(np.asarray(spoof_data_predict), density=False, bins=100, color='red')  # density=False would make counts
        plt.ylabel('Number')
        plt.xlabel('Prediction_' + str(num))
        plt.show()

    @staticmethod
    def plot_evaluate_results(precisions,recalls,f1scores,accuracies, ranges):
        plt.plot(ranges, precisions, "b--", label="Precision")
        plt.plot(ranges, recalls, "g--", label="Recall")
        plt.plot(ranges, f1scores, "r--", label="F1Score")
        plt.plot(ranges, accuracies, "m--", label="Accuracy")
        plt.xlabel("Checkpoints")
        plt.legend(loc='best')
        plt.show()