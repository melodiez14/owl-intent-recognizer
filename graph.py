import numpy as np
import matplotlib.pyplot as plt

def draw_accuracy():
    model_1_acc = np.mean(np.load('save/winf_1_test_accuracy.npy'), axis=0)
    model_2_acc = np.mean(np.load('save/winf_2_test_accuracy.npy'), axis=0)
    model_3_acc = np.mean(np.load('save/winf_3_test_accuracy.npy'), axis=0)
    model_4_acc = np.mean(np.load('save/winf_4_test_accuracy.npy'), axis=0)
    model_5_acc = np.mean(np.load('save/winf_5_test_accuracy.npy'), axis=0)
    model_6_acc = np.mean(np.load('save/winf_6_test_accuracy.npy'), axis=0)

    plt.figure('Accuracy')
    plt.title('Accuracy')
    plt.plot(model_1_acc)
    plt.plot(model_2_acc)
    plt.plot(model_3_acc)
    plt.plot(model_4_acc)
    plt.plot(model_5_acc)
    plt.plot(model_6_acc)
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(
        [
            'Arch A + NR + CF + TKN', \
            'Arch A + NR + CF + TKN + SWR', \
            'Arch A + NR + CF + TKN + SWR + STM', \
            'Arch B + NR + CF + TKN', \
            'Arch B + NR + CF + TKN + SWR', \
            'Arch B + NR + CF + TKN + SWR + STM' \
        ], loc='lower right')

def draw_precision():
    model_1_precision = np.mean(np.load('save/winf_1_test_precision.npy'), axis=0)
    model_2_precision = np.mean(np.load('save/winf_2_test_precision.npy'), axis=0)
    model_3_precision = np.mean(np.load('save/winf_3_test_precision.npy'), axis=0)
    model_4_precision = np.mean(np.load('save/winf_4_test_precision.npy'), axis=0)
    model_5_precision = np.mean(np.load('save/winf_5_test_precision.npy'), axis=0)
    model_6_precision = np.mean(np.load('save/winf_6_test_precision.npy'), axis=0)

    plt.figure('Precision')
    plt.title('Precision')
    plt.plot(model_1_precision)
    plt.plot(model_2_precision)
    plt.plot(model_3_precision)
    plt.plot(model_4_precision)
    plt.plot(model_5_precision)
    plt.plot(model_6_precision)
    plt.ylabel('precision')
    plt.xlabel('epoch')
    plt.legend(
        [
            'Arch A + NR + CF + TKN', \
            'Arch A + NR + CF + TKN + SWR', \
            'Arch A + NR + CF + TKN + SWR + STM', \
            'Arch B + NR + CF + TKN', \
            'Arch B + NR + CF + TKN + SWR', \
            'Arch B + NR + CF + TKN + SWR + STM' \
        ], loc='lower right')

def draw_recall():
    model_1_recall = np.mean(np.load('save/winf_1_test_recall.npy'), axis=0)
    model_2_recall = np.mean(np.load('save/winf_2_test_recall.npy'), axis=0)
    model_3_recall = np.mean(np.load('save/winf_3_test_recall.npy'), axis=0)
    model_4_recall = np.mean(np.load('save/winf_4_test_recall.npy'), axis=0)
    model_5_recall = np.mean(np.load('save/winf_5_test_recall.npy'), axis=0)
    model_6_recall = np.mean(np.load('save/winf_6_test_recall.npy'), axis=0)

    plt.figure('Recall')
    plt.title('Recall')
    plt.plot(model_1_recall)
    plt.plot(model_2_recall)
    plt.plot(model_3_recall)
    plt.plot(model_4_recall)
    plt.plot(model_5_recall)
    plt.plot(model_6_recall)
    plt.ylabel('recall')
    plt.xlabel('epoch')
    plt.legend(
        [
            'Arch A + NR + CF + TKN', \
            'Arch A + NR + CF + TKN + SWR', \
            'Arch A + NR + CF + TKN + SWR + STM', \
            'Arch B + NR + CF + TKN', \
            'Arch B + NR + CF + TKN + SWR', \
            'Arch B + NR + CF + TKN + SWR + STM' \
        ], loc='lower right')

if __name__ == '__main__':
    draw_accuracy()
    draw_precision()
    draw_recall()
    plt.show()