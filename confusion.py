import numpy as np

model_1 = np.sum(np.load('save/winf_1_confusion.npy'), axis=0)
model_2 = np.sum(np.load('save/winf_2_confusion.npy'), axis=0)
model_3 = np.sum(np.load('save/winf_3_confusion.npy'), axis=0)
model_4 = np.sum(np.load('save/winf_4_confusion.npy'), axis=0)
model_5 = np.sum(np.load('save/winf_5_confusion.npy'), axis=0)
model_6 = np.sum(np.load('save/winf_6_confusion.npy'), axis=0)

models = np.divide(model_1 + model_2 + model_3 + model_4 + model_5 + model_6, 6)
print(np.round(model_1))