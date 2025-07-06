import re
import collections
import sys
import os
import traceback
sys.path.append(os.path.abspath(".."))
import scipy.sparse
import numpy as np
import torch
from torch.utils.data import Dataset
import copy
import datetime

def tokenize_corpus(texts, minSize = 4, regul = r"[\w\d]+"):
  TOKEN_RE = re.compile(regul)
  tokenized_texts = []
  for text in texts:
    tokens = TOKEN_RE.findall(text)
    tokenized_texts.append([token.lower() for token in tokens if len(token)>=minSize])
  return tokenized_texts
def build_vocabulary_and_freq(tokenized_collection, minSize = 0, maxfreq = 1, maxCount = 10000000, returnDF = False):
  doc_n = 0
  word_count = collections.defaultdict(int)
  for tokens in tokenized_collection:
    doc_n +=1
    unique_tokens = set(tokens)
    for token in unique_tokens:
      word_count[token]+=1

  #удаление малозначимых токенов
  word_count = {word : count for word, count in word_count.items() if count>=minSize and count/doc_n<=maxfreq}
  sorted_word_count = sorted(word_count.items(), reverse = False, key = lambda pair : (pair[1], pair[0]))
  if (len(sorted_word_count))>maxCount:
    sorted_word_count = sorted_word_count[:maxCount]
  print("debug")
  print(sorted_word_count)
  word2id = {word : rank for rank, (word, _) in enumerate(sorted_word_count)}
  word2freq = np.array([cnt/doc_n for _, cnt in sorted_word_count], dtype = "float32")
  if (returnDF):
    DF = [(word, cnt/doc_n) for word, cnt in sorted_word_count]
    return DF
  
  return word2id, word2freq
def vectorize_texts(tokenize_corpus, word2id, word2freq, mode, scale = True):
  assert mode in {"bin", "tf", "idf", "tfidf", "custom_tfidf"}
  print(tokenize_corpus)
  result = scipy.sparse.dok_matrix((len(tokenize_corpus), len(word2id)), dtype = "float32")
  for text_i, text in enumerate(tokenize_corpus):
    for token in text:
      if token in word2id:
        result[text_i, word2id[token]] +=1
  if mode == "bin":
    result = (result>0).astype("float32")
  elif mode == "tf":
     result = result.tocsr()
     result = result.multiply(1/result.sum(1))
  elif mode == "idf":
    result = result.tocsr()
    result = (result>0).astype("Float32").multiply(1/word2freq)
  elif mode == "tfidf":
    result = result.tocsr()
    result = result.multiply(1/result.sum(1))
    result = result.multiply(1/word2freq)
  elif mode == "custom_tfidf":
    result = result.tocsr()
    result = result.multiply(1/result.sum(1)+1e-9)
    result.data = np.log(result.data+1)
    result = result.multiply(1/word2freq)
  if scale:
    result = result.tocsc()
    result-=result.min()
    result/=result.max()+1e-6
    #result = result.toarray()
    #std_x = result.std(axis = 0, ddof = 1)
    #mean = result.mean(axis = 0)
    #result = (result-mean)/std_x
    print("dubag count")
    print(result.toarray())
    print("dubag")
    return result.tocsr()

class SparseFeatureDataset(Dataset):
  def __init__(self, feature, target):
    self.feature = feature
    self.target = target
  def __len__(self):
    return self.feature.shape[0]
  def __getitem__(self, idx):
      cur_feature = torch.from_numpy(self.feature[idx].toarray()[0]).float()
      cur_target = torch.tensor(self.target[idx], dtype = torch.long)
      return cur_feature, cur_target
def train_eval_loop(model, train_dataset, test_dataset,
                    lr = 1e-4, epoch = 10, batch_size = 32, l2_reg_alpha = 0, criterion = None, epoch_n = 200,
                    device = 'cpu', optimizer_ctor = None, lr_scheduler_ctor = None, shuffle_train = True,
                    max_batch_per_epoch_val = 1000, max_batch_per_epoch_train = 10000, pattience_epoch = 10
                    ):
  device = torch.device(device)
  model.to(device)
  
  if (optimizer_ctor is None):
    optimizer = torch.optim.Adam(model.parameters(), lr = lr, weight_decay=l2_reg_alpha)
  else:
    optimizer = optimizer_ctor(model.parameters(), lr = lr)
  if lr_scheduler_ctor is not None:
    lr_scheduler = lr_scheduler_ctor(optimizer)
  else:
    lr_scheduler = None
  train_dataLoader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle = shuffle_train)
  test_dataLoader = torch.utils.data.DataLoader(test_dataset, batch_size = batch_size, shuffle = False)
  
  best_model = copy.deepcopy(model)
  best_epoch_i = 0
  best_val_loss = float("inf")
  for epoch in range(epoch_n):
    try:
      epoch_start = datetime.datetime.now()
      print("Эпоха: {}".format(epoch))
      model.train()
      mean_train_loss = 0
      train_batches = 0
      for batch_i, (batch_x, batch_y) in enumerate(train_dataLoader):
        if batch_i > max_batch_per_epoch_train:
          break
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        pred = model(batch_x)
        #print(type(batch_x))
        loss = criterion(pred, batch_y)
        model.zero_grad()
        loss.backward()
        optimizer.step()
        train_batches+=1
        mean_train_loss+=float(loss)
      mean_train_loss/=train_batches
      print("Эпоха: {} итераций, {:0.2f} секунд".format(batch_i, (datetime.datetime.now() - epoch_start).total_seconds()))
      print("Среднее значение функции потерь на выборке: ", mean_train_loss)
      mean_test_loss = 0
      val_bench_n = 0
      model.eval()
      with torch.no_grad():
        for batch_i, (batch_x, batch_y) in enumerate(train_dataLoader):
          if batch_i > max_batch_per_epoch_val:
            break
          batch_x = batch_x.to(device)
          batch_y = batch_y.to(device)
          pred = model(batch_x)
          loss = criterion(pred, batch_y)
          mean_test_loss+=float(loss)
          val_bench_n+=1
      mean_test_loss/=val_bench_n
      print("Среднее значение функции потерь на валидаци: ", mean_test_loss)
      if (mean_test_loss < best_val_loss):
        best_epoch_i = epoch
        best_model = copy.deepcopy(model)
        best_val_loss = mean_test_loss
      elif (epoch-best_epoch_i>pattience_epoch):
        print("После {} эпох результат функции потерь не изменился, прерываю обучение".format(pattience_epoch))
        break
      if lr_scheduler is not None:
        lr_scheduler.step(mean_test_loss)
      print()
    except KeyboardInterrupt:
      print("Досрочно остановлено пользователем")
      break
    except Exception as ex:
      print("Ошибка обучения: {} /n {}".format(ex, traceback.format_exc()))
      break 
  return best_val_loss, best_model
    
        
        



  


  
  
