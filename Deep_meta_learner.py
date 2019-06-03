import warnings; warnings.simplefilter('ignore')
import os
import copy
import time
import math
import datetime
import numpy as np
import pandas as pd
from pathlib import Path
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import sklearn.metrics as metrics
from sklearn.preprocessing import MinMaxScaler
# Core features is the most important feature, which in this case the measurement value of the target station.
class Core(torch.nn.Module): #output network
  def __init__(self, core_features, n_output, state, **_):
    super(Core, self).__init__()
    self.core_features = core_features
    self.n_output = n_output
    self.input_core = nn.Linear(self.core_features, 128)
    
    
    self.FC_merge = nn.Linear(128*5, (128*5)) 
    
    
    
    self.dropout_high = nn.Dropout(0.4)
    self.dropout_normal = nn.Dropout(0.2)
    self.dropout_low = nn.Dropout(0.1)
    self.state = state #1 hour ahead,3 hours ahead, 6 hours ahead, 12 hours ahead, 24 hours ahead, 36 hours ahead, 48 hours ahead

    self.FC_6_hour = nn.Linear(128*5,self.n_output)
    self.FC_12_hour = nn.Linear(128*5,self.n_output)
    self.FC_24_hour = nn.Linear(128*5,self.n_output)
    self.FC_36_hour = nn.Linear(128*5,self.n_output)
    self.FC_48_hour = nn.Linear(128*5,self.n_output)
    self.FC_merge_info = nn.Linear(self.n_output*5,self.n_output)
    return 
  def forward(self, x_core, x_generated, x_time, x_weather, x_stations):
    x_core = F.relu(((self.input_core(x_core))))
    #x_generated = F.relu(self.FC3_generated(x_generated)) 
    x = (F.relu(self.dropout_normal(self.FC_merge(torch.cat([x_core, x_stations,x_weather, x_time, x_generated], dim = 1)))))
    
    if self.state == 6:
        x = self.FC_6_hour(x)
    if self.state == 12:
        x = self.FC_12_hour(x)
    if self.state == 24:
        x = self.FC_24_hour(x)
    if self.state == 36:
        x = self.FC_36_hour(x)
    if self.state == 48:
        x = self.FC_48_hour(x)
    if self.state == 0:   
        x1 = self.FC_6_hour(x)
        x2 = self.FC_12_hour(x)
        x3 = self.FC_24_hour(x)
        x4 = self.FC_36_hour(x)
        x5 = self.FC_48_hour(x)
        x = self.FC_merge_info(torch.cat([x1, x2,x3, x4, x5], dim = 1))
    return x
class Base(torch.nn.Module):
  def __init__(self, n_feature, n_hidden, n_output, n_hidden_layers, core_features, dropout, **_):
    super(Base, self).__init__()
    #this network is going to freeze                      
    self.n_hidden_layers = n_hidden_layers
    #self.conv1d = nn.Conv1d[10:71]
    
    self.core_features = core_features
    self.station_features = 18
    self.weather_features = 6
    self.time_features = 6
    self.generated_features = n_feature-18-6-6
    
    
    
    self.input_station= nn.Linear(self.station_features, 128)
    self.input_weather= nn.Linear(self.weather_features, 128)
    self.input_time = nn.Linear(self.time_features, 128)
    self.input_generated = nn.Linear(self.generated_features, 128)
    #self.encoder = nn.Linear(n_feature, 64)
    #core feature:
    self.FC1_core = nn.Linear(128, 128 )
    #self.FC2_core = nn.Linear(128, 128)
    
    #station feature:
    self.FC1_station = nn.Linear(128, 128)
    self.FC2_station = nn.Linear(128, 128)
    
    #weather feature:
    self.FC1_weather = nn.Linear(128, 128)
    self.FC2_weather = nn.Linear(128, 128)
    
    #time feature:
    self.FC1_time = nn.Linear(128, 128)
    self.FC2_time = nn.Linear(128, 128)    
    
    #generated_feature:
    self.FC1_generated = nn.Linear(128, 128)
    self.FC2_generated = nn.Linear(128, 128)
    self.FC3_generated = nn.Linear(128, 128)
    #merged
    self.FC3_generated = nn.Linear(128, 128)
    #misc
    self.bn = nn.BatchNorm1d(num_features=128)
    self.pool = nn.MaxPool1d(2, stride = 1)
    self.dropout_high = nn.Dropout(0.4)
    self.dropout_normal = nn.Dropout(0.2)
    self.dropout_low = nn.Dropout(0.1)
    
    self.apply(self.init_weights)
    #output
    self.sigmoid = nn.Sigmoid()
    #self.output = nn.Linear(128*5, n_output)
  def forward(self, x): #messy ....
    temp = torch.split(x, [self.core_features, self.station_features-self.core_features, self.weather_features, self.time_features, self.generated_features ], dim=1)  # split the input
    x_core = temp[0]
    x_stations = F.relu(self.dropout_low((self.input_station(torch.cat([temp[0], temp[1]], dim = 1)))))
    x_weather = F.relu(self.dropout_low((self.input_weather(temp[2]))))
    x_time = F.relu(self.dropout_low((self.input_time(temp[3]))))
    x_generated = F.relu(self.dropout_low((self.input_generated(temp[4]))))
    
    #first layer
    #x_core = F.relu(self.FC1_station(x_core))
    x_stations = F.relu(self.dropout_normal(self.FC1_station(x_stations)))
    x_weather = (F.relu(self.dropout_normal(self.FC1_weather(x_weather))))
    x_time = F.relu(self.dropout_normal(self.FC1_time(self.FC1_time(x_time))))
    x_generated = (F.relu(self.dropout_high(F.relu(self.FC1_generated(x_generated)))))
    
    
    #second layer
    #x_core = F.relu(((self.FC1_station(x_core))))
    x_stations = F.relu((self.dropout_low(self.FC2_station(x_stations))))
    x_weather = F.relu(self.dropout_low(self.FC2_weather(x_weather)))
    x_time = F.relu(self.dropout_low(self.FC1_time(x_time)))
    x_generated = F.relu(self.dropout_high(self.FC2_generated(x_generated)))   
    #fourth_layer:
    x_generated = F.relu(self.dropout_high(self.FC3_generated(x_generated)))  

   
    
   # x_output = self.sigmoid(self.output(x_output))
    return x_core, x_generated, x_time, x_weather, x_stations
  def init_weights(self, model):
    if type(model) == nn.Linear:
      nn.init.uniform_(model.weight, 0, 0.001)

  def apply_dropout(self):
    def apply_drops(m):
      if type(m) == nn.Dropout:
        m.train()
    self.apply(apply_drops)

## Class to computes and stores the average and current value
def load_checkpoint(model, optimizer, Path):
    checkpoint = torch.load(Path)
    model.load_state_dict(checkpoint['model_state_dict'])
    #model2.load_state_dict(checkpoint['model_state_dict'])
    #FC.load_state_dict(checkpoint['FC_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    model.eval()
    #FC.eval()
def save_checkpoint(model, optimizer, Path):
        torch.save({
            'model_state_dict': model.state_dict(),
            #'model2_state_dict': model2.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
            }, Path)        

class AverageMeter(object):
  def __init__(self):
    self.reset()

  def reset(self):
    self.val = 0
    self.avg = 0
    self.sum = 0
    self.count = 0

  def update(self, val, n=1):
    self.val = val
    self.sum += val*n
    self.count += n
    self.avg = self.sum/self.count

## Dataset
class Dataset(object):
  def __init__(self, X, y):
    assert len(X) == len(y)
    self.X = X
    self.y = y

  def __len__(self):
    return len(self.X)

  def __getitem__(self, index):
    x = torch.Tensor(self.X[index])
    y = torch.Tensor(self.y[index])
    return x, y

## Getter for dataloaders for all datasets
def get_datasets(data, batch_size, shuffle, num_workers=0, y_scaler=None, X_scaler=None, **_):
  # Prepare for predictions with single loader
  if X_scaler and y_scaler:
    X_test = X_scaler.transform(data['X_test'])
    y_test = y_scaler.transform(data['y_test'])
    pred_generator = DataLoader(Dataset(X_test, y_test), shuffle=False, batch_size=batch_size, num_workers=num_workers)
    return None, None, None, pred_generator, None, y_scaler

  # Prepare for training with all loaders
  X_scaler = MinMaxScaler(feature_range=(0, 5))
  y_scaler = MinMaxScaler(feature_range=(0, 5))
  
  X_train = X_scaler.fit_transform(data['X_train'])
  y_train = y_scaler.fit_transform(data['y_train'])
  
  X_val = X_scaler.transform(data['X_val'])
  y_val = y_scaler.transform(data['y_val'])
  
  X_test = X_scaler.transform(data['X_test'])
  y_test = y_scaler.transform(data['y_test'])

  training_generator = DataLoader(Dataset(X_train, y_train), shuffle=shuffle, batch_size=batch_size, num_workers=num_workers)
  validation_generator = DataLoader(Dataset(X_val, y_val), shuffle=shuffle, batch_size=batch_size, num_workers=num_workers)
  test_generator = DataLoader(Dataset(X_test, y_test), shuffle=shuffle, batch_size=batch_size, num_workers=num_workers)
  pred_generator = DataLoader(Dataset(X_test, y_test), shuffle=False, batch_size=batch_size, num_workers=num_workers)

  return training_generator, validation_generator, test_generator, pred_generator, X_scaler, y_scaler


filename = Path('./Deep_meta_learner.joblib')
filename2 = Path('./Deep_meta_learner2.joblib')
## Train model
def train(config, data):
  
  if (os.path.exists(filename)) and config['load'] == False:
      return
    

  params = {
    'shuffle': True,
    'num_workers': 4,
    'core_features': config['core_features'],
    'n_feature':len(data['X_train'].columns),
    'n_output': len(data['y_train'].columns),
    
    'n_hidden_layers': 6,
    'batch_size': 30,
    'n_hidden': 512,
    'learning_rate': 1e-4,
    'epochs': 40,
    'dropout': 0.0,
    'log_nth': 1,
    'mode': 'train',
    'state': config['state'],
  }

  
  # Activate gpu optimization
  device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
  if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True

  torch.manual_seed(42)
  
  model = Base(**params).to(device)
  core_model = Core(**params).to(device) 
  optimizer_parameter = (list(model.parameters()) + list(core_model.parameters()))
  optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, optimizer_parameter), lr=params['learning_rate'])
  if config['load'] == True:
      load_checkpoint(model,optimizer, "model_train")  
  if config['load_state'] == True:
      load_checkpoint(core_model, optimizer, "model_train2") 
      optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, optimizer_parameter), lr=params['learning_rate'])
      if config['state'] >= 6:
          core_model.FC_6_hour.weight.requires_grad = False
          core_model.FC_6_hour.bias.requires_grad = False   
          if config['state'] >= 12:
              core_model.FC_12_hour.weight.requires_grad = False
              core_model.FC_12_hour.bias.requires_grad = False   
              if config['state'] >= 24:
                  core_model.FC_24_hour.weight.requires_grad = False
                  core_model.FC_24_hour.bias.requires_grad = False 
                  if config['state'] >= 36:
                      core_model.FC_36_hour.weight.requires_grad = False
                      core_model.FC_36_hour.bias.requires_grad = False 
                      if config['state'] >= 48:
                          core_model.FC_48_hour.weight.requires_grad = False
                          core_model.FC_48_hour.bias.requires_grad = False 
  # free layers
  if config['freeze1'] == True: # freeze layer 1 (never freeze core)
      model.input_station.weight.requires_grad = False
      model.input_station.bias.requires_grad = False   
      model.input_weather.weight.requires_grad = False
      model.input_weather.bias.requires_grad = False       
      model.input_time.weight.requires_grad = False
      model.input_time.bias.requires_grad = False       
      model.input_generated.weight.requires_grad = False
      model.input_generated.bias.requires_grad = False       
  if config['freeze2'] == True:
      model.FC1_station.weight.requires_grad = False
      model.FC1_station.bias.requires_grad = False   
      model.FC1_weather.weight.requires_grad = False
      model.FC1_weather.bias.requires_grad = False       
      model.FC1_time.weight.requires_grad = False
      model.FC1_time.bias.requires_grad = False       
      model.FC1_generated.weight.requires_grad = False
      model.FC1_generated.bias.requires_grad = False 
  if config['freeze3'] == True:
      model.FC2_station.weight.requires_grad = False
      model.FC2_station.bias.requires_grad = False   
      model.FC2_weather.weight.requires_grad = False
      model.FC2_weather.bias.requires_grad = False       
      model.FC2_time.weight.requires_grad = False
      model.FC2_time.bias.requires_grad = False       
      model.FC2_generated.weight.requires_grad = False
      model.FC2_generated.bias.requires_grad = False 
  if config['freeze4'] == True:    
      model.FC3_generated.weight.requires_grad = False
      model.FC3_generated.bias.requires_grad = False 
  

  criterion = nn.L1Loss(reduction='sum')

  model_dict, val_score = fit(data, model, core_model, device, params, config, optimizer, criterion)
  torch.save(model_dict, filename)
  save_checkpoint(model, optimizer, "model_train") 
  save_checkpoint(core_model, optimizer, "model_train2") 
## Prediction
def predict(config, data):

  device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
  if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
  
  state = torch.load(filename)
  params = state['params']
  params['mode'] = 'predict'
  params['X_scaler'] = state['X_scaler']
  params['y_scaler'] = state['y_scaler']
  
  model = Base(**params).to(device)
  core_model = Core( **params).to(device)
  model.load_state_dict(state['model_dict'])
  core_model.load_state_dict(state['model2_dict'])

                              
  pred_np, score = fit(data, model, core_model, device, params, config)
  pred_df = pd.DataFrame(index=data['X_test'].index)
  temp1 = data['y_test'].iloc[::config['window'], :].values.flatten()
  temp2 = pred_np[::config['window'], :].flatten()
  pred_df['MLP'] = temp2[:len(pred_np)]
  pred_df['True'] = temp1[:len(pred_np)]
  rmse = math.sqrt(metrics.mean_squared_error(pred_df['True'], pred_df['MLP']))
  r2 = metrics.r2_score(pred_df['True'], pred_df['MLP'])
  return pred_df, rmse, r2
#Will stacking work ?
def predict_for_train(config, data):

  device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
  if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
  
  state = torch.load(filename)
  params = state['params']
  params['mode'] = 'predict'
  params['X_scaler'] = state['X_scaler']
  params['y_scaler'] = state['y_scaler']
  
  model = Base(**params).to(device)
  core_model = Core(**params).to(device)
  model.load_state_dict(state['model_dict'])
  core_model.load_state_dict(state['model2_dict'])

  pred_np, score = fit(data, model, core_model, device, params, config)

  return pred_np
## Pytorch Pipe 🐥
def fit(data, model, core_model, device, params, config, optimizer=None, criterion=None):
  training_generator, validation_generator, test_generator, pred_generator, X_scaler, y_scaler = get_datasets(data, **params)

  ## Run single training batch with backprop {loss}
  def runBatches(generator):
    losses = AverageMeter()
    
    for i, (X, y) in enumerate(generator):
      X, y = Variable(X, requires_grad=True).to(device), Variable(y).to(device)
            
      x_core, x_generated, x_time, x_weather, x_stations = model(X)
      output = core_model(x_core, x_generated, x_time, x_weather, x_stations)
                         
      loss = criterion(output, y)
      optimizer.zero_grad()
      loss.backward()
      # nn.utils.clip_grad_norm_(model.parameters(), params['clip'])
      optimizer.step()
      losses.update(loss.item())

    return losses.avg
  
  ## Run single prediction batch {y_true, y_pred}
  def predict(generator):
    model.eval()
    model.apply_dropout()
    y_trues = []
    y_preds = []
    
    for i, (X, y) in enumerate(generator):
      X, y = X.to(device), y.to(device)
      x_core, x_generated, x_time, x_weather, x_stations = model.forward(X)
      output = core_model(x_core, x_generated, x_time, x_weather, x_stations)
      y_trues = np.append(y_trues, y.cpu().numpy())
      y_preds = np.append(y_preds, output.detach().cpu().numpy())

    return np.array(y_trues), np.array(y_preds)

  ## Do Training
  if params['mode'] == 'train':
    start_time = datetime.datetime.now()
    train_scores = []
    val_scores = []
    
    best_model_dict = copy.deepcopy(model.state_dict())
    best_predict_dict = copy.deepcopy(core_model.state_dict())
    best_score = 999
    for epoch in range(params['epochs']):

      # Training
      model.train()
      train_score = runBatches(generator=training_generator)
      train_scores.append(train_score)

      # Validation
      model.eval()
      val_score = runBatches(generator=validation_generator)
      val_scores.append(val_score)

      # Keep the best model
      if val_score < best_score :
        best_score = val_score
        best_model_dict = copy.deepcopy(model.state_dict())
        best_predict_dict = copy.deepcopy(core_model.state_dict())

      time = (datetime.datetime.now() - start_time).total_seconds()
      
      if not epoch%params['log_nth']:
        print('e {e:<3} time: {t:<4.0f} train: {ts:<4.2f} val: {vs:<4.2f}'.format(e=epoch, t=time, ts=train_score, vs=val_score))
    
    # Test the trained model
    test_score = runBatches(generator=test_generator)
    trues, preds = predict(generator=pred_generator)

    # Return results, model and params for saving
    result_dict = {
      'model_dict': best_model_dict,
      'model2_dict': best_predict_dict,
      'params': params,
      'train_scores': train_scores,
      'val_scores': val_scores,
      'X_scaler': X_scaler,
      'y_scaler': y_scaler,
    }
    return result_dict, best_score

  ## Do Predictions
  if params['mode'] == 'predict':
    trues, preds = predict(generator=pred_generator)
    score = math.sqrt(metrics.mean_squared_error(trues, preds))
    return y_scaler.inverse_transform(preds.reshape(-1, config['window'])), score

