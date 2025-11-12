import ml_collections


def get_config():
  config = ml_collections.ConfigDict()
  # training
  config.training = training = ml_collections.ConfigDict()
  training.run_name = 'UniAD'
  training.n_iters = 30000
  training.snapshot_freq = 5000
  training.log_freq = 50
  training.early_stop = False
  training.patience = 3
  training.min_delta = 4.0
  training.early_stop_model = 99  # just the number for saving the early stop model
  assert training.early_stop_model > training.n_iters // training.snapshot_freq
  


  config.accelerator = accelerator = ml_collections.ConfigDict()
  accelerator.train_continue = False
  accelerator.checkpoint = 0
  accelerator.clip_grad_norm = 0.1


  config.data = data = ml_collections.ConfigDict()
  data.valset_healthy = ""
  data.valset_lesion = ""
  data.dataset = ""
  data.image_size = 224
  data.batch_size = 128
  data.num_channels = 1
  data.workers = 4

  
  
  config.model = model = ml_collections.ConfigDict()
  model.backbone = 'EfficientNet-B4'
  model.layers =  ['features.1','features.2','features.3','features.4']
  model.normalize = False
  model.input_shape = (3,224,224)
  model.feature_size = 14
  model.pos_embed_type = 'learned'
  model.hidden_dim = 256
  model.nhead = 8 
  model.num_encoder_layers = 4 
  model.num_decoder_layers = 4 
  model.dim_feedforward = 1024  
  model.dropout = 0.1
  model.activation = 'relu'
  model.normalize_before = False
  model.feature_jitter = True
  model.feature_jitter_scale = 5.0
  model.feature_jitter_prob = 1.0
  model.neighbor_size = [7,7]  
  model.neighbor_mask = [True, True, True]
  


  config.optim = optim = ml_collections.ConfigDict()
  optim.lr = 1e-4
  optim.scheduler_step = 10



  config.seed = 42

  return config
