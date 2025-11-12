import ml_collections


def get_config():
  config = ml_collections.ConfigDict()
  # training
  config.training = training = ml_collections.ConfigDict()
  training.run_name = 'RD'
  training.n_iters = 25000
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


  config.data = data = ml_collections.ConfigDict()
  data.valset_healthy = ""
  data.valset_lesion = ""
  data.dataset = ""
  data.image_size = 224
  data.batch_size = 128
  data.num_channels = 1
  data.workers = 4

  
  
  config.model = model = ml_collections.ConfigDict()
  model.input_shape = (3,224,224)
  model.layers = ['layer1', 'layer2', 'layer3']
  model.normalize = False


  config.optim = optim = ml_collections.ConfigDict()
  optim.lr = 1e-4
  optim.betas = (0.9, 0.999)



  config.seed = 42

  return config
