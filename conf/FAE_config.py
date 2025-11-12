import ml_collections


def get_config():
  config = ml_collections.ConfigDict()
  # training
  config.training = training = ml_collections.ConfigDict()
  training.run_name = 'FAE'
  training.n_iters = 20000
  training.snapshot_freq = 5000
  training.log_freq = 50
  training.early_stop = False
  training.patience = 3
  training.min_delta = 0.1
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
  model.version = 'vanilla'

  if model.version == 'vanilla':
    model.backbone = 'ResNet18'
    model.layers = ['maxpool', 'layer1', 'layer2']
    model.channels = [100,150,200,300]

  elif model.version == 'v2':
    model.backbone = 'EfficientNet-B4'
    model.layers = ['features.1', 'features.2', 'features.3', 'features.4']
    model.channels = (128,192,256,320)
    model.blocks_per_channel = (1,1,1,1)
    model.kernel_size = 5

  model.dropout = 0.1
  model.input_shape = (3,224,224)
  model.normalize = True
  model.featmap_size = 56
  model.window_size = 5


  config.optim = optim = ml_collections.ConfigDict()
  optim.lr = 1e-4



  config.seed = 42

  return config
