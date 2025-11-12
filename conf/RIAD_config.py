import ml_collections


def get_config():
  config = ml_collections.ConfigDict()
  # training
  config.training = training = ml_collections.ConfigDict()
  training.run_name = 'RIAD'
  training.n_iters = 100000
  training.snapshot_freq = 5000
  training.log_freq = 50
  training.early_stop = False
  training.patience = 3
  training.min_delta = 0.1
  training.early_stop_model = 99  # just the number for saving the early stop model
  assert training.early_stop_model > training.n_iters // training.snapshot_freq
  training.alpha = 1
  training.beta = 1
  training.gamma = 1
  training.k = [8,16,28,32]  # size rectangular regions
  training.n = 5  # number disjoint subsets
  training.num_up = 3 # number of images to upload


  config.accelerator = accelerator = ml_collections.ConfigDict()
  accelerator.train_continue = False
  accelerator.checkpoint = 0


  config.data = data = ml_collections.ConfigDict()
  data.valset_healthy = ""
  data.valset_lesion = ""
  data.dataset = ""
  data.image_size = 224
  data.batch_size = 16
  data.workers = 4
  data.num_channels = 1


  config.model = model = ml_collections.ConfigDict()
  model.nonlinearity = 'swish'
  model.nf = 32
  model.ch_mult = (1, 1, 1, 2, 3, 4)
  model.num_res_blocks = (1, 1, 1, 2, 3, 2)
  model.attn_resolutions = (14,7)
  model.attn_emb_dim = 32
  model.rescale = False
  model.embedding_type = None
  model.dropout = 0.1
  model.dropout_at = 14


  config.optim = optim = ml_collections.ConfigDict()
  optim.lr = 5e-6  #5e-5 T2


  config.seed = 42

  return config
