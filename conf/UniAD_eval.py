import ml_collections


def get_config():
  config = ml_collections.ConfigDict()
  # eval
  config.eval = eval = ml_collections.ConfigDict()
  eval.run_name = 'UniAD'
  eval.output = "/results/"
  eval.output_mf = "/results/"
  eval.split_size = 155
  eval.thr_start = 100
  eval.thr_end = 150          
  eval.thr_step = 5     
  eval.fpr = [0.004,0.01] #T2    #[0.01] T1
  eval.calc_thr = False
  eval.thr_fpr = [120, 115.54]  #T2   #[130] T1
  eval.opt_thr = 120 #T2     #130 T1
  eval.data_is_healthy = False
  if eval.calc_thr:
    assert eval.data_is_healthy
  eval.slice = False
  eval.distribution = False


  config.accelerator = accelerator = ml_collections.ConfigDict()
  accelerator.checkpoint = 5


  config.data = data = ml_collections.ConfigDict()
  data.dataset = ""
  data.image_size = 224
  data.batch_size = 1
  data.num_channels = 1
  data.workers = 1
  data.data_type = 'T2w'

  
  
  config.model = model = ml_collections.ConfigDict()
  model.backbone = 'EfficientNet-B4'
  model.layers = ['features.1','features.2','features.3','features.4']
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
  model.feature_jitter = False
  model.feature_jitter_scale = 5.0
  model.feature_jitter_prob = 1.0
  model.neighbor_size = [7,7]
  model.neighbor_mask = [True, True, True]

  config.postprocessing = postprocessing = ml_collections.ConfigDict()
  postprocessing.kernel_size = 4

  
  config.seed = 42

  return config
