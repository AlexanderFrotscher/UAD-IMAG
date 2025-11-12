import ml_collections


def get_config():
  config = ml_collections.ConfigDict()
  # eval
  config.eval = eval = ml_collections.ConfigDict()
  eval.run_name = 'FAE'
  eval.output = "/results/"
  eval.output_mf = "/results/"
  eval.split_size = 310
  eval.thr_start = 0.35
  eval.thr_end = 0.7
  eval.thr_step = 0.01
  eval.fpr = [0.00267, 0.01] #T2 [0.003, 0.01] #T1
  eval.calc_thr = False
  eval.thr_fpr = [0.51, 0.471]  #[0.51, 0.471] T2  #[0.58, 0.552] T1
  eval.opt_thr = 0.51   #0.58 T1
  eval.data_is_healthy = False
  if eval.calc_thr:
    assert eval.data_is_healthy
  eval.slice = False
  eval.distribution = False


  config.accelerator = accelerator = ml_collections.ConfigDict()
  accelerator.checkpoint = 8   #T2w  # 4 for T1w


  config.data = data = ml_collections.ConfigDict()
  data.dataset = ""
  data.image_size = 224
  data.batch_size = 2
  data.num_channels = 1
  data.workers = 1
  data.data_type = 'T2w'


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
  model.window_size = 11


  config.postprocessing = postprocessing = ml_collections.ConfigDict()
  postprocessing.kernel_size = 4


  config.seed = 42

  return config
