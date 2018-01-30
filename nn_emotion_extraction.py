from adlframework.retrievals.BlobLocalCache import BlobLocalCache
from adlframework.datasource import DataSource
from adlframework.dataentity.arff_de import ARFFDataEntity
from adlframework.experiment import SimpleExperiment
from keras.losses import KLD
from keras.optimizers import Adadelta
from nets import dense_network
import pdb
import numpy as np
from keras.callbacks import ModelCheckpoint

### Custom Controllers/Filters
def remove_wavs(entity):
	'''
	Filter out wav and extraneous files
	'''
	return entity.unique_id[-4:] == 'arff'

def split_to_features(sample):
	data, label = sample
	data.next() # Remove empty row
	data = np.array([x for x in data.next()][1:])
	return data, label

### Metadata
cache_path = 'local_cache/Segmented_AVEC/'
train_retrieval = BlobLocalCache(cache_path+'wav/train', cache_path+'labels/train')
val_retrieval = BlobLocalCache(cache_path+'wav/dev', cache_path+'labels/dev')

controllers = [split_to_features]
prefilters = [remove_wavs]

### Data Sources
train_ds = DataSource(train_retrieval, ARFFDataEntity,
						ignore_cache=True,
						batch_size=2,
						verbosity=3,
						controllers=controllers,
						prefilters=prefilters)

val_ds = DataSource(val_retrieval, ARFFDataEntity,
						ignore_cache=True,
						batch_size=2,
						verbosity=3,
						controllers=controllers,
						prefilters=prefilters)

pdb.set_trace()

### Callbacks
callbacks = [ModelCheckpoint('weights.{epoch:02d}-{val_loss:.2f}.hdf5')]


### Network
net = dense_network(input_shape=(989, ),
					target_shape=(3))

### Create and run Experiment
exp = SimpleExperiment(train_datasource=train_ds,
						validation_datasource=val_ds,
						loss=KLD,
						optimizer=Adadelta(),
						metrics=['mae'],
						network = net,
						callbacks=callbacks)

exp.run()