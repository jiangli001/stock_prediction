import os
import sys
import numpy as np
import pandas as pd
import datetime
import warnings
warnings.filterwarnings("ignore")
import tensorflow as tf
from tensorflow.keras import models
from tensorflow.keras import models
sys.path.append("..")
sys.path.append("../..")
from stock_prediction.util.getSampleData import get_data
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

@tf.function
def mse(y, pred):
    return tf.reduce_mean(tf.square(tf.subtract(pred, y)))
    
@tf.function
def rmse(y, pred):
    return tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(pred, y))))

class RNN(object):
    def __init__(self, name, stock_id, input_strategy=1, training_model="LSTM", stateful=True,
                 return_sequences=True, input_size=33,
                 rnn_output_size=[], rnn_activation="tanh", dense_output_size=[], dense_activation="",
                 optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss="rmse",
                 sample_path="/root/sampleTest", seed=2021,
                 patience=0, reload=True
                 ):
        """
        Args:
            name ([string]): 名字，自定义
            stock_id ([string]): 股票id
            input_strategy (int, optional):
                values:
                    1. 简单拼接： 每一条样的所有特征作为一个向量，里面每个元素对应一个特征。
                    2. embedding： 每一个连续值进行分桶操作（取log），将连续值变成离散值，之后做embedding操作，最后进行concat
                    3. 添加 mlp 层...
                Defaults to 1
            training_model (str, optional):
                values:
                    1. RNN: 简单RNN
                    2. LSTM
                    3. GRU
                Defaults to LSTM
            stateful (bool, optional): 是否有状态. Defaults to True.
            return_sequences (bool, optional): 是否返回序列. Defaults to True.
            input_size (int, optional): [description]. Defaults to 33.
            rnn_output_size (list, optional): rnn 隐藏层 units 大小，list的长度代表有多少层rnn. Defaults to [].
            rnn_activation (str, optional): 激活函数, "sigmoid", "tahn", "relu"等. Defaults to "tanh".
            dense_output_size (list, optional): 同 rnn_activation，仅当 input_strategy=3时生效. Defaults to [].
            dense_activation (str, optional): 同rnn_activation. Defaults to "".
            loss (str, optional): "rmse" or "mse". Defaults to "rmse"
            sample_path (str, optional): 数据路径. Defaults to "/root/sampleTest".
            seed (int, optional): 随机种子. Defaults to 2021.
            patience (int, optional): early_stop 可忍受的 epoch 数量，若为0，代表不使用 early_stop. Defaults to 0.
            reload (bool, optional): 是否读取最新模型. Defaults to True.
        """
        super(RNN, self).__init__()
        self.name=name
        tf.random.set_seed(seed)
        if isinstance(stock_id, int):
            stock_id = str(stock_id)
        self.stock_id = stock_id
        
        assert input_strategy in [1, 2, 3], \
            "input_strategy should be 1 for simple concat, or 2 for embedding, or 3"
        self.input_strategy = input_strategy
        
        assert training_model in ["RNN", "LSTM", "GRU"], \
            "training_model should be \"RNN\" or \"LSTM\" or \"GRU\"."
        self.training_model = training_model
        if self.training_model == "RNN":
            self.rnn = tf.keras.layers.SimpleRNN
        elif self.training_model == "LSTM":
            self.rnn = tf.keras.layers.LSTM
        elif self.training_model == "GRU":
            self.rnn = tf.keras.layers.GRU
        
        self.stateful = stateful
        self.return_sequences = return_sequences
        
        self.input_size = input_size
        
        assert len(rnn_output_size) >= 1,\
            "rnn output size should be a non empty list"
        self.rnn_output_size = rnn_output_size
        self.rnn_layers_num = len(rnn_output_size)
        
        if input_strategy==3:
            assert len(dense_output_size)>=1,\
                "the `dense_output_size` should be a non empty list if `input_strategy`=3"
        if dense_output_size:
            self.dense_output_size = dense_output_size
            self.dense_layers_num = len(dense_output_size)
            self.dense_activation = dense_activation
        
        loss = loss.lower()
        if loss == "mse":
            self.loss = mse
        elif loss == "rmse":
            self.loss = rmse
            
        self.optimizer = optimizer
        self.train_loss = tf.keras.metrics.Mean(name="train_loss")
        self.valid_loss = tf.keras.metrics.Mean(name="valid_loss")
        self.test_loss = tf.keras.metrics.Mean(name="test_loss")
        
        self.sample_path = sample_path
        for dir in ["train", "valid", "test"]:
            assert dir in os.listdir(self.sample_path),\
                "no directory \"{}\" in sample path: {}".format(dir, self.sample_path) 
        self.train_path = os.path.join(self.sample_path, "train", self.stock_id+".csv")
        self.valid_path = os.path.join(self.sample_path, "valid", self.stock_id+".csv")
        self.test_path = os.path.join(self.sample_path, "test", self.stock_id+".csv")
        
        self.rnn_activation = rnn_activation
        
        # early_stop
        self.patience = patience        
        self.min_eval_loss = float("inf")
        
        if "rnn_save_model" not in os.listdir():
            os.mkdir("rnn_save_model")  
        self.model_path = os.path.join("./rnn_save_model", self.stock_id)
        if reload:
            if os.path.exists(self.model_path):
                self.model = tf.keras.models.load_model(self.model_path)
                print("load model from path: {}".format(self.model_path))
            else:
                print("stock {} has no model in path: {}".format(self.stock_id, "./rnn_save_model"))
                print("build the model for {}".format(self.stock_id))
                self._build()
        else:
            self._build()
            
        print("======== self.attributes ==========")
        print(self.__dict__)
        print("===================================")
        
    def _build(self):
        x_input = tf.keras.Input(shape=(None, self.input_size), batch_size=1, dtype=tf.float32) # 固定只有 1 个batch_size
        x = x_input
        
        if self.rnn_output_size[0] > 0:
            for i in tf.range(self.rnn_layers_num):
                x = self.rnn(units=self.rnn_output_size[i], activation=self.rnn_activation, return_sequences=True, stateful=self.stateful)(x)
            # x = self.rnn(units=self.rnn_output_size[-1], activation=self.rnn_activation, return_sequences=self.return_sequences, stateful=self.stateful)(x)
            if self.input_strategy == 1 and self.rnn_output_size[-1] != 1:
                x = self.rnn(units=1, activation=None, return_sequences=self.return_sequences, stateful=self.stateful)(x)
        
        if self.input_strategy == 3:
            if self.rnn_output_size[0] > 0:
                x = tf.reshape(x, [-1, self.rnn_output_size[-1]])
            else:
                x = tf.reshape(x, [-1, self.input_size])
            for i in tf.range(self.dense_layers_num):
                x = tf.keras.layers.Dense(self.dense_output_size[i], activation=self.dense_activation)(x)
            if self.dense_output_size and self.dense_output_size[-1] != 1:
                x = tf.keras.layers.Dense(1, activation=self.dense_activation)(x)
            x = tf.reshape(x, [1, -1, 1])
        self.model = models.Model(inputs=[x_input], outputs=x, name=self.name)
        self.model.summary()
                
    @tf.function
    def _train_step(self, X, y):
        with tf.GradientTape() as tape:
            predictions = self.model(X, training=True)
            loss = self.loss(y, predictions)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        # 梯度裁剪
        
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        
        self.train_loss.update_state(loss)

    @tf.function
    def _valid_step(self, X, y):
        predictions = self.model(X, training=False)
        loss = tf.keras.losses.mean_squared_error(y, predictions)
        loss = self.loss(y, predictions)
        self.valid_loss.update_state(loss)

    def fit(self, epochs=10, show_epoch=1):
        cur_patience = 0
        train_X, train_y = get_data(path=self.train_path, input_strategy=self.input_strategy)
        valid_X, valid_y = get_data(path=self.valid_path, input_strategy=self.input_strategy)
        for i in tf.range(epochs):
            self._train_step(train_X, train_y)
            self._valid_step(valid_X, valid_y)
            cur_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            logs = '{}: Stock_id={}, Epoch={}, Loss:{}, Valid Loss:{}'
            if i%show_epoch ==0:
                tf.print(tf.strings.format(logs,
                                           (cur_time, self.stock_id, i, self.train_loss.result(), self.valid_loss.result())))
            
            # early stop
            if self.patience is not None:
                if self.valid_loss.result() > self.min_eval_loss:
                    cur_patience += 1
                    if cur_patience >= self.patience:
                        tf.print(tf.strings.format("Early Stop at epoch: {}, cur loss is {}, the best loss is {}",
                                 (i, self.valid_loss.result(), self.min_eval_loss)))
                        break
                else:
                    self.min_eval_loss = self.valid_loss.result()
                    cur_patience = 0
                    self.model.save(self.model_path, save_format='tf')
                    
            if self.stateful:
                self.model.reset_states()
        
    def predict(self, save=False):
        x, y = get_data(path=self.test_path, input_strategy=self.input_strategy)
        pred = self.model(x, training=False)
        print(tf.concat([y, pred], axis=-1))
        loss = self.loss(y, pred)
        print(loss)
        if save:
            save_dir = os.path.join(self.sample_path, "test_rnn")
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)
            save_path =  os.path.join(save_dir, self.stock_id + ".csv")
            data = pd.read_csv(self.test_path)
            res = np.squeeze(np.array(pred), axis=0)
            data["GrowthRatePredict"] = res
            data.to_csv(save_path, index=None)
            print("Saving prediction result to path {} !".format(save_path))
        
if __name__ == "__main__":    
    model = RNN(name="RNN", stock_id="000002.SZ", input_strategy=1, training_model="RNN", stateful=True,
                return_sequences=True, input_size=33,
                rnn_output_size=[16, 8, 4], rnn_activation="sigmoid", dense_output_size=[], dense_activation="",
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), loss="mse",
                sample_path="/root/sampleTest", patience=5, reload=True
                )
    model.fit(epochs=100)
    model.predict(False)
