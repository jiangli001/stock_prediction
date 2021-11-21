import os
import sys
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
import tensorflow as tf
from tensorflow.keras import models
from tensorflow.keras import models
sys.path.append("..")
sys.path.append("../..")
from stock_prediction.util.embedding import Embedding

@tf.function
def mse(y, pred):
    return tf.reduce_mean(tf.square(tf.subtract(pred, y)))
    
@tf.function
def rmse(y, pred):
    return tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(pred, y))))

class RNN(object):
    def __init__(self, name, stock_id, input_strategy=1, training_model="LSTM", stateful=True,
                 return_sequences=True, batch_size=1, input_size=33,
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
            batch_size (int, optional): Defaults to 1.
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
        self.batch_size = batch_size
        
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
        
    def _get_data(self, stage=["train"], union=[]):
        """根据 self.input_strategy 获得数据

        Args:
            stage (list, optional): ["train", "valid", "test"]，想要获得的数据。可返回多个.
                example1:
                    X_train, y_train = self._get_dat(["train"])
                    X_valid, y_valid = self._get_dat(["valid"])
                    X_test, y_test = self._get_dat(["test"])
                example2:
                    X_train, y_train, X_valid, y_valid = self._get_dat(["train", "valid"])
                    X_valid, y_valid, X_test, y_test = self._get_dat(["valid", "test"])
                Defaults to ["train"].
            union (list, optional): 需要合并的数据集,例如["train", "valid"],则返回的x,y会将这两份数据集合并.
                example1:
                    X_train, y_train, X_valid, y_valid = self._get_dat(["train", "valid"])
                    X, y = self._get_dat(stage=["train", "valild"], union=["train", "valid"])
                example2:
                    X, y, X_test, y_test = self._get_dat(stage=["train", "valild", "test"], union=["train", "valid"])
                Defaults to [].
        """
        assert (len(stage) >= len(union)),\
            "the length of `union` must be smaller than or equal to that of `stage`"
            
        train_data, valid_data, test_data = None, None, None
        if "train" in stage: train_data = pd.read_csv(self.train_path)
        if "valid" in stage: valid_data = pd.read_csv(self.valid_path)
        if "test" in stage: test_data = pd.read_csv(self.test_path)
        
        res = []
        data_dict = {"train": train_data, "valid": valid_data, "test": test_data}
        
        for _s in stage:
            data = data_dict[_s]     
            X, y = data.iloc[:, 4:-1].values, data.iloc[:, -1].values
            if self.input_strategy == 1:
                X = tf.cast(tf.expand_dims(X, axis=0), tf.float32, name="x_cast") # [1, samlples, 33]
                X = tf.nn.l2_normalize(X, axis=-1)
                if len(tf.shape(y)) == 1:
                    y = tf.multiply(tf.cast(tf.reshape(y, [1, -1, 1]), tf.float32, name="y_cast"), 100)  # [1, samples, 1]
                res.append(X)
                res.append(y)
            elif self.input_strategy == 2:
                return 
        
            elif self.input_strategy == 3:
                X = tf.cast(tf.expand_dims(X, axis=0), tf.float32, name="x_cast") # [1, samlples, 33]
                y = tf.multiply(tf.cast(y, tf.float32, name="y_cast"), 1)  # [samples, 1]
                res.append(X)
                res.append(y)
        return res  

    @tf.function
    def _printbar(self):
        today_ts = tf.timestamp()%(24*60*60)

        hour = tf.cast(today_ts//3600+8,tf.int32)%tf.constant(24)
        minite = tf.cast((today_ts%3600)//60,tf.int32)
        second = tf.cast(tf.floor(today_ts%60),tf.int32)
        return (hour, minite, second)

    def _build(self):
        x_input = tf.keras.Input(shape=(None, self.input_size), batch_size=1, dtype=tf.float32) # 固定只有 1 个batch_size
        x = x_input
        for i in tf.range(self.rnn_layers_num-1):
            x = self.rnn(units=self.rnn_output_size[i], activation=self.rnn_activation, return_sequences=True, stateful=self.stateful)(x)
        x = self.rnn(units=self.rnn_output_size[-1], activation=self.rnn_activation, return_sequences=self.return_sequences, stateful=self.stateful)(x)
        if self.input_strategy == 1 and self.rnn_output_size[-1] != 1:
            x = self.rnn(units=1, activation=None, return_sequences=self.return_sequences, stateful=self.stateful)(x)
        
        if self.input_strategy == 3:
            x = tf.reshape(x, [-1, self.rnn_output_size[-1]])
            for i in tf.range(self.dense_layers_num-1):
                x = tf.keras.layers.Dense(self.dense_output_size[i], activation=self.dense_activation)
        
        self.model = models.Model(inputs=[x_input], outputs=x, name=self.name)
        self.model.summary()
        
    @tf.function
    def _train_step(self, X, y):
        with tf.GradientTape() as tape:
            predictions = self.model(X, training=True)
            loss = self.loss(y, predictions)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        # 梯度裁剪
        for i, gradient in enumerate(gradients):
            gradients[i] = tf.clip_by_value(gradient, 1e-6, 1.0-1e-6)
            
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
        train_X, train_y, valid_X, valid_y = self._get_data(stage=["train", "valid"])
        for i in tf.range(epochs):
            self._train_step(train_X, train_y)
            self._valid_step(valid_X, valid_y)
            logs = '{}:{}:{}: Epoch={}, Loss:{}, Valid Loss:{}'
            if i%show_epoch ==0:
                hour, minite, second = self._printbar()
                tf.print(tf.strings.format(logs,
                                           (hour, minite, second, i, self.train_loss.result(), self.valid_loss.result())))
                tf.print("")
                
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
        
    def predict(self, stage="test", save=False):
        x, y = self._get_data([stage])
        pred = self.model.predict(x)
        print(tf.concat([y/100, pred/100], axis=-1))
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
                return_sequences=True, batch_size=1, input_size=33,
                rnn_output_size=[16, 8, 4], rnn_activation="sigmoid", dense_output_size=[], dense_activation="",
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), loss="mse",
                sample_path="/root/sampleTest", patience=5, reload=True
                )
    model.fit(epochs=100)
    model.predict("train")
