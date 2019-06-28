import tensorflow as tf
print(tf.__version__)

from tensorflow.keras.metrics import Metric
import tensorflow.keras.backend as K

from sklearn.metrics import f1_score

class F1_Score(Metric):
    def __init__(self, name='f1_score', **kwargs):
        super(F1_Score, self).__init__(name=name, **kwargs)
        self.f1_score = self.add_weight(name='f1', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = K.eval(y_true)
        y_pred = K.eval(y_pred)

        values = f1_score(y_true, y_pred)

        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, 'float32')
            values = tf.multiply(values, sample_weight)
        self.f1_score.assign(values)

    def result(self):
        return self.f1_score

    def reset_states(self):
        self.f1_score.assign(0.)
