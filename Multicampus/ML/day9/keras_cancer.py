import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from sklearn.preprocessing import StandardScaler
import numpy as np

# breast cancer 데이터를 가져온다.
data = load_breast_cancer()

feature = data['data']
target = data['target']

# Z-score normalization
scaler = StandardScaler()
z_cancer = scaler.fit_transform(feature)

x_train, x_test, y_train, y_test = train_test_split(z_cancer, target, test_size=0.2)

x_train.shape, y_train.shape

x_input = Input(batch_shape = (None, 30))
h_layer = Dense(16, activation='relu')(x_input)
y_output = Dense(1, activation='sigmoid')(h_layer)
model = Model(x_input, y_output)
model.compile(loss='binary_crossentropy', optimizer='adam')
model.summary()

hist = model.fit(x_train, y_train, epochs=500, validation_data = (x_test, y_test))

plt.plot(hist.history['loss'], label='train')
plt.plot(hist.history['val_loss'], label='test')
plt.legend()
plt.show()

y_pred = model.predict(x_test).reshape(-1)
y_pred = np.where(y_pred > 0.5, 1, 0)

acc = (y_pred == y_test).mean()
print(acc)