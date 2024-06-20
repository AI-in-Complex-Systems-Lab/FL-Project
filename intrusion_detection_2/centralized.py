import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical

from fed_learn import*

from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler


scaler = StandardScaler()

X_train = scaler.fit_transform(train.drop(['y'], axis=1).to_numpy())
y_train = train['y'].to_numpy()
X_test = scaler.transform(test.drop(['y'], axis=1).to_numpy())
y_test = test['y'].to_numpy()


y_cat_train = to_categorical(y_train)
y_cat_test = to_categorical(y_test)

model = tf.keras.models.Sequential([
    tf.keras.layers.InputLayer(input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(units=50, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(units=y_cat_test.shape[1], activation='softmax')
])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=20)

model.fit(
    x=X_train, 
    y=y_cat_train, 
    epochs=300,
    validation_data=(X_test, y_cat_test),
    verbose=1,
    batch_size=64,
    callbacks=[early_stop]
)

model_loss = pd.DataFrame(model.history.history)
model_loss[['loss','val_loss']].plot()

plt.show()

print("Loss, Accuracy: ", model.evaluate(X_test, y_cat_test))
print("F1-score: ", f1_score(y_test, np.argmax(model.predict(X_test), axis=1), average='weighted'))