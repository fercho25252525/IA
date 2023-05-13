# pip install tensorflow
import tensorflow as tf

# Definir los datos de entrada y salida
inputs = tf.keras.Input(shape=(4,))
outputs = tf.keras.layers.Dense(3, activation='softmax')(inputs)

# Crear el modelo
model = tf.keras.Model(inputs=inputs, outputs=outputs)

# Compilar el modelo
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Entrenar el modelo
x_train = [[0, 0, 1, 0], [1, 0, 1, 1], [1, 1, 0, 1], [0, 1, 1, 0]]
y_train = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0]]
model.fit(x_train, y_train, epochs=10)

# Hacer una predicción con el modelo entrenado
x_test = [[1, 0, 0, 1], [0, 1, 0, 0]]
y_pred = model.predict(x_test)
print(y_pred)