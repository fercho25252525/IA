# Instalar y cargar la biblioteca keras
install.packages("keras")
library(keras)

# Definir los datos de entrada y salida
inputs <- keras::layer_input(shape = c(4))
outputs <- inputs %>%
  keras::layer_dense(units = 3, activation = "softmax")

# Crear el modelo
model <- keras::keras_model(inputs = inputs, outputs = outputs)

# Compilar el modelo
model %>% keras::compile(
  loss = "categorical_crossentropy",
  optimizer = "adam",
  metrics = c("accuracy")
)

# Entrenar el modelo
x_train <- matrix(c(0,0,1,0,1,0,1,1,1,1,0,1,0,1,1,0), nrow = 4, byrow = TRUE)
y_train <- matrix(c(1,0,0,0,1,0,0,0,1,0,0,1), nrow = 4, byrow = TRUE)
model %>% keras::fit(x_train, y_train, epochs = 10)

# Hacer una predicción con el modelo entrenado
x_test <- matrix(c(1,0,0,1,0,1,0,0), nrow = 2, byrow = TRUE)
y_pred <- model %>% keras::predict(x_test)
print(y_pred)

