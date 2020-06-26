import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st


@st.cache
def load_data():
    return keras.datasets.fashion_mnist.load_data()

# @st.cache
def get_model(size, activation, optimizer, loss, metrics):
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28,28)),
        keras.layers.Dense(size, activation='relu'),
        keras.layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    return model


# @st.cache
def train_model(model, epochs, X, Y):
    history = model.fit(X, Y, epochs=epochs)
    return history, model


class_names = ['Tshirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 
               'Bag', 'Ankle boot']

(train_images, train_labels), (test_images, test_labels) = load_data()

st.title('Análise - Fashion MNIST')
st.subheader('Por: Cairo Rocha (https://github.com/bractus)')


st.header('1. Analisar dados')

st.write('Tamanho do conjunto de treino: ', train_images.shape[0])
st.write('Tamanho do conjunto de testes: ', test_images.shape[0])

st.subheader('1.1. Visualizar imagens')

slider_st3 = st.slider('Tamanho do grid: ', 1, 5, 1)

fig = plt.figure(figsize=(10,10))
for i in range(slider_st3**2):
    plt.subplot(slider_st3, slider_st3, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    r = np.random.randint(0, train_images.shape[0])
    plt.imshow(train_images[r], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[r]])
st.pyplot(fig)

st.header('2. Treinar modelo')

slider_st = st.slider('Quantidade de épocas: ', 1, 100, 10)
cmd_oculta = st.number_input('Quantos neurônios na camada oculta? : ', value=128) 
# ativacao = 
# otimizador = 
# loss = 
# metrica = 

btn = st.button("Treinar")

model = get_model(cmd_oculta, 'relu', 'adam', 'sparse_categorical_crossentropy', ['accuracy'] )

if btn:

    my_bar = st.progress(0)
    for i in range(slider_st):
        history, model = train_model(model, 1, train_images, train_labels)
        st.write('Perda', round(history.history['loss'][0], 2), 'Acurácia', round(history.history['accuracy'][0], 2))
        my_bar.progress(i/slider_st)
    st.success('Modelo treinado com sucesso!')

    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose = 2)
    st.write('Perda (Conjunto de teste)', round(test_loss, 2), 'Acurácia (Conjunto de teste)', round(test_acc, 2))

st.subheader('2.1. Prever valores')

slider_st2 = st.slider('Número da imagem: ', 1, test_images.shape[0], np.random.randint(0, test_images.shape[0]))

plt.grid(False)
plt.imshow(test_images[slider_st2], cmap=plt.cm.binary)
plt.xlabel(class_names[test_labels[slider_st2]])
st.pyplot(fig)

btn2 = st.button('Prever')

if btn2 and model:
    img = np.expand_dims(test_images[slider_st2], 0)
    predictions = model.predict(img)
    pred_label = predictions.argmax()
    pct = predictions.max()
    pct = str((pct*100).round(0))
    st.write("Previsto: ", class_names[pred_label], " Porcentagem: ", pct)

elif btn2 and not model:
    st.error("Modelo não está treinado!")
