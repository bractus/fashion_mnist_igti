from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st


def get_model(size, activation, optimizer, loss, metrics):
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28,28)),
        keras.layers.Dense(size, activation=activation),
        keras.layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    return model


(train_images, train_labels), (test_images, test_labels) = keras.datasets.fashion_mnist.load_data()

class_names = ['Tshirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 
               'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

class_dict = {'Tshirt/top':0, 'Trouser':1, 'Pullover':2, 'Dress':3, 'Coat':4, 'Sandal':5, 
               'Shirt':6, 'Sneaker':7, 'Bag':8, 'Ankle boot':9}


# activation = 'relu'
# optimizer = 'adam'
# loss = 'sparse_categorical_crossentropy'
# metrics = ['accuracy']
# cmd_oculta = 128
# model = get_model(cmd_oculta, activation, optimizer, loss, metrics)

def main():

    opt_menu = st.sidebar.radio('Navegação', ('1. Analisar dados', '2. Treinar modelo', 
                                '3. Prever valores', '4. Sobre'), 3)

    if opt_menu == '4. Sobre':
        st.title('Análise - Fashion MNIST')
        st.image('fashion.jpeg')
        st.subheader('Por: Cairo Rocha (https://github.com/bractus)')

        st.markdown('Esse projeto tem como objetivo mostrar de forma interativa o treinamento \
                 e previsão de valores utilizando uma rede neural convolucional implementada \
                 em [Tensorflow 2.0](https://www.tensorflow.org/)')

    elif opt_menu == '1. Analisar dados':

        st.header('1. Analisar dados')
    
        st.write('Tamanho do conjunto de treino: ', train_images.shape[0])
        st.write('Tamanho do conjunto de testes: ', test_images.shape[0])

        st.subheader('1.1. Visualizar imagens')

        opt_classe = st.multiselect('Selecione a classe', class_names, 
                                    default=class_names)

        selected_classes = [class_dict[o] for o in opt_classe]
        selected_classes = np.isin(train_labels, selected_classes)
        selected_classes = np.where(selected_classes)[0]

        slider_st3 = st.slider('Tamanho do grid: ', 1, 5, 3)

        fig = plt.figure(figsize=(10, 10))
        for i in range(slider_st3**2):
            plt.subplot(slider_st3, slider_st3, i+1)
            plt.xticks([])
            plt.yticks([])
            plt.grid(False)
            r = np.random.choice(selected_classes)
            plt.imshow(train_images[r], cmap=plt.cm.binary)
            plt.xlabel(class_names[train_labels[r]])
        st.pyplot(fig)

    elif opt_menu == '2. Treinar modelo':

        st.header('2. Treinar modelo')

        slider_st = st.slider('Quantidade de épocas: ', 1, 100, 10)
        cmd_oculta = st.number_input('Quantos neurônios na camada oculta? : ', value=128) 
        activation = st.selectbox('Ativação da camada oculta', ('relu', 'sigmoid', 'softmax',
                                  'softplus', 'softsign', 'tanh', 'selu', 'elu', 'exponential',))
        optimizer = st.selectbox('Otimizador', ('adam', 'RMSProp', 'SGD'))
        loss = st.selectbox('Perda', ('sparse_categorical_crossentropy', 'poisson', 'categorical_hinge'))
        metrics = st.multiselect('Métricas', ('accuracy', 'categorical_hinge'), default=['accuracy'])
       
        btn = st.button("Treinar")
        if btn:

            model = get_model(cmd_oculta, activation, optimizer, loss, metrics)

            my_bar = st.progress(0)
            los = []
            acc = []
            hin = []
            for i in range(slider_st):
                history = model.fit(train_images, train_labels, epochs=1)
                txt = '%d - Perda %.4f' % (i+1, history.history['loss'][0])
                if 'accuracy' in metrics:
                    txt += ' - Acurácia %.4f' % (history.history['accuracy'][0])
                    acc.append(history.history['accuracy'][0])

                if 'categorical_hinge' in metrics:
                    txt += ' - Categorical Hinge %.4f' % (history.history['categorical_hinge'][0])
                    hin.append(history.history['categorical_hinge'][0])

                st.write(txt)
                my_bar.progress(i/slider_st)
                los.append(history.history['loss'][0])

            model.save('model.tf')
            model.save_weights('model.h5')

            my_bar.progress(100)
            st.success('Modelo treinado com sucesso!')

            test = model.evaluate(test_images, test_labels)
            st.write('Perda (Teste)', round(test[0], 4))
            index = 1
            if 'accuracy' in metrics:
                st.write('Acurácia (Teste)', round(test[index], 4))
                index += 1
            if 'categorical_hinge' in metrics:
                st.write('Categorical Hinge  (Teste)', round(test[index], 4))

            st.subheader("Perda (Treino)")
            st.line_chart(los)
            if 'accuracy' in metrics:
                st.subheader("Acurácia (Treino)")
                st.line_chart(acc)
            if 'categorical_hinge' in metrics:
                st.subheader("Categorical Hinge (Treino)")
                st.line_chart(hin)

    elif opt_menu == '3. Prever valores':

        st.header('3. Prever valores')

        slider_st2 = st.slider('Número da imagem: ', 0, test_images.shape[0]-1, 0)

        fig = plt.figure(figsize=(5, 5))
        plt.grid(False)
        plt.imshow(test_images[slider_st2], cmap=plt.cm.binary)
        plt.xlabel(class_names[test_labels[slider_st2]])
        st.pyplot(fig)

        btn2 = st.button('Prever')

        if btn2:
            try:
                model = keras.models.load_model('model.tf')
                model.load_weights('model.h5')

                img = np.expand_dims(test_images[slider_st2], 0)
                predictions = model.predict(img)

                pred_label = predictions.argmax()
                pct = predictions.max()
                pct = str((pct*100).round(0))
                st.write("Previsto: ", class_names[pred_label], " Porcentagem: ", pct)

                fig = plt.figure(figsize=(5, 5))
                colors = ['red']*10
                colors[pred_label] = 'green'
                plt.bar(class_names, predictions[0], color=colors)
                plt.xticks(rotation=45)
                st.pyplot(fig)
            except:
                st.error("Modelo não foi definido! Treine-o antes!")

if __name__ == '__main__':

    main()
